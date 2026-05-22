//! **Gemma 3 (text) architecture support.**
//!
//! Gemma 3 `Gemma3ForCausalLM` (`model_type = gemma3_text`) is the
//! Gemma 2 transformer with three deltas:
//!
//! 1. **Per-head QK-Norm** — an extra RMSNorm applied to the q / k
//!    projections after the reshape-to-heads and before RoPE, with
//!    γ of shape `[head_dim]` (`*.self_attn.{q,k}_norm.weight`).
//!    Topologically identical to Qwen3's QK-Norm. The
//!    `1/√query_pre_attn_scalar` attention scale is absorbed into
//!    the `k_norm` γ (a pre-normalize scale on k_proj would be
//!    stripped by the K-Norm RMSNorm), folded after the Gemma
//!    `(1+γ)` term so the loaded γ is `(1+γ)·scale`.
//!
//! 2. **No soft-cap** — Gemma 3 dropped Gemma 2's attention-logit
//!    and final-logit soft-caps (`attn`/`final_logit_softcapping`
//!    are `None`). The builder's `match Some/None` simply skips
//!    the `SoftCap` node, identical to the Gemma 2 builder's
//!    behaviour when the caps are absent.
//!
//! 3. **Dual RoPE base frequency** — Gemma 3 alternates *local*
//!    (sliding-window) and *global* (full-attention) layers and
//!    gives them different RoPE bases: layer `i` is global when
//!    `(i + 1) % sliding_window_pattern == 0` and uses
//!    `rope_theta` (`1_000_000`); otherwise it is local and uses
//!    `rope_local_base_freq` (`10_000`). See [`layer_rope_theta`].
//!
//! Everything else is Gemma 2: dual-norm per block, `(1+γ)·rms(x)`
//! RMSNorm, GeGLU FFN, `× sqrt(hidden_size)` embedding scale, tied
//! LM head.
//!
//! ## Sliding-window attention (deferred)
//!
//! Like the Gemma 2 builder, this ships full causal attention on
//! every layer. For any prompt + generation length below the
//! window (`sliding_window = 512` for Gemma 3 1B) sliding-window
//! is mathematically equivalent to full causal attention — the
//! per-layer alternation is invisible to the output. The dual RoPE
//! base frequency, however, **is** applied per layer because it
//! changes the result at any context length.
//!
//! No new AMG ops are introduced — the graph uses the existing
//! `RmsNorm`, `BroadcastMul`, `RoPE`, `MatMul`, `Softmax`,
//! `Reshape`, `Permute`, `Concat`, `GeLU` set.

use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::{KvCacheBuildSpec, KvCacheHandles, KvLayerHandle};
use crate::amg::weight_store::{SharedParam, WeightStore};
use crate::model_adapters::tensor_spec::{GEMMA3_SPEC, TransformParams, resolve_transforms};
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::tensor::Tensor;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::{LoadTransform, WeightMapper};

/// Handles produced by [`build_gemma3`]. Same shape as the other
/// builders' handle structs.
pub struct Gemma3Handles {
    pub token_input_id: usize,
    pub logits_id: usize,
    pub param_ids: Vec<usize>,
    pub param_names: Vec<String>,
}

/// RoPE base frequency for layer `layer_idx`. Gemma 3 alternates
/// local (sliding-window) and global (full-attention) layers:
/// layer `i` is **global** when `(i + 1) % sliding_window_pattern
/// == 0` and uses `rope_theta`; otherwise it is **local** and uses
/// `rope_local_base_freq`. When the dual-RoPE metadata is absent
/// (`sliding_window_pattern` / `rope_local_base_freq` are `None` —
/// e.g. a malformed config) the function falls back to the single
/// `rope_theta`, which is the safe Gemma-2-equivalent behaviour.
pub fn layer_rope_theta(config: &LlamaConfig, layer_idx: usize) -> u32 {
    match (config.sliding_window_pattern, config.rope_local_base_freq) {
        (Some(pattern), Some(local_freq)) if pattern > 0 => {
            if (layer_idx as u32 + 1) % pattern == 0 {
                config.rope_theta
            } else {
                local_freq
            }
        }
        _ => config.rope_theta,
    }
}

fn register_param_owned(
    gb: &mut GraphBuilder,
    full_name: &str,
    shape: Vec<usize>,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> usize {
    let numel: usize = shape.iter().product();
    let tensor = Tensor::new_cpu(shape, vec![0.0_f32; numel]);
    let node_id = gb.parameter(tensor);
    gb.set_node_debug_name(node_id, full_name);
    param_ids.push(node_id);
    param_names.push(full_name.to_string());
    node_id
}

/// Register a graph-local mutable F32 zero buffer (KV cache slot —
/// runtime-owned, must NOT appear in `param_names`).
fn register_local_zero_f32(gb: &mut GraphBuilder, shape: Vec<usize>) -> usize {
    let numel: usize = shape.iter().product();
    gb.parameter(Tensor::new_cpu(shape, vec![0.0_f32; numel]))
}

/// Apply the Gemma embedding scale `× sqrt(hidden_size)`.
fn embed_scale_gemma3(gb: &mut GraphBuilder, x_embed: usize, config: &LlamaConfig) -> usize {
    let scale = (config.hidden_size as f32).sqrt();
    let scale_id = gb.parameter(Tensor::new_cpu(vec![1, 1, 1], vec![scale]));
    gb.broadcast_mul(x_embed, scale_id)
}

/// Per-layer Gemma 3 transformer block. The `register` closure
/// abstracts no-store (owned zero slots) vs with-store (Arc-shared
/// tensors), exactly as in the Gemma 2 builder.
#[allow(clippy::too_many_arguments)]
fn build_block_gemma3<R>(
    gb: &mut GraphBuilder,
    layer_idx: usize,
    x_residual_outer: usize,
    causal_mask_id: usize,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    kv_cache: Option<&KvCacheBuildSpec>,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
    kv_handles_out: Option<&mut Vec<KvLayerHandle>>,
    register: &mut R,
) -> Result<usize, BuildError>
where
    R: FnMut(
        &mut GraphBuilder,
        &str,
        Vec<usize>,
        &mut Vec<usize>,
        &mut Vec<String>,
    ) -> Result<usize, BuildError>,
{
    let prefix = format!("model.layers.{}", layer_idx);
    let hidden = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let head_dim = config.effective_head_dim();
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;
    let bs = (batch * seq) as isize;
    let q_dim = n_heads * head_dim;

    // ---- 1. Input layernorm (PRE-attention) ----
    let input_ln_gamma = register(
        gb,
        &format!("{prefix}.input_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let h_normed = gb.rms_norm(x_residual_outer, config.rms_norm_eps);
    let h = gb.broadcast_mul(h_normed, input_ln_gamma);

    // ---- 2. Q/K/V projections (K/V registered post-tile, like Gemma 2) ----
    let q_proj_w = register(
        gb,
        &format!("{prefix}.self_attn.q_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;
    let k_proj_w = register(
        gb,
        &format!("{prefix}.self_attn.k_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;
    let v_proj_w = register(
        gb,
        &format!("{prefix}.self_attn.v_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;

    let h_flat = gb.reshape(h, vec![bs, hidden as isize]);
    let q_flat = gb.matmul(h_flat, q_proj_w);
    let k_flat = gb.matmul(h_flat, k_proj_w);
    let v_flat = gb.matmul(h_flat, v_proj_w);

    // ---- 3. Multi-head reshape ----
    let split_shape = vec![
        batch as isize,
        seq as isize,
        n_heads as isize,
        head_dim as isize,
    ];
    let q = gb.reshape(q_flat, split_shape.clone());
    let k = gb.reshape(k_flat, split_shape.clone());
    let v = gb.reshape(v_flat, split_shape);

    // ---- 4. Gemma 3 QK-Norm: per-head RMSNorm on Q and K
    //         (γ shape [1,1,1,head_dim], broadcast over heads).
    //         The 1/√query_pre_attn_scalar attention scale lives in
    //         the k_norm γ via the load transform. ----
    let q_norm_gamma = register(
        gb,
        &format!("{prefix}.self_attn.q_norm.weight"),
        vec![1, 1, 1, head_dim],
        param_ids,
        param_names,
    )?;
    let k_norm_gamma = register(
        gb,
        &format!("{prefix}.self_attn.k_norm.weight"),
        vec![1, 1, 1, head_dim],
        param_ids,
        param_names,
    )?;
    let q_rmsnormed = gb.rms_norm(q, config.rms_norm_eps);
    let q_normed = gb.broadcast_mul(q_rmsnormed, q_norm_gamma);
    let k_rmsnormed = gb.rms_norm(k, config.rms_norm_eps);
    let k_normed = gb.broadcast_mul(k_rmsnormed, k_norm_gamma);

    // ---- 5. RoPE with the per-layer base frequency (local vs
    //         global — see `layer_rope_theta`). ----
    let layer_theta = layer_rope_theta(config, layer_idx);
    let position_offset: u32 = kv_cache.map(|spec| spec.cached_len as u32).unwrap_or(0);
    let q_rope = gb.rope_with_offset(q_normed, head_dim, layer_theta, position_offset);
    let k_rope = gb.rope_with_offset(k_normed, head_dim, layer_theta, position_offset);

    // ---- 6. [b, s, h, d] → [b, h, s, d] ----
    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let new_k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let new_v_perm = gb.permute(v, vec![0, 2, 1, 3]);

    // ---- 7. Cache concat (cache-aware path) ----
    let (k_full, v_full, layer_handle): (usize, usize, Option<KvLayerHandle>) = match kv_cache {
        None => (new_k_perm, new_v_perm, None),
        Some(spec) => {
            let cache_shape = vec![batch, n_heads, spec.cached_len, head_dim];
            let cache_k_id = register_local_zero_f32(gb, cache_shape.clone());
            let cache_v_id = register_local_zero_f32(gb, cache_shape);
            let k_full_id = gb.concat(cache_k_id, new_k_perm, 2);
            let v_full_id = gb.concat(cache_v_id, new_v_perm, 2);
            let handle = KvLayerHandle {
                cache_k_param_id: cache_k_id,
                cache_v_param_id: cache_v_id,
                k_full_node_id: k_full_id,
                v_full_node_id: v_full_id,
            };
            (k_full_id, v_full_id, Some(handle))
        }
    };

    // ---- 8. Attention: scores → (optional softcap, None for
    //         Gemma 3) → mask → softmax → AV. The attention scale
    //         is already folded into k_norm γ. ----
    let k_full_t = gb.transpose_last_two(k_full);
    let scores_raw = gb.batch_matmul(q_perm, k_full_t);
    let scores = match config.attn_logit_softcapping {
        Some(cap) => gb.soft_cap(scores_raw, cap),
        None => scores_raw,
    };
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);
    let attn_weights = gb.softmax(scores_masked);
    let attn_out = gb.batch_matmul(attn_weights, v_full);
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    // ---- 9. Output projection ----
    let attn_out_flat = gb.reshape(attn_out_back, vec![bs, q_dim as isize]);
    let o_proj_w = register(
        gb,
        &format!("{prefix}.self_attn.o_proj.weight"),
        vec![q_dim, hidden],
        param_ids,
        param_names,
    )?;
    let attn_proj_flat = gb.matmul(attn_out_flat, o_proj_w);
    let attn_proj = gb.reshape(
        attn_proj_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 10. Post-attention layernorm (on the attention OUTPUT,
    //          before the residual add — Gemma dual-norm) ----
    let post_attn_ln_gamma = register(
        gb,
        &format!("{prefix}.post_attention_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let attn_proj_normed_pre = gb.rms_norm(attn_proj, config.rms_norm_eps);
    let attn_proj_normed = gb.broadcast_mul(attn_proj_normed_pre, post_attn_ln_gamma);

    // ---- 11. First residual ----
    let x_residual_1 = gb.add(x_residual_outer, attn_proj_normed);

    // ---- 12. Pre-feedforward layernorm ----
    let pre_ffn_ln_gamma = register(
        gb,
        &format!("{prefix}.pre_feedforward_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let h2_normed_pre = gb.rms_norm(x_residual_1, config.rms_norm_eps);
    let h2 = gb.broadcast_mul(h2_normed_pre, pre_ffn_ln_gamma);

    // ---- 13. GeGLU FFN ----
    let gate_proj_w = register(
        gb,
        &format!("{prefix}.mlp.gate_proj.weight"),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    )?;
    let up_proj_w = register(
        gb,
        &format!("{prefix}.mlp.up_proj.weight"),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    )?;
    let down_proj_w = register(
        gb,
        &format!("{prefix}.mlp.down_proj.weight"),
        vec![intermediate, hidden],
        param_ids,
        param_names,
    )?;
    let h2_flat = gb.reshape(h2, vec![bs, hidden as isize]);
    let gate_flat = gb.matmul(h2_flat, gate_proj_w);
    let up_flat = gb.matmul(h2_flat, up_proj_w);
    let gelu_gate_flat = gb.gelu(gate_flat);
    let ffn_pre_down_flat = gb.mul(gelu_gate_flat, up_flat);
    let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
    let ffn_out = gb.reshape(
        ffn_out_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 14. Post-feedforward layernorm (Gemma dual-norm) ----
    let post_ffn_ln_gamma = register(
        gb,
        &format!("{prefix}.post_feedforward_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let ffn_out_normed_pre = gb.rms_norm(ffn_out, config.rms_norm_eps);
    let ffn_out_normed = gb.broadcast_mul(ffn_out_normed_pre, post_ffn_ln_gamma);

    // ---- 15. Second residual ----
    let layer_out = gb.add(x_residual_1, ffn_out_normed);

    if let (Some(handles_out), Some(handle)) = (kv_handles_out, layer_handle) {
        handles_out.push(handle);
    }
    Ok(layer_out)
}

/// Build the head: final RmsNorm, tied/untied LM head, optional
/// final-logit SoftCap (`None` for Gemma 3), output reshape.
fn build_head_gemma3<R>(
    gb: &mut GraphBuilder,
    x: usize,
    embed_w: usize,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
    register: &mut R,
) -> Result<usize, BuildError>
where
    R: FnMut(
        &mut GraphBuilder,
        &str,
        Vec<usize>,
        &mut Vec<usize>,
        &mut Vec<String>,
    ) -> Result<usize, BuildError>,
{
    let final_ln_gamma = register(
        gb,
        "model.norm.weight",
        vec![1, 1, config.hidden_size],
        param_ids,
        param_names,
    )?;
    let x_normed = gb.rms_norm(x, config.rms_norm_eps);
    let x_final = gb.broadcast_mul(x_normed, final_ln_gamma);

    let bs = (runtime.batch * runtime.seq) as isize;
    let x_flat = gb.reshape(x_final, vec![bs, config.hidden_size as isize]);
    let logits_flat_raw = if config.tie_word_embeddings {
        gb.matmul_rhs_transposed(x_flat, embed_w)
    } else {
        let lm_head_input = register(
            gb,
            "lm_head.weight",
            vec![config.hidden_size, config.vocab_size],
            param_ids,
            param_names,
        )?;
        gb.matmul(x_flat, lm_head_input)
    };
    let logits_flat = match config.final_logit_softcapping {
        Some(cap) => gb.soft_cap(logits_flat_raw, cap),
        None => logits_flat_raw,
    };
    let logits = gb.reshape(
        logits_flat,
        vec![
            runtime.batch as isize,
            runtime.seq as isize,
            config.vocab_size as isize,
        ],
    );
    Ok(logits)
}

/// Build the no-store Gemma 3 graph (zero-init parameter slots the
/// loader populates via the weight mapper).
pub fn build_gemma3(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
) -> Gemma3Handles {
    config
        .validate()
        .expect("invalid LlamaConfig (Gemma 3 path)");

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();

    let seq = runtime.seq;
    let mut mask_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    let embed_w = register_param_owned(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let x_embed = gb.index_select(embed_w, token_input_id);
    let mut x = embed_scale_gemma3(gb, x_embed, config);

    let mut register = |gb: &mut GraphBuilder,
                        full_name: &str,
                        shape: Vec<usize>,
                        ids: &mut Vec<usize>,
                        names: &mut Vec<String>|
     -> Result<usize, BuildError> {
        Ok(register_param_owned(gb, full_name, shape, ids, names))
    };
    for layer_idx in 0..config.num_hidden_layers {
        x = build_block_gemma3(
            gb,
            layer_idx,
            x,
            causal_mask_id,
            config,
            runtime,
            None,
            &mut param_ids,
            &mut param_names,
            None,
            &mut register,
        )
        .expect("no-store Gemma 3 register closure cannot fail");
    }

    let logits = build_head_gemma3(
        gb,
        x,
        embed_w,
        config,
        runtime,
        &mut param_ids,
        &mut param_names,
        &mut register,
    )
    .expect("no-store Gemma 3 register closure cannot fail");

    Gemma3Handles {
        token_input_id,
        logits_id: logits,
        param_ids,
        param_names,
    }
}

/// Build the store-backed, cache-aware Gemma 3 graph.
pub fn build_gemma3_with_store(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
    store: &WeightStore,
    kv_cache: Option<&KvCacheBuildSpec>,
) -> Result<LlamaHandlesShared, BuildError> {
    config
        .validate()
        .expect("invalid LlamaConfig (Gemma 3 path)");

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();
    let mut kv_handles_inner: Vec<KvLayerHandle> = Vec::new();

    let cached_len = kv_cache.map(|s| s.cached_len).unwrap_or(0);
    let seq = runtime.seq;
    let total_kv_seq = seq + cached_len;
    let mut mask_data = vec![0.0_f32; seq * total_kv_seq];
    for i in 0..seq {
        for j in 0..total_kv_seq {
            if j >= cached_len && (j - cached_len) > i {
                mask_data[i * total_kv_seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, total_kv_seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    let mut register = |gb: &mut GraphBuilder,
                        full_name: &str,
                        expected_shape: Vec<usize>,
                        ids: &mut Vec<usize>,
                        names: &mut Vec<String>|
     -> Result<usize, BuildError> {
        let p: &SharedParam =
            store
                .get_by_name(full_name)
                .ok_or_else(|| BuildError::MissingParameter {
                    name: full_name.to_string(),
                })?;
        if p.shape() != expected_shape.as_slice() {
            return Err(BuildError::ParameterShapeMismatch {
                name: full_name.to_string(),
                expected: expected_shape,
                got: p.shape().to_vec(),
            });
        }
        let node_id = gb.parameter(p.to_tensor());
        gb.set_node_debug_name(node_id, full_name);
        ids.push(node_id);
        names.push(full_name.to_string());
        Ok(node_id)
    };

    let embed_w = register(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let x_embed = gb.index_select(embed_w, token_input_id);
    let mut x = embed_scale_gemma3(gb, x_embed, config);

    for layer_idx in 0..config.num_hidden_layers {
        let kv_out_ref: Option<&mut Vec<KvLayerHandle>> = if kv_cache.is_some() {
            Some(&mut kv_handles_inner)
        } else {
            None
        };
        x = build_block_gemma3(
            gb,
            layer_idx,
            x,
            causal_mask_id,
            config,
            runtime,
            kv_cache,
            &mut param_ids,
            &mut param_names,
            kv_out_ref,
            &mut register,
        )?;
    }

    let logits = build_head_gemma3(
        gb,
        x,
        embed_w,
        config,
        runtime,
        &mut param_ids,
        &mut param_names,
        &mut register,
    )?;

    let kv_handles = kv_cache.map(|_| KvCacheHandles {
        per_layer: kv_handles_inner,
    });

    Ok(LlamaHandlesShared {
        token_input_id,
        logits_id: logits,
        param_ids,
        param_names,
        kv_handles,
    })
}

/// Build the [`WeightMapper`] for a Gemma 3 HF (safetensors)
/// checkpoint. Drives per-name transforms from
/// `GEMMA3_SPEC.hf_transforms`.
pub fn gemma3_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let head_dim = config.effective_head_dim();
    let kv_groups = config.kv_groups();
    let qpas = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
    let attention_scale = 1.0_f32 / qpas.sqrt();
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms =
            gemma3_transforms_for_name(name, hidden_size, head_dim, kv_groups, attention_scale);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

/// Per-name Gemma 3 HF transform list (drives `GEMMA3_SPEC.hf_transforms`).
pub fn gemma3_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    resolve_transforms(
        GEMMA3_SPEC.hf_transforms,
        name,
        &TransformParams {
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        },
    )
}

/// Build the [`WeightMapper`] for a Gemma 3 GGUF checkpoint.
/// Drives `GEMMA3_SPEC.gguf_transforms` — identical to the HF
/// table except the RMSNorm `+1` fold is dropped (llama.cpp
/// pre-folds it into the Gemma GGUF norm weights).
pub fn gemma3_gguf_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let head_dim = config.effective_head_dim();
    let kv_groups = config.kv_groups();
    let qpas = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
    let attention_scale = 1.0_f32 / qpas.sqrt();
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms =
            gemma3_gguf_transforms_for_name(name, hidden_size, head_dim, kv_groups, attention_scale);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

/// Per-name Gemma 3 GGUF transform list (drives `GEMMA3_SPEC.gguf_transforms`).
pub fn gemma3_gguf_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    resolve_transforms(
        GEMMA3_SPEC
            .gguf_transforms
            .expect("GEMMA3_SPEC carries a GGUF-specific transform table"),
        name,
        &TransformParams {
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        },
    )
}
