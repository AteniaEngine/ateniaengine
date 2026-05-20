//! **Phase Q (Qwen3 family support).** Graph builder and weight
//! mapper for the Qwen3 family (`Qwen3ForCausalLM` /
//! `model_type=qwen3`).
//!
//! Qwen3 is a Llama-topology family with two attention-block deltas:
//!
//! 1. **Per-head QK-Norm** before RoPE: an extra RMSNorm applied to
//!    the q / k projections after the reshape-to-heads, with γ of
//!    shape `[head_dim]`. The 1 / √head_dim attention scale is
//!    absorbed into the `k_norm.weight` γ via the load-transform
//!    (`ReshapeHeadDim4D` + `ScaleAttn`), preserving Atenia's
//!    "scale lives in the weights" convention; this is sound
//!    because the per-head γ is the *output* of the RMSNorm
//!    (`y = γ · x̂`), so a constant multiplied into γ survives the
//!    normalization (which would otherwise strip a pre-normalize
//!    scale).
//! 2. **No QKV biases** (`attention_bias=false`).
//!
//! Plus a checkpoint-layout quirk: Qwen3 safetensors physically
//! stores `lm_head.weight` even when `tie_word_embeddings=true`.
//! `build_qwen3` therefore always registers a separate `lm_head`
//! parameter (independent of the config tie flag), which loads
//! straight from the checkpoint without needing a completeness-
//! gate change. The math is unaffected — when the checkpoint's
//! `lm_head.weight` equals `embed_tokens.weight` (the standard
//! tied case) the result is identical to the tied path.
//!
//! No new AMG ops are introduced. The graph uses the existing
//! `RmsNorm`, `BroadcastMul`, `RoPE`, `MatMul`, `Softmax`,
//! `Reshape`, `Permute`, `Concat` set.
//!
//! Q-3 of the Phase Q microplan.

use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::{KvCacheBuildSpec, KvCacheHandles, KvLayerHandle};
use crate::amg::weight_store::{SharedParam, WeightStore};
use crate::nn::llama::builder::{LlamaHandles, LlamaRuntime};
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::tensor::Tensor;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::{LoadTransform, WeightMapper};

// ============================================================
// Local helpers (cloned from builder.rs / builder_shared.rs).
// Kept private and Qwen3-only so this module remains self
// contained, matching the Phi-3 / Gemma 2 pattern.
// ============================================================

fn register_param(
    gb: &mut GraphBuilder,
    full_name: &str,
    shape: Vec<usize>,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> usize {
    let numel: usize = shape.iter().product();
    let data = vec![0.0_f32; numel];
    let tensor = Tensor::new_cpu(shape, data);
    let node_id = gb.parameter(tensor);
    gb.set_node_debug_name(node_id, full_name);
    param_ids.push(node_id);
    param_names.push(full_name.to_string());
    node_id
}

fn register_param_from_store(
    gb: &mut GraphBuilder,
    store: &WeightStore,
    full_name: &str,
    expected_shape: Vec<usize>,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> Result<usize, BuildError> {
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
    param_ids.push(node_id);
    param_names.push(full_name.to_string());
    Ok(node_id)
}

fn register_local_zero_f32(gb: &mut GraphBuilder, shape: Vec<usize>) -> usize {
    let numel: usize = shape.iter().product();
    let data = vec![0.0_f32; numel];
    let tensor = Tensor::new_cpu(shape, data);
    gb.parameter(tensor)
}

// ============================================================
// Scratch-graph block (clone of build_transformer_block_llama
// with the Qwen3 deltas: q_dim from head_dim, no biases, QK-Norm,
// o_proj input dim = q_dim).
// ============================================================

#[allow(clippy::too_many_arguments)]
fn build_transformer_block_qwen3(
    gb: &mut GraphBuilder,
    layer_idx: usize,
    x: usize,
    causal_mask_id: usize,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> usize {
    let prefix = format!("model.layers.{}", layer_idx);
    let hidden = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let head_dim = config.effective_head_dim();
    let q_dim = n_heads * head_dim; // Qwen3: hidden != q_dim in general.
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;
    let bs = (batch * seq) as isize;

    // ---- 1. Input layernorm ----
    let input_ln_gamma = register_param(
        gb,
        &format!("{}.input_layernorm.weight", prefix),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    );
    let h_normed = gb.rms_norm(x, config.rms_norm_eps);
    let h = gb.broadcast_mul(h_normed, input_ln_gamma);

    // ---- 2. Q/K/V projections (post weight-mapper TileGroupedDim:
    //         all three end up at [hidden, q_dim]). NO biases. ----
    let q_proj_w = register_param(
        gb,
        &format!("{}.self_attn.q_proj.weight", prefix),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    );
    let k_proj_w = register_param(
        gb,
        &format!("{}.self_attn.k_proj.weight", prefix),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    );
    let v_proj_w = register_param(
        gb,
        &format!("{}.self_attn.v_proj.weight", prefix),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    );

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

    // ---- 4. **Qwen3 QK-Norm**: per-head RMSNorm on Q and K
    //         (γ shape [1,1,1,head_dim], broadcast over heads).
    //         The 1/√head_dim attention scale lives in `k_norm.weight`
    //         γ via the load-transform (`ReshapeHeadDim4D` +
    //         `ScaleAttn`); a pre-normalize scale would be stripped
    //         by RMSNorm, but a post-normalize γ scale survives. ----
    let q_norm_gamma = register_param(
        gb,
        &format!("{}.self_attn.q_norm.weight", prefix),
        vec![1, 1, 1, head_dim],
        param_ids,
        param_names,
    );
    let k_norm_gamma = register_param(
        gb,
        &format!("{}.self_attn.k_norm.weight", prefix),
        vec![1, 1, 1, head_dim],
        param_ids,
        param_names,
    );
    let q_rmsnormed = gb.rms_norm(q, config.rms_norm_eps);
    let q_normed = gb.broadcast_mul(q_rmsnormed, q_norm_gamma);
    let k_rmsnormed = gb.rms_norm(k, config.rms_norm_eps);
    let k_normed = gb.broadcast_mul(k_rmsnormed, k_norm_gamma);

    // ---- 5. RoPE on normalized Q and K (V unchanged) ----
    let (q_rope, k_rope) = match config.effective_rope_scaling() {
        None => (
            gb.rope(q_normed, head_dim, config.rope_theta),
            gb.rope(k_normed, head_dim, config.rope_theta),
        ),
        Some(other) => panic!(
            "Qwen3 builder does not accept rope_scaling = {other:?} \
             (Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B all ship without)."
        ),
    };

    // ---- 6. [b, s, h, d] → [b, h, s, d] ----
    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let v_perm = gb.permute(v, vec![0, 2, 1, 3]);

    // ---- 7. Attention scores: Q @ K^T (4D BMM).
    //         The 1/√d scale is already absorbed into k_norm γ. ----
    let k_perm_t = gb.transpose_last_two(k_perm);
    let scores = gb.batch_matmul(q_perm, k_perm_t);

    // ---- 8. Causal mask ----
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);

    // ---- 9. Softmax ----
    let attn_weights = gb.softmax(scores_masked);

    // ---- 10. weights @ V ----
    let attn_out = gb.batch_matmul(attn_weights, v_perm);
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    // ---- 11. Output projection: [bs, q_dim] @ [q_dim, hidden] ----
    let attn_out_flat = gb.reshape(attn_out_back, vec![bs, q_dim as isize]);
    let o_proj_w = register_param(
        gb,
        &format!("{}.self_attn.o_proj.weight", prefix),
        vec![q_dim, hidden],
        param_ids,
        param_names,
    );
    let attn_proj_flat = gb.matmul(attn_out_flat, o_proj_w);
    let attn_proj = gb.reshape(
        attn_proj_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 12. Attention residual ----
    let x_residual_1 = gb.add(x, attn_proj);

    // ---- 13. Post-attention layernorm ----
    let post_ln_gamma = register_param(
        gb,
        &format!("{}.post_attention_layernorm.weight", prefix),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    );
    let h2_normed = gb.rms_norm(x_residual_1, config.rms_norm_eps);
    let h2 = gb.broadcast_mul(h2_normed, post_ln_gamma);

    // ---- 14. SwiGLU FFN (separate gate / up / down — not fused) ----
    let gate_proj_w = register_param(
        gb,
        &format!("{}.mlp.gate_proj.weight", prefix),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    );
    let up_proj_w = register_param(
        gb,
        &format!("{}.mlp.up_proj.weight", prefix),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    );
    let down_proj_w = register_param(
        gb,
        &format!("{}.mlp.down_proj.weight", prefix),
        vec![intermediate, hidden],
        param_ids,
        param_names,
    );

    let h2_flat = gb.reshape(h2, vec![bs, hidden as isize]);
    let gate_flat = gb.matmul(h2_flat, gate_proj_w);
    let up_flat = gb.matmul(h2_flat, up_proj_w);
    let silu_gate_flat = gb.silu(gate_flat);
    let ffn_pre_down_flat = gb.mul(silu_gate_flat, up_flat);
    let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
    let ffn_out = gb.reshape(
        ffn_out_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 15. FFN residual ----
    gb.add(x_residual_1, ffn_out)
}

/// Build the complete Qwen3 graph from scratch parameters
/// (zero-initialized; the caller populates them via the
/// `WeightMapper`). Mirrors `build_llama`.
pub fn build_qwen3(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
) -> LlamaHandles {
    config.validate().expect("invalid LlamaConfig (Qwen3 path)");

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

    let embed_w = register_param(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let mut x = gb.index_select(embed_w, token_input_id);

    for layer_idx in 0..config.num_hidden_layers {
        x = build_transformer_block_qwen3(
            gb,
            layer_idx,
            x,
            causal_mask_id,
            config,
            runtime,
            &mut param_ids,
            &mut param_names,
        );
    }

    let final_ln_gamma = register_param(
        gb,
        "model.norm.weight",
        vec![1, 1, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let x_normed = gb.rms_norm(x, config.rms_norm_eps);
    let x_final = gb.broadcast_mul(x_normed, final_ln_gamma);

    // **Qwen3 LM head:** always registered as a separate parameter,
    // even when `tie_word_embeddings=true`, because Qwen3
    // safetensors physically ship `lm_head.weight`. Tying the
    // graph would mark the checkpoint's `lm_head.weight` as
    // unexpected at the completeness gate. The math is unaffected
    // when the checkpoint stores tied weights (lm_head ==
    // embed_tokens).
    let bs = (runtime.batch * runtime.seq) as isize;
    let x_flat = gb.reshape(x_final, vec![bs, config.hidden_size as isize]);
    let lm_head_w = register_param(
        gb,
        "lm_head.weight",
        vec![config.hidden_size, config.vocab_size],
        &mut param_ids,
        &mut param_names,
    );
    let logits_flat = gb.matmul(x_flat, lm_head_w);
    let logits = gb.reshape(
        logits_flat,
        vec![runtime.batch as isize, seq as isize, config.vocab_size as isize],
    );

    let logits_id = gb.output(logits);

    LlamaHandles {
        token_input_id,
        logits_id,
        param_ids,
        param_names,
    }
}

// ============================================================
// Store-backed block + wrapper.
// ============================================================

#[allow(clippy::too_many_arguments)]
fn build_transformer_block_qwen3_with_store(
    gb: &mut GraphBuilder,
    layer_idx: usize,
    x: usize,
    causal_mask_id: usize,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    store: &WeightStore,
    kv_cache: Option<&KvCacheBuildSpec>,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
    kv_handles_out: Option<&mut Vec<KvLayerHandle>>,
) -> Result<usize, BuildError> {
    let prefix = format!("model.layers.{}", layer_idx);
    let hidden = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let head_dim = config.effective_head_dim();
    let q_dim = n_heads * head_dim;
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;
    let bs = (batch * seq) as isize;

    let input_ln_gamma = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.input_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let h_normed = gb.rms_norm(x, config.rms_norm_eps);
    let h = gb.broadcast_mul(h_normed, input_ln_gamma);

    let q_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.q_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;
    let k_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.k_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;
    let v_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.v_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;

    let h_flat = gb.reshape(h, vec![bs, hidden as isize]);
    let q_flat = gb.matmul(h_flat, q_proj_w);
    let k_flat = gb.matmul(h_flat, k_proj_w);
    let v_flat = gb.matmul(h_flat, v_proj_w);

    let split_shape = vec![
        batch as isize,
        seq as isize,
        n_heads as isize,
        head_dim as isize,
    ];
    let q = gb.reshape(q_flat, split_shape.clone());
    let k = gb.reshape(k_flat, split_shape.clone());
    let v = gb.reshape(v_flat, split_shape);

    let q_norm_gamma = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.q_norm.weight"),
        vec![1, 1, 1, head_dim],
        param_ids,
        param_names,
    )?;
    let k_norm_gamma = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.k_norm.weight"),
        vec![1, 1, 1, head_dim],
        param_ids,
        param_names,
    )?;
    let q_rmsnormed = gb.rms_norm(q, config.rms_norm_eps);
    let q_normed = gb.broadcast_mul(q_rmsnormed, q_norm_gamma);
    let k_rmsnormed = gb.rms_norm(k, config.rms_norm_eps);
    let k_normed = gb.broadcast_mul(k_rmsnormed, k_norm_gamma);

    let position_offset: u32 = kv_cache.map(|spec| spec.cached_len as u32).unwrap_or(0);
    let (q_rope, k_rope) = match config.effective_rope_scaling() {
        None => (
            gb.rope_with_offset(q_normed, head_dim, config.rope_theta, position_offset),
            gb.rope_with_offset(k_normed, head_dim, config.rope_theta, position_offset),
        ),
        Some(other) => panic!(
            "Qwen3 builder does not accept rope_scaling = {other:?}."
        ),
    };

    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let new_k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let new_v_perm = gb.permute(v, vec![0, 2, 1, 3]);

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

    let k_full_t = gb.transpose_last_two(k_full);
    let scores = gb.batch_matmul(q_perm, k_full_t);
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);
    let attn_weights = gb.softmax(scores_masked);
    let attn_out = gb.batch_matmul(attn_weights, v_full);
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    let attn_out_flat = gb.reshape(attn_out_back, vec![bs, q_dim as isize]);
    let o_proj_w = register_param_from_store(
        gb,
        store,
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
    let x_residual_1 = gb.add(x, attn_proj);

    let post_ln_gamma = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.post_attention_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let h2_normed = gb.rms_norm(x_residual_1, config.rms_norm_eps);
    let h2 = gb.broadcast_mul(h2_normed, post_ln_gamma);

    let gate_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.mlp.gate_proj.weight"),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    )?;
    let up_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.mlp.up_proj.weight"),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    )?;
    let down_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.mlp.down_proj.weight"),
        vec![intermediate, hidden],
        param_ids,
        param_names,
    )?;

    let h2_flat = gb.reshape(h2, vec![bs, hidden as isize]);
    let gate_flat = gb.matmul(h2_flat, gate_proj_w);
    let up_flat = gb.matmul(h2_flat, up_proj_w);
    let silu_gate_flat = gb.silu(gate_flat);
    let ffn_pre_down_flat = gb.mul(silu_gate_flat, up_flat);
    let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
    let ffn_out = gb.reshape(
        ffn_out_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    let layer_out = gb.add(x_residual_1, ffn_out);

    if let (Some(handles_out), Some(handle)) = (kv_handles_out, layer_handle) {
        handles_out.push(handle);
    }

    Ok(layer_out)
}

/// Build the complete Qwen3 graph backed by an `Arc`-shared
/// `WeightStore`, optionally wired for cache-aware decode.
/// Mirrors `build_llama_with_store`.
pub fn build_qwen3_with_store(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
    store: &WeightStore,
    kv_cache: Option<&KvCacheBuildSpec>,
) -> Result<LlamaHandlesShared, BuildError> {
    config.validate().expect("invalid LlamaConfig (Qwen3 path)");

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();
    let mut kv_handles: Vec<KvLayerHandle> = Vec::new();

    let seq = runtime.seq;
    let mut mask_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    let embed_w = register_param_from_store(
        gb,
        store,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let mut x = gb.index_select(embed_w, token_input_id);

    for layer_idx in 0..config.num_hidden_layers {
        let kv_out = if kv_cache.is_some() {
            Some(&mut kv_handles)
        } else {
            None
        };
        x = build_transformer_block_qwen3_with_store(
            gb,
            layer_idx,
            x,
            causal_mask_id,
            config,
            runtime,
            store,
            kv_cache,
            &mut param_ids,
            &mut param_names,
            kv_out,
        )?;
    }

    let final_ln_gamma = register_param_from_store(
        gb,
        store,
        "model.norm.weight",
        vec![1, 1, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let x_normed = gb.rms_norm(x, config.rms_norm_eps);
    let x_final = gb.broadcast_mul(x_normed, final_ln_gamma);

    let bs = (runtime.batch * runtime.seq) as isize;
    let x_flat = gb.reshape(x_final, vec![bs, config.hidden_size as isize]);
    let lm_head_w = register_param_from_store(
        gb,
        store,
        "lm_head.weight",
        vec![config.hidden_size, config.vocab_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let logits_flat = gb.matmul(x_flat, lm_head_w);
    let logits = gb.reshape(
        logits_flat,
        vec![runtime.batch as isize, seq as isize, config.vocab_size as isize],
    );

    let logits_id = gb.output(logits);

    let kv_handles_out = if kv_cache.is_some() {
        Some(KvCacheHandles {
            per_layer: kv_handles,
        })
    } else {
        None
    };

    Ok(LlamaHandlesShared {
        token_input_id,
        logits_id,
        param_ids,
        param_names,
        kv_handles: kv_handles_out,
    })
}

// ============================================================
// Weight mapper (Q-2 spec lookup).
// ============================================================

/// Build a `WeightMapper` for a Qwen3 HuggingFace checkpoint.
/// Drives per-name transforms from `QWEN3_SPEC.hf_transforms`
/// (see `model_adapters::tensor_spec`).
pub fn qwen3_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let head_dim = config.effective_head_dim();
    let kv_groups = config.kv_groups();
    let attention_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms =
            qwen3_transforms_for_name(name, hidden_size, head_dim, kv_groups, attention_scale);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

/// Per-name Qwen3 transform list. Driven by
/// `QWEN3_SPEC.hf_transforms` via the toolkit's `resolve_transforms`.
/// Public so unit tests can exercise dispatch directly.
pub fn qwen3_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    use crate::model_adapters::tensor_spec::{QWEN3_SPEC, TransformParams, resolve_transforms};
    resolve_transforms(
        QWEN3_SPEC.hf_transforms,
        name,
        &TransformParams {
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        },
    )
}
