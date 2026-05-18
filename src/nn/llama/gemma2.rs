//! **M11.C** — Google Gemma 2 architecture support.
//!
//! Gemma 2 (`Gemma2ForCausalLM`) is structurally a Llama-family
//! transformer with five architectural deltas:
//!
//! 1. **Dual-norm per layer** — every block has four RmsNorms
//!    (Llama has two): `input_layernorm` (pre-attention),
//!    `post_attention_layernorm` (on the attention output,
//!    *before* the residual add), `pre_feedforward_layernorm`
//!    (pre-FFN), `post_feedforward_layernorm` (on the FFN
//!    output, *before* the second residual add). The Llama
//!    pattern only normalises the running residual stream;
//!    Gemma 2 also normalises each sub-block's contribution.
//!
//! 2. **`(1 + γ) · rms(x)` RmsNorm** instead of Llama's
//!    `γ · rms(x)`. Folded into the loaded gamma at load time
//!    via `LoadTransform::AddScalar { scalar: 1.0 }` (M11.C
//!    step 3) so the runtime graph stays Llama-shape.
//!
//! 3. **Soft-cap** at two graph positions: `cap=50.0` on the
//!    pre-softmax attention scores (every layer) and `cap=30.0`
//!    on the final LM-head logits. Realised by `NodeType::SoftCap`
//!    (M11.C step 2). Both caps are conditional on the config
//!    fields `attn_logit_softcapping` / `final_logit_softcapping`
//!    being `Some(c)` — `None` skips the node.
//!
//! 4. **GeGLU FFN** — `gelu_tanh(gate) · up · W_down` instead of
//!    Llama's SwiGLU (`silu(gate) · up · W_down`). The existing
//!    `nn_act::gelu` is the tanh-approximation Gemma 2 expects
//!    (`hidden_act: "gelu_pytorch_tanh"`).
//!
//! 5. **Embedding scale** `× sqrt(hidden_size)` immediately after
//!    the embed lookup. For Gemma 2 2B this is `× 48.0` exactly
//!    (`sqrt(2304) = 48.0`). Realised as a graph-level
//!    `BroadcastMul` against a `[1,1,1]` constant — the scale
//!    cannot be folded into `embed_tokens.weight` because
//!    `lm_head.weight` is tied to it (`tie_word_embeddings = true`)
//!    and the scaled embed table would corrupt the LM-head output.
//!
//! ## Sliding-window attention (deferred)
//!
//! Gemma 2's reference implementation alternates between
//! sliding-window attention (`sliding_window = 4096`) on the
//! even-indexed layers and full attention on the odd-indexed
//! layers. For any prompt + generation length below the window
//! (≪ 4096 tokens) sliding-window is mathematically equivalent
//! to full causal attention — the per-layer alternation is
//! invisible to the output. The M11.C smoke-validation prompt
//! generates ≤ 50 tokens, so this builder ships full causal
//! attention on every layer and matches the reference
//! bit-exactly within the M11.C scope. Long-context support
//! (> 4096 tokens) lands as a follow-up that wires a sliding
//! mask through the attention executor.
//!
//! ## Q-side attention scale
//!
//! Gemma 2 scales the queries by `1 / sqrt(query_pre_attn_scalar)`
//! before the QK product, where `query_pre_attn_scalar` is a
//! config-supplied integer (`256` for Gemma 2 2B). The math is
//! `(Q / s) @ K^T = (Q @ K^T) / s = Q @ (K / s)^T`, so folding
//! the scale into K_proj rows yields a bit-identical result to
//! the reference and stays consistent with the Llama-family
//! mapper convention (which folds `1/sqrt(head_dim)` into K).
//! The mapper does this via `LoadTransform::Scale`. For Gemma 2
//! 2B `head_dim == query_pre_attn_scalar == 256`, so the K-fold
//! factor is `1/16` — numerically identical to the Llama path
//! when `head_dim = 256` would have been used.

use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::{KvCacheBuildSpec, KvCacheHandles, KvLayerHandle};
use crate::amg::weight_store::{SharedParam, WeightStore};
use crate::model_adapters::tensor_spec::{GEMMA2_SPEC, TransformParams, resolve_transforms};
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::tensor::Tensor;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::{LoadTransform, WeightMapper};

/// Handles produced by [`build_gemma2`]. Same shape as Phi3Handles
/// / LlamaHandles so the pipeline orchestration can treat all
/// builders uniformly downstream of the architecture branch.
pub struct Gemma2Handles {
    pub token_input_id: usize,
    pub logits_id: usize,
    pub param_ids: Vec<usize>,
    pub param_names: Vec<String>,
}

/// Convenience helper: register an owned zero-initialised graph
/// parameter under `full_name` and return the node id. Mirrors
/// the same pattern used by [`super::phi3::build_phi3`] — kept
/// private because the Gemma-2 weight-name set is
/// architecture-specific.
fn register_param_owned(
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

/// Per-layer Gemma 2 transformer block.
///
/// Exposed as a free function so the no-store and with-store
/// builders can both call it via different parameter-resolution
/// closures. The `register` closure abstracts the
/// "by-name → node_id" resolution: the no-store path returns
/// freshly-allocated zero slots; the with-store path pulls
/// `Arc`-shared tensors out of the store. Both append to
/// `param_ids` / `param_names` so the downstream weight mapper
/// sees an identical name list either way.
fn build_block_gemma2<R>(
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
    let n_kv = config.num_key_value_heads;
    let head_dim = config.effective_head_dim();
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;
    let bs = (batch * seq) as isize;
    let _ = n_kv; // reserved for future native-GQA path; today the
    // mapper tile-expands K/V to MHA shape.

    // **M11.C step 4** — Gemma 2 decouples `n_heads_q * head_dim`
    // from `hidden_size`. Gemma 2 2B: 8 * 256 = 2048 ≠ 2304.
    // Llama-family checkpoints always have these equal, so the
    // shared builder hard-codes `hidden`. Compute q_dim
    // explicitly so this builder works on any Gemma 2 variant.
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

    // ---- 2. Q/K/V projections (separate, like Llama) ----
    let q_proj_w = register(
        gb,
        &format!("{prefix}.self_attn.q_proj.weight"),
        vec![hidden, q_dim],
        param_ids,
        param_names,
    )?;
    // K and V are registered post-tile shape: the mapper's
    // `TileGroupedDim` expands the kv_dim rows to q_dim before
    // the load-time `Transpose2D`, so the slot here matches the
    // post-transform [hidden, q_dim] layout.
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
    // The mapper's TileGroupedDim tiles K/V from kv_heads to
    // n_heads_q on load, so all three projections share the same
    // post-load shape. This matches the Llama path exactly.
    let split_shape = vec![
        batch as isize,
        seq as isize,
        n_heads as isize,
        head_dim as isize,
    ];
    let q = gb.reshape(q_flat, split_shape.clone());
    let k = gb.reshape(k_flat, split_shape.clone());
    let v = gb.reshape(v_flat, split_shape);

    // ---- 4. RoPE on Q and new K (plain, no scaling) ----
    let position_offset: u32 = kv_cache.map(|spec| spec.cached_len as u32).unwrap_or(0);
    let q_rope = gb.rope_with_offset(q, head_dim, config.rope_theta, position_offset);
    let k_rope = gb.rope_with_offset(k, head_dim, config.rope_theta, position_offset);

    // ---- 5. [b, s, h, d] → [b, h, s, d] ----
    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let new_k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let new_v_perm = gb.permute(v, vec![0, 2, 1, 3]);

    // ---- 6. Cache concat (cache-aware path) ----
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

    // ---- 7. Attention scores → SoftCap(50) → mask → softmax → AV ----
    let k_full_t = gb.transpose_last_two(k_full);
    let scores_raw = gb.batch_matmul(q_perm, k_full_t);
    // SoftCap(attn_logit_softcapping) when the config carries it
    // (Some(50.0) for Gemma 2). Skip when None — guards the
    // generic Gemma-2-derived family with this builder when the
    // checkpoint disables the cap.
    let scores = match config.attn_logit_softcapping {
        Some(cap) => gb.soft_cap(scores_raw, cap),
        None => scores_raw,
    };
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);
    let attn_weights = gb.softmax(scores_masked);
    let attn_out = gb.batch_matmul(attn_weights, v_full);
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    // ---- 8. Output projection ----
    // Gemma 2 o_proj maps q_dim → hidden (Gemma 2 2B: 2048 → 2304).
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

    // ---- 9. Post-attention layernorm (on the attention OUTPUT,
    // before the first residual add) ----
    //
    // This is the first Gemma-2 dual-norm delta: Llama applies
    // its `post_attention_layernorm` AFTER the residual add (on
    // the running stream, before the FFN). Gemma 2 applies it
    // BEFORE the residual add, on the attention sub-block's
    // contribution only.
    let post_attn_ln_gamma = register(
        gb,
        &format!("{prefix}.post_attention_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let attn_proj_normed_pre = gb.rms_norm(attn_proj, config.rms_norm_eps);
    let attn_proj_normed = gb.broadcast_mul(attn_proj_normed_pre, post_attn_ln_gamma);

    // ---- 10. First residual ----
    let x_residual_1 = gb.add(x_residual_outer, attn_proj_normed);

    // ---- 11. Pre-feedforward layernorm (PRE-FFN, on the
    // running stream — same role as Llama's
    // post_attention_layernorm) ----
    let pre_ffn_ln_gamma = register(
        gb,
        &format!("{prefix}.pre_feedforward_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let h2_normed_pre = gb.rms_norm(x_residual_1, config.rms_norm_eps);
    let h2 = gb.broadcast_mul(h2_normed_pre, pre_ffn_ln_gamma);

    // ---- 12-14. GeGLU FFN ----
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
    // GeGLU: tanh-approximated GELU (matches Gemma 2's
    // hidden_act = "gelu_pytorch_tanh"). The default
    // `gb.gelu(.)` already routes to the tanh-approx kernel —
    // see `src/nn/activations.rs`.
    let gelu_gate_flat = gb.gelu(gate_flat);
    let ffn_pre_down_flat = gb.mul(gelu_gate_flat, up_flat);
    let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
    let ffn_out = gb.reshape(
        ffn_out_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 15. Post-feedforward layernorm (on FFN OUTPUT, before
    // second residual) — the second Gemma-2 dual-norm delta ----
    let post_ffn_ln_gamma = register(
        gb,
        &format!("{prefix}.post_feedforward_layernorm.weight"),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    )?;
    let ffn_out_normed_pre = gb.rms_norm(ffn_out, config.rms_norm_eps);
    let ffn_out_normed = gb.broadcast_mul(ffn_out_normed_pre, post_ffn_ln_gamma);

    // ---- 16. Second residual ----
    let layer_out = gb.add(x_residual_1, ffn_out_normed);

    if let (Some(handles_out), Some(handle)) = (kv_handles_out, layer_handle) {
        handles_out.push(handle);
    }
    Ok(layer_out)
}

/// Register a graph-local mutable F32 zero buffer (used for KV
/// cache slots — these are runtime-owned, not loaded from the
/// checkpoint, and must NOT appear in `param_names`).
fn register_local_zero_f32(gb: &mut GraphBuilder, shape: Vec<usize>) -> usize {
    let numel: usize = shape.iter().product();
    let data = vec![0.0_f32; numel];
    gb.parameter(Tensor::new_cpu(shape, data))
}

/// Build the head: final RmsNorm, LM head (tied or untied),
/// optional final-logit SoftCap, output reshape.
///
/// Returns `logits_id`. The dual mode (no-store / with-store) is
/// abstracted via the `register` closure, exactly as in
/// [`build_block_gemma2`].
fn build_head_gemma2<R>(
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
    // SoftCap(final_logit_softcapping) when the config carries
    // it. For Gemma 2 this is `Some(30.0)`; the cap is applied
    // before the reshape but the operation is shape-preserving
    // either way (SoftCap is elementwise).
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

/// Build the no-store Gemma 2 graph (parameter slots are owned
/// zero-init tensors that the loader will populate via the weight
/// mapper). Mirrors [`super::phi3::build_phi3`].
pub fn build_gemma2(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
) -> Gemma2Handles {
    config
        .validate()
        .expect("invalid LlamaConfig (Gemma 2 path)");

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();

    // ---- Causal mask (full causal — sliding-window deferred,
    // see module doc-comment) ----
    let seq = runtime.seq;
    let mut mask_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    // ---- Embedding lookup + scale ----
    let embed_w = register_param_owned(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let x_embed = gb.index_select(embed_w, token_input_id);
    let mut x = embed_scale_or_passthrough(gb, x_embed, config);

    // ---- Transformer blocks ----
    let mut register = |gb: &mut GraphBuilder,
                        full_name: &str,
                        shape: Vec<usize>,
                        ids: &mut Vec<usize>,
                        names: &mut Vec<String>|
     -> Result<usize, BuildError> {
        Ok(register_param_owned(gb, full_name, shape, ids, names))
    };
    for layer_idx in 0..config.num_hidden_layers {
        x = build_block_gemma2(
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
        .expect("no-store Gemma 2 register closure cannot fail");
    }

    // ---- Head ----
    let logits = build_head_gemma2(
        gb,
        x,
        embed_w,
        config,
        runtime,
        &mut param_ids,
        &mut param_names,
        &mut register,
    )
    .expect("no-store Gemma 2 register closure cannot fail");

    Gemma2Handles {
        token_input_id,
        logits_id: logits,
        param_ids,
        param_names,
    }
}

/// Build the store-backed, cache-aware Gemma 2 graph. Mirrors
/// [`super::phi3::build_phi3_with_store`] /
/// [`crate::nn::llama::builder_shared::build_llama_with_store`].
pub fn build_gemma2_with_store(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
    store: &WeightStore,
    kv_cache: Option<&KvCacheBuildSpec>,
) -> Result<LlamaHandlesShared, BuildError> {
    config
        .validate()
        .expect("invalid LlamaConfig (Gemma 2 path)");

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();
    let mut kv_handles_inner: Vec<KvLayerHandle> = Vec::new();

    // ---- Causal mask (cache-aware) ----
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

    // Store lookup closure — same shape as `register_param_owned`
    // but pulls Arc-shared tensors from the WeightStore.
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
        let tensor = p.to_tensor();
        let node_id = gb.parameter(tensor);
        gb.set_node_debug_name(node_id, full_name);
        ids.push(node_id);
        names.push(full_name.to_string());
        Ok(node_id)
    };

    // ---- Embedding lookup + scale ----
    let embed_w = register(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let x_embed = gb.index_select(embed_w, token_input_id);
    let mut x = embed_scale_or_passthrough(gb, x_embed, config);

    // ---- Transformer blocks ----
    for layer_idx in 0..config.num_hidden_layers {
        let kv_out_ref: Option<&mut Vec<KvLayerHandle>> = if kv_cache.is_some() {
            Some(&mut kv_handles_inner)
        } else {
            None
        };
        x = build_block_gemma2(
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

    // ---- Head ----
    let logits = build_head_gemma2(
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

/// Apply the Gemma 2 embedding scale `× sqrt(hidden_size)` if the
/// model is gemma2 (or any future family that opts in by setting
/// `model_type = "gemma2"`). The scale is registered as a
/// graph-local `[1, 1, 1]` constant — it must NOT appear in
/// `param_names` because the loader has nothing to map onto it.
///
/// Caller passes `x_embed` (output of `IndexSelect`), receives the
/// scaled activation. For non-Gemma model_types the function
/// passes through unchanged so the helper is reusable.
fn embed_scale_or_passthrough(
    gb: &mut GraphBuilder,
    x_embed: usize,
    config: &LlamaConfig,
) -> usize {
    if config.model_type.as_deref() == Some("gemma2") {
        let scale = (config.hidden_size as f32).sqrt();
        let scale_id = gb.parameter(Tensor::new_cpu(vec![1, 1, 1], vec![scale]));
        gb.broadcast_mul(x_embed, scale_id)
    } else {
        x_embed
    }
}

/// Build the [`WeightMapper`] for a Gemma 2 checkpoint.
///
/// Differences vs `llama_weight_mapper`:
/// - **Every RmsNorm gamma** carries an extra
///   `LoadTransform::AddScalar { scalar: 1.0 }` so the loaded
///   gamma matches Gemma 2's `(1 + γ) · rms(x)` semantics. The
///   `Reshape([1, 1, hidden])` from the Llama mapper still
///   applies — the order is `[Reshape, AddScalar]` so the
///   operation runs on the post-reshape buffer (the helpers
///   commute, but pinning the order keeps the load pipeline
///   deterministic for future mixed-transform additions).
/// - **K-side attention scale** uses
///   `1/sqrt(query_pre_attn_scalar)` instead of
///   `1/sqrt(head_dim)`. For Gemma 2 2B these are numerically
///   equal (`head_dim = qpas = 256`); the field-driven
///   computation generalises to other Gemma 2 variants where
///   they may diverge.
/// - **Four layer norms per layer** (Llama has two): the
///   `pre_feedforward_layernorm` and `post_feedforward_layernorm`
///   gammas land via the same `*.layernorm.weight` suffix
///   pattern, so no extra dispatch is needed — the suffix match
///   covers all four.
pub fn gemma2_weight_mapper(
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
            gemma2_transforms_for_name(name, hidden_size, head_dim, kv_groups, attention_scale);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

/// Per-name Gemma 2 transform list. Pure function — exposed
/// `pub` so unit tests can verify the dispatch directly without
/// going through the full `gemma2_weight_mapper` round-trip.
///
/// **AT-1c**: declarative via `GEMMA2_SPEC.hf_transforms`
/// (ADR-006). Same shape as the Llama HF ladder but the norm rule
/// adds the Gemma 2 `+1` fold and there are no QKV biases. The
/// K-side Scale uses `1/sqrt(query_pre_attn_scalar)` instead of
/// `1/sqrt(head_dim)` — the caller still threads this through
/// `attention_scale`. Behaviour byte-identical to the previous
/// ladder — pinned by the AT-1a golden
/// `golden_gemma2_hf_matches_live_gemma2_transforms` and the AT-2
/// conformance suite.
pub fn gemma2_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    resolve_transforms(
        GEMMA2_SPEC.hf_transforms,
        name,
        &TransformParams {
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amg::builder::GraphBuilder;
    use crate::nn::llama::builder::LlamaRuntime;

    /// Helper: minimal Gemma 2-shaped LlamaConfig for builder tests.
    /// Tiny dimensions to keep the synthetic graph cheap.
    fn tiny_gemma2_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 32,
            max_position_embeddings: 128,
            rope_theta: 10_000,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_bias: Some(false),
            model_type: Some("gemma2".to_string()),
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
            head_dim: Some(4),
            rope_scaling: None,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(4096),
            query_pre_attn_scalar: Some(4.0),
        }
    }

    /// **M11.C step 4 test** — `build_gemma2` registers exactly
    /// the names the Gemma 2 weight mapper expects, in a
    /// deterministic order. Per layer: 1 input_layernorm + 4
    /// attn projections + 1 post_attention_layernorm + 1
    /// pre_feedforward_layernorm + 3 mlp projections + 1
    /// post_feedforward_layernorm = 11 names. Plus 1 embed + 1
    /// final norm = 13 head-level. With L = 2 layers and tied
    /// embeddings: total = 2 + 11 * 2 = 24.
    #[test]
    fn build_gemma2_registers_expected_param_names() {
        let config = tiny_gemma2_config();
        let runtime = LlamaRuntime { batch: 1, seq: 4 };
        let mut gb = GraphBuilder::new();
        let token_in = gb.input();
        let h = build_gemma2(&mut gb, &config, &runtime, token_in);
        // 2 (embed + final_norm) + 11 per layer × 2 layers = 24.
        assert_eq!(h.param_names.len(), 24);
        assert_eq!(h.param_names[0], "model.embed_tokens.weight");
        assert_eq!(h.param_names[1], "model.layers.0.input_layernorm.weight");
        assert_eq!(h.param_names[2], "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(h.param_names[3], "model.layers.0.self_attn.k_proj.weight");
        assert_eq!(h.param_names[4], "model.layers.0.self_attn.v_proj.weight");
        assert_eq!(h.param_names[5], "model.layers.0.self_attn.o_proj.weight");
        assert_eq!(
            h.param_names[6],
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            h.param_names[7],
            "model.layers.0.pre_feedforward_layernorm.weight"
        );
        assert_eq!(h.param_names[8], "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(h.param_names[9], "model.layers.0.mlp.up_proj.weight");
        assert_eq!(h.param_names[10], "model.layers.0.mlp.down_proj.weight");
        assert_eq!(
            h.param_names[11],
            "model.layers.0.post_feedforward_layernorm.weight"
        );
        // Layer 1 starts at index 12.
        assert_eq!(h.param_names[12], "model.layers.1.input_layernorm.weight");
        // Final norm last (tied embeddings → no lm_head.weight).
        assert_eq!(*h.param_names.last().unwrap(), "model.norm.weight");
    }

    /// **M11.C step 4 test** — every RmsNorm gamma in the Gemma 2
    /// mapper has `[Reshape, AddScalar(1.0)]` as its transform
    /// list, in that order. Guards the M11.C step 3 contract
    /// from accidentally regressing to Llama-style `[Reshape]`
    /// only.
    #[test]
    fn gemma2_mapper_adds_scalar_one_to_every_rmsnorm_gamma() {
        let config = tiny_gemma2_config();
        let head_dim = config.effective_head_dim();
        let kv_groups = config.kv_groups();
        let qpas = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
        let attn_scale = 1.0_f32 / qpas.sqrt();
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.5.post_attention_layernorm.weight",
            "model.layers.10.pre_feedforward_layernorm.weight",
            "model.layers.25.post_feedforward_layernorm.weight",
        ] {
            let t = gemma2_transforms_for_name(
                name,
                config.hidden_size,
                head_dim,
                kv_groups,
                attn_scale,
            );
            assert_eq!(
                t,
                vec![
                    LoadTransform::Reshape {
                        target: vec![1, 1, config.hidden_size],
                    },
                    LoadTransform::AddScalar { scalar: 1.0 },
                ],
                "RmsNorm gamma '{name}' must carry [Reshape, AddScalar(1.0)]"
            );
        }
    }

    /// **M11.C step 4 test** — K-projection mapper transforms
    /// folds `1/sqrt(query_pre_attn_scalar)` (not
    /// `1/sqrt(head_dim)`) into the K weight via Scale. For the
    /// tiny fixture `qpas = 4.0` so `factor = 0.5`.
    #[test]
    fn gemma2_mapper_k_proj_scale_uses_query_pre_attn_scalar() {
        let config = tiny_gemma2_config();
        let head_dim = config.effective_head_dim();
        let kv_groups = config.kv_groups();
        let qpas = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
        let attn_scale = 1.0_f32 / qpas.sqrt();
        let t = gemma2_transforms_for_name(
            "model.layers.3.self_attn.k_proj.weight",
            config.hidden_size,
            head_dim,
            kv_groups,
            attn_scale,
        );
        // K weight: [TileGroupedDim, Transpose2D, Scale(1/sqrt(qpas))]
        assert_eq!(t.len(), 3);
        match &t[2] {
            LoadTransform::Scale { factor } => {
                assert!(
                    (factor - 0.5_f32).abs() < 1e-6,
                    "K-fold factor must be 1/sqrt(qpas=4) = 0.5, got {factor}"
                );
            }
            other => panic!("expected Scale at index 2, got {other:?}"),
        }
    }

    /// **M11.C step 4 test** — `gemma2_weight_mapper` round-trip:
    /// every name in `build_gemma2`'s param list resolves to a
    /// non-empty transform list (or empty for the embed). No
    /// missing dispatch.
    #[test]
    fn gemma2_mapper_covers_every_builder_param_name() {
        let config = tiny_gemma2_config();
        let runtime = LlamaRuntime { batch: 1, seq: 2 };
        let mut gb = GraphBuilder::new();
        let token_in = gb.input();
        let h = build_gemma2(&mut gb, &config, &runtime, token_in);
        let mapper =
            gemma2_weight_mapper(&config, &h.param_names, &h.param_ids).expect("mapper builds");
        // Every name registered by the builder must be in the mapper.
        for name in &h.param_names {
            assert!(
                mapper.contains(name),
                "mapper missing '{name}' (Gemma 2 builder registered it)"
            );
        }
    }

    /// **M11.C step 4 test** — graph contains the expected
    /// number of SoftCap nodes: per-layer attention softcap +
    /// final logit softcap = `num_layers + 1`. Guards against
    /// the conditional `Some(cap)` branches dropping a node.
    #[test]
    fn build_gemma2_emits_expected_number_of_softcap_nodes() {
        use crate::amg::nodes::NodeType;
        let config = tiny_gemma2_config();
        let runtime = LlamaRuntime { batch: 1, seq: 2 };
        let mut gb = GraphBuilder::new();
        let token_in = gb.input();
        let _h = build_gemma2(&mut gb, &config, &runtime, token_in);
        let graph = gb.build();
        let softcap_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::SoftCap { .. }))
            .count();
        assert_eq!(
            softcap_count,
            config.num_hidden_layers + 1,
            "expected one SoftCap per layer + one on logits"
        );
    }
}
