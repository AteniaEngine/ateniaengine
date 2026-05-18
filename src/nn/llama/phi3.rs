//! **M11.B** — Microsoft Phi-3 / Phi-3.5 architecture support.
//!
//! Phi-3 / Phi-3.5 (`Phi3ForCausalLM`) is structurally a
//! Llama-family transformer with two architectural deltas:
//!
//! 1. **LongRope** instead of plain RoPE / Llama 3 scaling. The
//!    config carries `rope_scaling.type = "longrope"` with
//!    per-dimension `short_factor[d/2]` and `long_factor[d/2]`
//!    vectors. Atenia's `RopeScaling::LongRope` config variant
//!    (M11.B step 1) and the `NodeType::RoPE` `LongRope`
//!    executor branch (M11.B step 2) cover this.
//!
//! 2. **Fused weights**: `qkv_proj` and `gate_up_proj` are
//!    single safetensors tensors. The Llama builder expects
//!    three (Q, K, V) and two (gate, up) separate tensors. The
//!    runtime split at the AMG level is the
//!    `NodeType::SliceLastDim` primitive (M11.B step 3.5). The
//!    pre-load split utilities in this module
//!    ([`split_fused_qkv`], [`split_fused_gate_up`]) remain
//!    available as pure helpers; the production path uses the
//!    runtime slice instead because it does not require any
//!    changes to the `WeightMapper` 1-to-1 contract.
//!
//! [`build_phi3`] mirrors `build_llama` with the two deltas
//! above; [`phi3_weight_mapper`] mirrors `llama_weight_mapper`
//! with the Phi-3 weight names (`qkv_proj`, `gate_up_proj`) and
//! the Phi-3 transform list.

/// **M11.B step 3** — split a fused `qkv_proj` weight tensor
/// along its last (output-features) axis into three per-
/// projection tensors `(q, k, v)`.
///
/// ## Layout contract
///
/// HuggingFace stores `qkv_proj` in the linear layer's native
/// row-major form `[out_features, in_features]` where
/// `out_features = (n_heads_q + 2 * n_heads_kv) * head_dim`.
/// This function consumes the tensor in that layout and
/// returns three slices each of shape
/// `[n_heads_x * head_dim, in_features]`.
///
/// **Note**: Atenia's loader applies a `LoadTransform::Transpose2D`
/// to projection weights to reach `[in_features, out_features]`
/// (Atenia's matmul convention). This function is meant to run
/// **before** the transpose — so it operates on the HF layout
/// where the fused dimension is `out_features` (rows). The
/// caller can either transpose post-split or pre-split; both
/// work because slicing rows in `[out, in]` is equivalent to
/// slicing columns in `[in, out]` after transpose, modulo
/// memory layout.
///
/// ## Phi-3.5 Mini specifics
///
/// Phi-3.5 Mini: hidden = 3072, n_heads_q = n_heads_kv = 32,
/// head_dim = 96. Fused shape = `[(32 + 64) * 96, 3072] =
/// [9216, 3072]`. Output q / k / v shapes = `[3072, 3072]` each.
///
/// ## Returns
///
/// `(q, k, v)` — three `Vec<f32>` buffers in the input's row-
/// major layout. The output shapes are
/// `[n_heads_q * head_dim, in_features]`,
/// `[n_heads_kv * head_dim, in_features]`,
/// `[n_heads_kv * head_dim, in_features]`.
///
/// ## Errors
///
/// - `fused.len() != fused_shape[0] * fused_shape[1]`
/// - `fused_shape.len() != 2`
/// - `fused_shape[0] != (n_heads_q + 2 * n_heads_kv) * head_dim`
pub fn split_fused_qkv(
    fused: &[f32],
    fused_shape: &[usize],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if fused_shape.len() != 2 {
        return Err(format!(
            "split_fused_qkv: expected rank-2 tensor, got shape {fused_shape:?}"
        ));
    }
    let out_features = fused_shape[0];
    let in_features = fused_shape[1];
    if fused.len() != out_features * in_features {
        return Err(format!(
            "split_fused_qkv: data length {} does not match shape {fused_shape:?} (numel {})",
            fused.len(),
            out_features * in_features
        ));
    }
    let expected_out = (n_heads_q + 2 * n_heads_kv) * head_dim;
    if out_features != expected_out {
        return Err(format!(
            "split_fused_qkv: expected out_features = (n_heads_q + 2*n_heads_kv) * head_dim = {expected_out}, got {out_features}"
        ));
    }
    let q_rows = n_heads_q * head_dim;
    let kv_rows = n_heads_kv * head_dim;
    // Row ranges in [out, in] layout.
    //   q : 0                 .. q_rows
    //   k : q_rows             .. q_rows + kv_rows
    //   v : q_rows + kv_rows   .. q_rows + 2 * kv_rows
    let q_end = q_rows;
    let k_end = q_rows + kv_rows;
    let v_end = q_rows + 2 * kv_rows;
    let q = fused[0..q_end * in_features].to_vec();
    let k = fused[q_end * in_features..k_end * in_features].to_vec();
    let v = fused[k_end * in_features..v_end * in_features].to_vec();
    Ok((q, k, v))
}

/// **M11.B step 3** — split a fused `gate_up_proj` weight tensor
/// along its last (output-features) axis into two per-projection
/// tensors `(gate, up)`. Phi-3.5 Mini concatenates the two FFN
/// up-projections (gate_proj + up_proj) into a single weight
/// `gate_up_proj` of shape `[2 * intermediate, hidden]`; this
/// function halves the row count.
///
/// ## Returns
///
/// `(gate, up)` — two `Vec<f32>` buffers, each of shape
/// `[intermediate, in_features]` in the input's row-major
/// layout. The first half goes to gate_proj, the second to
/// up_proj — the order is the HF convention used by Phi-3.
///
/// ## Errors
///
/// - `fused_shape.len() != 2`
/// - `fused_shape[0] % 2 != 0`
/// - `fused.len() != fused_shape[0] * fused_shape[1]`
pub fn split_fused_gate_up(
    fused: &[f32],
    fused_shape: &[usize],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if fused_shape.len() != 2 {
        return Err(format!(
            "split_fused_gate_up: expected rank-2 tensor, got shape {fused_shape:?}"
        ));
    }
    let out_features = fused_shape[0];
    let in_features = fused_shape[1];
    if out_features % 2 != 0 {
        return Err(format!(
            "split_fused_gate_up: out_features {out_features} is not divisible by 2"
        ));
    }
    if fused.len() != out_features * in_features {
        return Err(format!(
            "split_fused_gate_up: data length {} does not match shape {fused_shape:?} (numel {})",
            fused.len(),
            out_features * in_features
        ));
    }
    let half = out_features / 2;
    let split_idx = half * in_features;
    let gate = fused[0..split_idx].to_vec();
    let up = fused[split_idx..].to_vec();
    Ok((gate, up))
}

// =====================================================================
// **M11.B step 4** — Phi-3 graph builder + weight mapper.
// =====================================================================

use crate::amg::builder::GraphBuilder;
use crate::amg::nodes::RopeScalingLongRope;
use crate::model_adapters::tensor_spec::{PHI3_SPEC, TransformParams, resolve_transforms};
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::config::{LlamaConfig, RopeScaling};
use crate::tensor::Tensor;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::{LoadTransform, WeightMapper};

/// Handles produced by [`build_phi3`]. Same shape as
/// `LlamaHandles` so the pipeline orchestration can treat both
/// builders uniformly downstream of the architecture branch.
pub struct Phi3Handles {
    pub token_input_id: usize,
    pub logits_id: usize,
    /// Index-aligned with [`Self::param_names`].
    pub param_ids: Vec<usize>,
    /// Parameter names in HuggingFace Phi-3 convention. Notably
    /// includes `qkv_proj` (fused) and `gate_up_proj` (fused)
    /// instead of separate q/k/v/gate/up entries.
    pub param_names: Vec<String>,
}

/// Convenience: register a graph parameter under the given HF
/// name and return its node id. Mirrors the same helper used by
/// the Llama builder; kept private because the names + shapes
/// are Phi-3-specific.
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

/// Build a single Phi-3 transformer block. Same overall shape
/// as the Llama block but with:
///   - one `qkv_proj` parameter that gets split via
///     `SliceLastDim` after the projection matmul
///   - one `gate_up_proj` parameter split likewise for the FFN
///   - LongRope-aware `rope_longrope` calls on Q and K
///   - K-side attention scale applied at runtime (not folded
///     into a weight transform like the Llama path) because the
///     fused `qkv_proj` shares its rows with Q and V
fn build_transformer_block_phi3(
    gb: &mut GraphBuilder,
    layer_idx: usize,
    x: usize,
    causal_mask_id: usize,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    longrope_scaling: &RopeScalingLongRope,
    attention_scale_param_id: usize,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> usize {
    let prefix = format!("model.layers.{}", layer_idx);
    let hidden = config.hidden_size;
    let n_heads_q = config.num_attention_heads;
    let n_heads_kv = config.num_key_value_heads;
    let head_dim = config.effective_head_dim();
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

    // ---- 2. Fused QKV projection ----
    //
    // `qkv_proj.weight` is `[hidden, (n_q + 2*n_kv) * head_dim]`
    // after the load-time Transpose2D. The matmul produces
    // `[bs, qkv_out]` where `qkv_out = (n_q + 2*n_kv) * head_dim`.
    let qkv_out = (n_heads_q + 2 * n_heads_kv) * head_dim;
    let qkv_proj_w = register_param(
        gb,
        &format!("{}.self_attn.qkv_proj.weight", prefix),
        vec![hidden, qkv_out],
        param_ids,
        param_names,
    );
    let h_flat = gb.reshape(h, vec![bs, hidden as isize]);
    let qkv_flat = gb.matmul(h_flat, qkv_proj_w);
    // Slice the fused activation along its last axis into Q / K / V.
    let q_end = n_heads_q * head_dim;
    let k_end = q_end + n_heads_kv * head_dim;
    let v_end = qkv_out;
    let q_flat = gb.slice_last_dim(qkv_flat, 0, q_end);
    let k_flat = gb.slice_last_dim(qkv_flat, q_end, k_end);
    let v_flat = gb.slice_last_dim(qkv_flat, k_end, v_end);

    // ---- 3. Multi-head reshape ----
    let q_split = vec![
        batch as isize,
        seq as isize,
        n_heads_q as isize,
        head_dim as isize,
    ];
    let kv_split = vec![
        batch as isize,
        seq as isize,
        n_heads_kv as isize,
        head_dim as isize,
    ];
    let q = gb.reshape(q_flat, q_split.clone());
    let k_unscaled = gb.reshape(k_flat, kv_split.clone());
    let v = gb.reshape(v_flat, kv_split);

    // ---- 3.5 K-side attention scale ----
    //
    // The Llama builder pre-folds `1/sqrt(head_dim)` into the
    // K_proj weight via a load-time `Scale` transform. For Phi-3
    // the fused `qkv_proj` shares its rows with Q and V, so the
    // scale cannot be folded into the weight without a row-range
    // -aware transform (which would require widening
    // `LoadTransform`). Apply it at the graph level instead by
    // broadcast-multiplying K against a `[1,1,1,1]` constant.
    // Numerically equivalent: scoring is bilinear in K so a
    // pre- or post-projection scale produces the same softmax
    // input.
    let k = gb.broadcast_mul(k_unscaled, attention_scale_param_id);

    // ---- 4. RoPE-LongRope on Q and K ----
    let q_rope = gb.rope_longrope(q, head_dim, config.rope_theta, longrope_scaling.clone());
    let k_rope = gb.rope_longrope(k, head_dim, config.rope_theta, longrope_scaling.clone());

    // ---- 5. [b, s, h, d] → [b, h, s, d] ----
    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let v_perm = gb.permute(v, vec![0, 2, 1, 3]);

    // ---- 6. Q @ K^T ----
    let k_perm_t = gb.transpose_last_two(k_perm);
    let scores = gb.batch_matmul(q_perm, k_perm_t);

    // ---- 7. Causal mask ----
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);

    // ---- 8. Softmax ----
    let attn_weights = gb.softmax(scores_masked);

    // ---- 9. weights @ V ----
    let attn_out = gb.batch_matmul(attn_weights, v_perm);
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    // ---- 10. Output projection ----
    let attn_out_flat = gb.reshape(attn_out_back, vec![bs, hidden as isize]);
    let o_proj_w = register_param(
        gb,
        &format!("{}.self_attn.o_proj.weight", prefix),
        vec![hidden, hidden],
        param_ids,
        param_names,
    );
    let attn_proj_flat = gb.matmul(attn_out_flat, o_proj_w);
    let attn_proj = gb.reshape(
        attn_proj_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 11. Attention residual ----
    let x_residual_1 = gb.add(x, attn_proj);

    // ---- 12. Post-attention layernorm ----
    let post_ln_gamma = register_param(
        gb,
        &format!("{}.post_attention_layernorm.weight", prefix),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    );
    let h2_normed = gb.rms_norm(x_residual_1, config.rms_norm_eps);
    let h2 = gb.broadcast_mul(h2_normed, post_ln_gamma);

    // ---- 13. Fused gate_up SwiGLU FFN ----
    //
    // `gate_up_proj.weight` is `[hidden, 2 * intermediate]`
    // after Transpose2D. Single matmul produces the fused
    // activation; SliceLastDim halves it into gate / up.
    let gate_up_out = 2 * intermediate;
    let gate_up_proj_w = register_param(
        gb,
        &format!("{}.mlp.gate_up_proj.weight", prefix),
        vec![hidden, gate_up_out],
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
    let gate_up_flat = gb.matmul(h2_flat, gate_up_proj_w);
    let gate_flat = gb.slice_last_dim(gate_up_flat, 0, intermediate);
    let up_flat = gb.slice_last_dim(gate_up_flat, intermediate, gate_up_out);
    let silu_gate_flat = gb.silu(gate_flat);
    let ffn_pre_down_flat = gb.mul(silu_gate_flat, up_flat);
    let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
    let ffn_out = gb.reshape(
        ffn_out_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 14. FFN residual ----
    gb.add(x_residual_1, ffn_out)
}

/// Build the complete Phi-3 graph.
///
/// Mirrors `build_llama`'s contract: `token_input_id` must be a
/// previously-registered `Input` of shape `[batch, seq]`; the
/// returned `Phi3Handles` carries the `logits_id` plus the
/// (param_ids, param_names) pair the loader needs to populate.
///
/// Panics if the config does not carry a `RopeScaling::LongRope`
/// scaling block — Phi-3 without LongRope is a malformed config
/// for this builder.
pub fn build_phi3(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
) -> Phi3Handles {
    config.validate().expect("invalid LlamaConfig (Phi-3 path)");

    // Resolve LongRope scaling from the parsed config. The
    // builder expects it to be present; otherwise the caller
    // routed a non-Phi-3 config to the Phi-3 builder.
    let longrope_scaling = match config.effective_rope_scaling() {
        Some(RopeScaling::LongRope {
            short_factor,
            long_factor,
            original_max_position_embeddings,
            max_position_embeddings,
        }) => RopeScalingLongRope::new(
            short_factor,
            long_factor,
            *original_max_position_embeddings,
            *max_position_embeddings,
        ),
        other => panic!(
            "build_phi3 requires rope_scaling = LongRope, got {other:?}. \
             Phi-3 / Phi-3.5 checkpoints must declare type=\"longrope\" in \
             config.json's rope_scaling block."
        ),
    };

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();

    // ---- Causal mask ----
    let seq = runtime.seq;
    let mut mask_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    // ---- Attention scale constant ----
    //
    // Used by every transformer block to scale K post-RoPE.
    // Registered once at the graph root so the cost is `[1]
    // float` total, not per-layer.
    let attention_scale = 1.0_f32 / (config.effective_head_dim() as f32).sqrt();
    let attention_scale_tensor = Tensor::new_cpu(vec![1, 1, 1, 1], vec![attention_scale]);
    let attention_scale_param_id = gb.parameter(attention_scale_tensor);

    // ---- Embedding ----
    let embed_w = register_param(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let mut x = gb.index_select(embed_w, token_input_id);

    // ---- Transformer blocks ----
    for layer_idx in 0..config.num_hidden_layers {
        x = build_transformer_block_phi3(
            gb,
            layer_idx,
            x,
            causal_mask_id,
            config,
            runtime,
            &longrope_scaling,
            attention_scale_param_id,
            &mut param_ids,
            &mut param_names,
        );
    }

    // ---- Final RMSNorm ----
    let final_ln_gamma = register_param(
        gb,
        "model.norm.weight",
        vec![1, 1, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let x_normed = gb.rms_norm(x, config.rms_norm_eps);
    let x_final = gb.broadcast_mul(x_normed, final_ln_gamma);

    // ---- LM head ----
    let bs = (runtime.batch * runtime.seq) as isize;
    let x_flat = gb.reshape(x_final, vec![bs, config.hidden_size as isize]);

    let logits_flat = if config.tie_word_embeddings {
        gb.matmul_rhs_transposed(x_flat, embed_w)
    } else {
        let lm_head_input = register_param(
            gb,
            "lm_head.weight",
            vec![config.hidden_size, config.vocab_size],
            &mut param_ids,
            &mut param_names,
        );
        gb.matmul(x_flat, lm_head_input)
    };
    let logits = gb.reshape(
        logits_flat,
        vec![
            runtime.batch as isize,
            runtime.seq as isize,
            config.vocab_size as isize,
        ],
    );

    Phi3Handles {
        token_input_id,
        logits_id: logits,
        param_ids,
        param_names,
    }
}

/// **M11.B step 4** — Phi-3 counterpart of
/// [`crate::nn::llama::builder_shared::build_llama_with_store`].
/// Same shape contract (`(token_input, logits, params, kv_handles)`),
/// same cache-aware behaviour (None → pre-fill / no cache; Some
/// → decode-step with pre-rotated cached K, freshly-rotated new
/// K, and `Concat`-axis-2 along the time dimension to produce
/// `K_full` / `V_full`). Differences vs the Llama variant:
///
///   - one fused `qkv_proj` parameter, runtime-split via three
///     `SliceLastDim` nodes after the matmul
///   - one fused `gate_up_proj` parameter, runtime-split via
///     two `SliceLastDim` nodes
///   - LongRope (`rope_longrope_with_offset`) on Q and new K
///   - K-side `1/sqrt(head_dim)` scale applied at the graph
///     level (BroadcastMul against a `[1,1,1,1]` constant
///     parameter) because the fused `qkv_proj` shares its
///     rows with Q and V
pub fn build_phi3_with_store(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
    store: &crate::amg::weight_store::WeightStore,
    kv_cache: Option<&crate::amg::kv_cache::KvCacheBuildSpec>,
) -> Result<
    crate::nn::llama::builder_shared::LlamaHandlesShared,
    crate::nn::llama::builder_shared::BuildError,
> {
    use crate::amg::kv_cache::{KvCacheHandles, KvLayerHandle};
    use crate::amg::weight_store::SharedParam;
    use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};

    config.validate().expect("invalid LlamaConfig (Phi-3 path)");

    // Resolve LongRope scaling — same precondition as
    // `build_phi3` (no-cache prefill builder).
    let longrope_scaling = match config.effective_rope_scaling() {
        Some(RopeScaling::LongRope {
            short_factor,
            long_factor,
            original_max_position_embeddings,
            max_position_embeddings,
        }) => RopeScalingLongRope::new(
            short_factor,
            long_factor,
            *original_max_position_embeddings,
            *max_position_embeddings,
        ),
        other => panic!("build_phi3_with_store requires rope_scaling = LongRope, got {other:?}"),
    };

    // Helper closures local to this builder. They mirror the
    // private helpers in `builder_shared.rs` but stay closed
    // over `store` so we don't have to re-expose them publicly.
    let lookup = |gb: &mut GraphBuilder,
                  full_name: &str,
                  expected_shape: Vec<usize>,
                  param_ids: &mut Vec<usize>,
                  param_names: &mut Vec<String>|
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
        param_ids.push(node_id);
        param_names.push(full_name.to_string());
        Ok(node_id)
    };
    let local_zero = |gb: &mut GraphBuilder, shape: Vec<usize>| -> usize {
        let numel: usize = shape.iter().product();
        let data = vec![0.0_f32; numel];
        gb.parameter(Tensor::new_cpu(shape, data))
    };

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();
    let mut kv_handles_inner: Vec<KvLayerHandle> = Vec::new();

    let hidden = config.hidden_size;
    let n_heads_q = config.num_attention_heads;
    let n_heads_kv = config.num_key_value_heads;
    let head_dim = config.effective_head_dim();
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;

    // ---- Causal mask (cache-aware) ----
    let cached_len = kv_cache.map(|s| s.cached_len).unwrap_or(0);
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

    // ---- Attention scale constant ----
    let attention_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let attention_scale_id = gb.parameter(Tensor::new_cpu(vec![1, 1, 1, 1], vec![attention_scale]));

    // ---- Embedding ----
    let embed_w = lookup(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let mut x = gb.index_select(embed_w, token_input_id);

    let position_offset: u32 = kv_cache.map(|spec| spec.cached_len as u32).unwrap_or(0);
    let bs = (batch * seq) as isize;

    // ---- Transformer blocks ----
    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", layer_idx);

        // 1. Input layernorm
        let input_ln_gamma = lookup(
            gb,
            &format!("{prefix}.input_layernorm.weight"),
            vec![1, 1, hidden],
            &mut param_ids,
            &mut param_names,
        )?;
        let h_normed = gb.rms_norm(x, config.rms_norm_eps);
        let h = gb.broadcast_mul(h_normed, input_ln_gamma);

        // 2. Fused QKV projection
        let qkv_out = (n_heads_q + 2 * n_heads_kv) * head_dim;
        let qkv_w = lookup(
            gb,
            &format!("{prefix}.self_attn.qkv_proj.weight"),
            vec![hidden, qkv_out],
            &mut param_ids,
            &mut param_names,
        )?;
        let h_flat = gb.reshape(h, vec![bs, hidden as isize]);
        let qkv_flat = gb.matmul(h_flat, qkv_w);
        let q_end = n_heads_q * head_dim;
        let k_end = q_end + n_heads_kv * head_dim;
        let v_end = qkv_out;
        let q_flat = gb.slice_last_dim(qkv_flat, 0, q_end);
        let k_flat = gb.slice_last_dim(qkv_flat, q_end, k_end);
        let v_flat = gb.slice_last_dim(qkv_flat, k_end, v_end);

        // 3. Multi-head reshape
        let q_split = vec![
            batch as isize,
            seq as isize,
            n_heads_q as isize,
            head_dim as isize,
        ];
        let kv_split = vec![
            batch as isize,
            seq as isize,
            n_heads_kv as isize,
            head_dim as isize,
        ];
        let q = gb.reshape(q_flat, q_split.clone());
        let k_unscaled = gb.reshape(k_flat, kv_split.clone());
        let v = gb.reshape(v_flat, kv_split);

        // 3.5 K-side attention scale (graph-level)
        let k = gb.broadcast_mul(k_unscaled, attention_scale_id);

        // 4. RoPE-LongRope with offset
        let q_rope = gb.rope_longrope_with_offset(
            q,
            head_dim,
            config.rope_theta,
            longrope_scaling.clone(),
            position_offset,
        );
        let k_rope = gb.rope_longrope_with_offset(
            k,
            head_dim,
            config.rope_theta,
            longrope_scaling.clone(),
            position_offset,
        );

        // 5. [b, s, h, d] → [b, h, s, d]
        let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
        let new_k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
        let new_v_perm = gb.permute(v, vec![0, 2, 1, 3]);

        // 6. Concat against cache (if any)
        let (k_full, v_full, layer_handle): (usize, usize, Option<KvLayerHandle>) = match kv_cache {
            None => (new_k_perm, new_v_perm, None),
            Some(spec) => {
                let cache_shape = vec![batch, n_heads_kv, spec.cached_len, head_dim];
                let cache_k_id = local_zero(gb, cache_shape.clone());
                let cache_v_id = local_zero(gb, cache_shape);
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

        // 7. Attention
        let k_full_t = gb.transpose_last_two(k_full);
        let scores = gb.batch_matmul(q_perm, k_full_t);
        let scores_masked = gb.broadcast_add(scores, causal_mask_id);
        let attn_weights = gb.softmax(scores_masked);
        let attn_out = gb.batch_matmul(attn_weights, v_full);
        let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

        // 8. Output projection
        let attn_out_flat = gb.reshape(attn_out_back, vec![bs, hidden as isize]);
        let o_proj_w = lookup(
            gb,
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
            &mut param_ids,
            &mut param_names,
        )?;
        let attn_proj_flat = gb.matmul(attn_out_flat, o_proj_w);
        let attn_proj = gb.reshape(
            attn_proj_flat,
            vec![batch as isize, seq as isize, hidden as isize],
        );
        let x_residual_1 = gb.add(x, attn_proj);

        // 9. Post-attention layernorm
        let post_ln_gamma = lookup(
            gb,
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![1, 1, hidden],
            &mut param_ids,
            &mut param_names,
        )?;
        let h2_normed = gb.rms_norm(x_residual_1, config.rms_norm_eps);
        let h2 = gb.broadcast_mul(h2_normed, post_ln_gamma);

        // 10. Fused gate_up SwiGLU
        let gate_up_out = 2 * intermediate;
        let gate_up_w = lookup(
            gb,
            &format!("{prefix}.mlp.gate_up_proj.weight"),
            vec![hidden, gate_up_out],
            &mut param_ids,
            &mut param_names,
        )?;
        let down_proj_w = lookup(
            gb,
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![intermediate, hidden],
            &mut param_ids,
            &mut param_names,
        )?;
        let h2_flat = gb.reshape(h2, vec![bs, hidden as isize]);
        let gate_up_flat = gb.matmul(h2_flat, gate_up_w);
        let gate_flat = gb.slice_last_dim(gate_up_flat, 0, intermediate);
        let up_flat = gb.slice_last_dim(gate_up_flat, intermediate, gate_up_out);
        let silu_gate_flat = gb.silu(gate_flat);
        let ffn_pre_down_flat = gb.mul(silu_gate_flat, up_flat);
        let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
        let ffn_out = gb.reshape(
            ffn_out_flat,
            vec![batch as isize, seq as isize, hidden as isize],
        );

        x = gb.add(x_residual_1, ffn_out);

        if let Some(handle) = layer_handle {
            kv_handles_inner.push(handle);
        }
    }

    // ---- Final RMSNorm ----
    let final_ln_gamma = lookup(
        gb,
        "model.norm.weight",
        vec![1, 1, hidden],
        &mut param_ids,
        &mut param_names,
    )?;
    let x_normed = gb.rms_norm(x, config.rms_norm_eps);
    let x_final = gb.broadcast_mul(x_normed, final_ln_gamma);

    // ---- LM head ----
    let x_flat = gb.reshape(x_final, vec![bs, hidden as isize]);
    let logits_flat = if config.tie_word_embeddings {
        gb.matmul_rhs_transposed(x_flat, embed_w)
    } else {
        let lm_head_input = lookup(
            gb,
            "lm_head.weight",
            vec![hidden, config.vocab_size],
            &mut param_ids,
            &mut param_names,
        )?;
        gb.matmul(x_flat, lm_head_input)
    };
    let logits = gb.reshape(
        logits_flat,
        vec![batch as isize, seq as isize, config.vocab_size as isize],
    );

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

/// Build the [`WeightMapper`] for a Phi-3 / Phi-3.5 checkpoint.
///
/// Differences vs `llama_weight_mapper`:
///   - `qkv_proj.weight` and `gate_up_proj.weight` are fused —
///     each gets a single `Transpose2D` transform (no per-section
///     scale; the K-side `1/sqrt(head_dim)` lives in the graph
///     as a `BroadcastMul` against a constant param).
///   - No `q_proj` / `k_proj` / `v_proj` / `gate_proj` /
///     `up_proj` entries — the Phi-3 builder doesn't register
///     those names, so the mapper never sees them.
///   - No QKV biases (Phi-3 uses `attention_bias = false`).
pub fn phi3_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms = phi3_transforms_for_name(name, hidden_size);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }

    Ok(mapper)
}

/// Per-name Phi-3 transform list. Pure function, exposed `pub`
/// so unit tests can verify dispatch directly.
///
/// **AT-1c**: declarative via `PHI3_SPEC.hf_transforms`
/// (ADR-006). `phi3_gguf_transforms_for_name` still delegates
/// here, so this is also the Phi-3 GGUF table. Phi-3's recipes
/// (`ReshapeHidden3D`, `Transpose2D`) never reference
/// head_dim/kv_groups/attention_scale, so the signature stays
/// `(name, hidden_size)` and the unused params are zero. Behaviour
/// is byte-identical to the previous ladder — pinned by the AT-1a
/// golden `golden_phi3_hf_matches_live_phi3_transforms` and the
/// AT-2 conformance suite.
pub fn phi3_transforms_for_name(name: &str, hidden_size: usize) -> Vec<LoadTransform> {
    resolve_transforms(
        PHI3_SPEC.hf_transforms,
        name,
        &TransformParams {
            hidden_size,
            head_dim: 0,
            kv_groups: 0,
            attention_scale: 0.0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Phi-3.5 Mini production shape** — `qkv_proj` row-count
    /// equals `(32 + 32 + 32) * 96 = 9216`. With `n_heads_q =
    /// n_heads_kv = 32` and `head_dim = 96` the resulting q / k
    /// / v slices each have row-count `32 * 96 = 3072`.
    #[test]
    fn split_qkv_phi35_mini_shape() {
        let n_q = 32usize;
        let n_kv = 32usize;
        let head_dim = 96usize;
        let in_features = 3072usize;
        let out_features = (n_q + 2 * n_kv) * head_dim;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 1e-4)
            .collect();
        let (q, k, v) = split_fused_qkv(&fused, &[out_features, in_features], n_q, n_kv, head_dim)
            .expect("split_fused_qkv must accept the Phi-3.5 Mini shape");
        assert_eq!(q.len(), n_q * head_dim * in_features);
        assert_eq!(k.len(), n_kv * head_dim * in_features);
        assert_eq!(v.len(), n_kv * head_dim * in_features);
        // First element of q must equal the first element of fused.
        assert_eq!(q[0], fused[0]);
        // First element of k starts after q's rows.
        let k_offset = n_q * head_dim * in_features;
        assert_eq!(k[0], fused[k_offset]);
        // First element of v starts after q + k rows.
        let v_offset = (n_q + n_kv) * head_dim * in_features;
        assert_eq!(v[0], fused[v_offset]);
        // Last element of v is the last element of fused.
        assert_eq!(v[v.len() - 1], fused[fused.len() - 1]);
    }

    /// GQA case: n_kv smaller than n_q. Verify the split still
    /// produces correctly-sized slices and the row offsets match.
    #[test]
    fn split_qkv_gqa_case() {
        let n_q = 32usize;
        let n_kv = 8usize;
        let head_dim = 64usize;
        let in_features = 2048usize;
        let out_features = (n_q + 2 * n_kv) * head_dim;
        let fused: Vec<f32> = (0..out_features * in_features).map(|i| i as f32).collect();
        let (q, k, v) = split_fused_qkv(&fused, &[out_features, in_features], n_q, n_kv, head_dim)
            .expect("split should succeed");
        assert_eq!(q.len(), 32 * 64 * 2048);
        assert_eq!(k.len(), 8 * 64 * 2048);
        assert_eq!(v.len(), 8 * 64 * 2048);
        // Spot-check element values at boundaries.
        assert_eq!(q[q.len() - 1], fused[q.len() - 1]);
        assert_eq!(k[0], fused[q.len()]);
        assert_eq!(v[0], fused[q.len() + k.len()]);
    }

    /// Shape mismatch: data length inconsistent with declared shape.
    #[test]
    fn split_qkv_rejects_inconsistent_data_length() {
        let fused: Vec<f32> = vec![0.0; 100];
        let err = split_fused_qkv(&fused, &[200, 1], 1, 1, 1)
            .expect_err("inconsistent data length must fail");
        assert!(err.contains("data length"), "got: {err}");
    }

    /// Out-features mismatch: shape claims 100 rows but the
    /// (n_q + 2*n_kv) * head_dim equation gives a different
    /// number.
    #[test]
    fn split_qkv_rejects_unexpected_out_features() {
        let fused: Vec<f32> = vec![0.0; 100 * 8];
        let err = split_fused_qkv(&fused, &[100, 8], 32, 32, 96)
            .expect_err("out_features mismatch must fail");
        assert!(err.contains("out_features"), "got: {err}");
    }

    /// **Phi-3.5 Mini production shape** — `gate_up_proj` has
    /// row-count `2 * 8192 = 16384` and column-count 3072. The
    /// half-split produces two `[8192, 3072]` slices.
    #[test]
    fn split_gate_up_phi35_mini_shape() {
        let intermediate = 8192usize;
        let in_features = 3072usize;
        let out_features = 2 * intermediate;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 1e-5)
            .collect();
        let (gate, up) =
            split_fused_gate_up(&fused, &[out_features, in_features]).expect("split must succeed");
        assert_eq!(gate.len(), intermediate * in_features);
        assert_eq!(up.len(), intermediate * in_features);
        assert_eq!(gate[0], fused[0]);
        assert_eq!(up[0], fused[intermediate * in_features]);
        assert_eq!(up[up.len() - 1], fused[fused.len() - 1]);
    }

    /// Odd row count must fail — the half-split is undefined.
    #[test]
    fn split_gate_up_rejects_odd_out_features() {
        let fused: Vec<f32> = vec![0.0; 7 * 4];
        let err = split_fused_gate_up(&fused, &[7, 4]).expect_err("odd out_features must fail");
        assert!(err.contains("divisible by 2"), "got: {err}");
    }

    /// **M11.B step 4** — `phi3_transforms_for_name` dispatch.
    /// The fused weights and o_proj / down_proj all land on a
    /// single `Transpose2D`. Layernorms reshape to `[1,1,hidden]`.
    /// Embed and unknown names get an empty list.
    #[test]
    fn phi3_transform_dispatch_covers_known_names() {
        let h = 3072;
        // Embed: empty
        assert_eq!(
            phi3_transforms_for_name("model.embed_tokens.weight", h),
            vec![]
        );
        // Layernorms: Reshape to [1, 1, hidden]
        let expect_norm = vec![LoadTransform::Reshape {
            target: vec![1, 1, h],
        }];
        assert_eq!(
            phi3_transforms_for_name("model.norm.weight", h),
            expect_norm
        );
        assert_eq!(
            phi3_transforms_for_name("model.layers.0.input_layernorm.weight", h),
            expect_norm
        );
        assert_eq!(
            phi3_transforms_for_name("model.layers.31.post_attention_layernorm.weight", h),
            expect_norm
        );
        // Fused weights + o_proj + down_proj + lm_head: single Transpose2D
        let expect_t = vec![LoadTransform::Transpose2D];
        for name in &[
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ] {
            assert_eq!(
                phi3_transforms_for_name(name, h),
                expect_t,
                "mismatch on {name}"
            );
        }
    }

    /// **M11.B step 4** — `build_phi3` builds a graph with the
    /// expected param-name set and node count. Uses a tiny
    /// synthetic config to keep the build cost trivial.
    #[test]
    fn build_phi3_produces_expected_param_names() {
        use crate::amg::builder::GraphBuilder;
        use crate::nn::llama::builder::LlamaRuntime;
        use crate::nn::llama::config::{LlamaConfig, RopeScaling};
        let config = LlamaConfig {
            vocab_size: 32_064,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 4096,
            rope_theta: 10_000,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            attention_bias: Some(false),
            model_type: Some("phi3".to_string()),
            bos_token_id: 1,
            eos_token_id: 32_000,
            pad_token_id: None,
            head_dim: None,
            rope_scaling: Some(RopeScaling::LongRope {
                short_factor: vec![1.0; 8],
                long_factor: vec![2.0; 8],
                original_max_position_embeddings: 2048,
                max_position_embeddings: 4096,
            }),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            sliding_window: None,
            query_pre_attn_scalar: None,
        };
        let runtime = LlamaRuntime { batch: 1, seq: 4 };
        let mut gb = GraphBuilder::new();
        let token_in = gb.input();
        let handles = build_phi3(&mut gb, &config, &runtime, token_in);

        // Sanity: token + logits.
        assert_eq!(handles.token_input_id, token_in);
        assert!(handles.param_names.len() == handles.param_ids.len());

        // Per-layer Phi-3 names: 6 entries × 2 layers = 12. Plus
        // model.embed_tokens, model.norm, lm_head = 3. Total 15.
        let expected_count = 6 * config.num_hidden_layers + 3;
        assert_eq!(
            handles.param_names.len(),
            expected_count,
            "Phi-3 param count: 6 per layer + 3 root, got {} entries: {:#?}",
            handles.param_names.len(),
            handles.param_names
        );

        // Confirm fused names present and split names absent.
        let names: std::collections::HashSet<_> =
            handles.param_names.iter().map(|s| s.as_str()).collect();
        assert!(names.contains("model.embed_tokens.weight"));
        assert!(names.contains("model.norm.weight"));
        assert!(names.contains("lm_head.weight"));
        assert!(names.contains("model.layers.0.self_attn.qkv_proj.weight"));
        assert!(names.contains("model.layers.0.mlp.gate_up_proj.weight"));
        assert!(!names.contains("model.layers.0.self_attn.q_proj.weight"));
        assert!(!names.contains("model.layers.0.mlp.gate_proj.weight"));
    }

    /// Round-trip: re-concatenating the splits along the row
    /// axis must reproduce the original buffer bit-exactly.
    #[test]
    fn split_qkv_round_trip_concat_matches_input() {
        let n_q = 8usize;
        let n_kv = 4usize;
        let head_dim = 16usize;
        let in_features = 32usize;
        let out_features = (n_q + 2 * n_kv) * head_dim;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32).sin())
            .collect();
        let (q, k, v) = split_fused_qkv(&fused, &[out_features, in_features], n_q, n_kv, head_dim)
            .expect("split should succeed");
        let mut concatenated = q;
        concatenated.extend_from_slice(&k);
        concatenated.extend_from_slice(&v);
        assert_eq!(concatenated.len(), fused.len());
        for (i, (a, b)) in concatenated.iter().zip(fused.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "split round-trip mismatch at index {i}: {a} vs {b}"
            );
        }
    }
}
