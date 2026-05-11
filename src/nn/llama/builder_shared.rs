//! M5.c.2.c — Llama-family graph builder over an
//! [`Arc`-shared `WeightStore`](crate::amg::weight_store::WeightStore),
//! with optional cache-aware attention path.
//!
//! ## Why this lives next to `builder` and not inside it
//!
//! The original [`super::builder::build_llama`] is the
//! load-bearing path for the M4.6 four-model F64 fixtures
//! and the M4.7 13B `momento guau`. Its parameter-registration
//! flow (`register_param_named` at line 72 of `builder.rs`)
//! creates **owned** zero-initialised `Tensor` slots that the
//! `WeightMapper::load_into` populates afterwards. Threading
//! `Option<&KvCacheBuildSpec>` and "weights from a store"
//! through that helper would (a) widen its surface for the
//! 99 % of call sites that don't want any of it and (b) put
//! the cache-aware control flow next to the bit-exact path
//! every existing test depends on.
//!
//! Keeping a separate `builder_shared.rs` lets the M5.c.2.c
//! code land as additive surface — the M4.6 / M4.7 / M4.8
//! tests don't touch this file. When the cache-aware path is
//! itself locked by R2 falsifiers (M5.c.2.c → M5.f), the
//! consolidation of the two builders into one is a future-M
//! refactor, not blocking the milestone.
//!
//! ## What this builder does
//!
//! Given a `LlamaConfig`, a `LlamaRuntime`, a `WeightStore`
//! whose `names` cover every parameter the architecture
//! expects, and an optional `KvCacheBuildSpec`, build a
//! complete inference graph:
//!
//!   - **No cache (`kv_cache = None`)**: structurally
//!     identical to `build_llama`, but parameter slots are
//!     [`TensorStorage::CpuShared`] / [`TensorStorage::CpuBf16Shared`]
//!     references over the store's `Arc`s. Output logits
//!     match `build_llama`-then-`load_into` bit-exactly when
//!     the store was populated from the same checkpoint.
//!
//!   - **With cache (`kv_cache = Some(spec)`)**: per-layer
//!     `cache_K` / `cache_V` parameter slots are registered
//!     at shape
//!     `[batch, num_attention_heads, cached_len, head_dim]`
//!     (post-tile cache; the load pipeline already tiles
//!     K/V to MHA). RoPE is applied with
//!     `position_offset = cached_len` for Q and the new K,
//!     while V is unchanged. New K, V get
//!     [`NodeType::Concat`] axis=2'd against the cache slot
//!     to produce `K_full`, `V_full`. The causal mask is
//!     sized `[1, 1, seq, cached_len + seq]`.
//!
//! ## Mask layout
//!
//! For a build with `cached_len = C` and runtime `seq = S`
//! the causal mask `[1, 1, S, C+S]` follows
//! `mask[0, 0, i, j] = -inf if j > i + C else 0`. Specialises:
//!
//!   - **Prefill (`C = 0`, `S = N`)**: classic upper-
//!     triangular `N × N`, identical to `build_llama`.
//!   - **Decode (`C > 0`, `S = 1`)**: row 0 is all zeros
//!     (the new token attends to every cached position plus
//!     itself).
//!   - **Prefill-with-cache (`C > 0`, `S > 1`)**: rows row 0
//!     to S-1 see all `C` cached positions plus their own
//!     prefix in the new chunk.

use super::builder::{LlamaHandles, LlamaRuntime};
use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::{KvCacheBuildSpec, KvCacheHandles, KvLayerHandle};
use crate::amg::weight_store::{SharedParam, WeightStore};
use crate::nn::llama::config::LlamaConfig;
use crate::tensor::Tensor;

/// Output of [`build_llama_with_store`].
///
/// Extends [`LlamaHandles`] with an optional `kv_handles`
/// surface populated when the build spec asked for the
/// cache-aware path.
pub struct LlamaHandlesShared {
    pub token_input_id: usize,
    pub logits_id: usize,
    pub param_ids: Vec<usize>,
    pub param_names: Vec<String>,
    /// Populated when `kv_cache = Some(_)`. `None` when the
    /// build was the bit-exact equivalent of `build_llama`.
    pub kv_handles: Option<KvCacheHandles>,
}

impl LlamaHandlesShared {
    /// Adapter to the legacy [`LlamaHandles`] surface for
    /// callers that don't care about the cache I/O.
    pub fn into_legacy(self) -> LlamaHandles {
        LlamaHandles {
            token_input_id: self.token_input_id,
            logits_id: self.logits_id,
            param_ids: self.param_ids,
            param_names: self.param_names,
        }
    }
}

/// Errors emitted at build time. Most surface from the
/// `WeightStore` lookup ("parameter named X not in store").
#[derive(Debug)]
pub enum BuildError {
    MissingParameter {
        name: String,
    },
    ParameterShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::MissingParameter { name } => write!(
                f,
                "build_llama_with_store: parameter '{name}' not in WeightStore"
            ),
            BuildError::ParameterShapeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "build_llama_with_store: '{name}' has shape {got:?} in store, expected {expected:?}"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Register a Llama parameter slot against a [`WeightStore`].
///
/// On lookup hit, materialises a fresh `Tensor` whose
/// storage is an `Arc` clone of the store's parameter
/// (cheap; no Vec copy). On miss, returns
/// [`BuildError::MissingParameter`]. Validates that the
/// store's recorded shape matches the expected shape.
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
    let tensor = p.to_tensor();
    let node_id = gb.parameter(tensor);
    param_ids.push(node_id);
    param_names.push(full_name.to_string());
    Ok(node_id)
}

/// Register a fresh **graph-local** mutable F32 parameter
/// slot. Used for KV cache buffers that the runtime owns
/// and rewrites every decode step via
/// [`crate::amg::graph::Graph::overwrite_parameter`].
///
/// Not added to `param_ids` / `param_names` — these are not
/// model weights and the WeightMapper must not see them.
fn register_local_zero_f32(gb: &mut GraphBuilder, shape: Vec<usize>) -> usize {
    let numel: usize = shape.iter().product();
    let data = vec![0.0_f32; numel];
    gb.parameter(Tensor::new_cpu(shape, data))
}

/// Build a Llama transformer block backed by the store, with
/// optional cache-aware attention.
fn build_block_shared(
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
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;
    let bs = (batch * seq) as isize;

    // ---- 1. Input layernorm ----
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

    // ---- 2. Q/K/V projections ----
    let q_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.q_proj.weight"),
        vec![hidden, hidden],
        param_ids,
        param_names,
    )?;
    let k_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.k_proj.weight"),
        vec![hidden, hidden],
        param_ids,
        param_names,
    )?;
    let v_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.v_proj.weight"),
        vec![hidden, hidden],
        param_ids,
        param_names,
    )?;

    let h_flat = gb.reshape(h, vec![bs, hidden as isize]);
    let q_flat_raw = gb.matmul(h_flat, q_proj_w);
    let k_flat_raw = gb.matmul(h_flat, k_proj_w);
    let v_flat_raw = gb.matmul(h_flat, v_proj_w);

    // ---- 2.b QKV biases (Qwen 2.5) ----
    let (q_flat, k_flat, v_flat) = if config.effective_attention_bias() {
        let q_proj_b = register_param_from_store(
            gb,
            store,
            &format!("{prefix}.self_attn.q_proj.bias"),
            vec![1, hidden],
            param_ids,
            param_names,
        )?;
        let k_proj_b = register_param_from_store(
            gb,
            store,
            &format!("{prefix}.self_attn.k_proj.bias"),
            vec![1, hidden],
            param_ids,
            param_names,
        )?;
        let v_proj_b = register_param_from_store(
            gb,
            store,
            &format!("{prefix}.self_attn.v_proj.bias"),
            vec![1, hidden],
            param_ids,
            param_names,
        )?;
        (
            gb.broadcast_add(q_flat_raw, q_proj_b),
            gb.broadcast_add(k_flat_raw, k_proj_b),
            gb.broadcast_add(v_flat_raw, v_proj_b),
        )
    } else {
        (q_flat_raw, k_flat_raw, v_flat_raw)
    };

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

    // ---- 4. RoPE on Q and new K ----
    //
    // M5.c.2.c — when a cache spec is present, Q and the new
    // K rotate at absolute position `cached_len + s`, not
    // `s`. Cached K was already rotated at its original
    // position when first produced; we don't re-rotate it.
    let position_offset: u32 = kv_cache.map(|spec| spec.cached_len as u32).unwrap_or(0);
    let (q_rope, k_rope) = match config.effective_rope_scaling() {
        None => (
            gb.rope_with_offset(q, head_dim, config.rope_theta, position_offset),
            gb.rope_with_offset(k, head_dim, config.rope_theta, position_offset),
        ),
        Some(crate::nn::llama::RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        }) => {
            let scaling = crate::amg::nodes::RopeScalingLlama3::new(
                *factor,
                *low_freq_factor,
                *high_freq_factor,
                *original_max_position_embeddings,
            );
            (
                gb.rope_scaled_with_offset(
                    q,
                    head_dim,
                    config.rope_theta,
                    scaling.clone(),
                    position_offset,
                ),
                gb.rope_scaled_with_offset(
                    k,
                    head_dim,
                    config.rope_theta,
                    scaling,
                    position_offset,
                ),
            )
        }
        // **M11.B** — see `builder::build_llama` for the same
        // panic. LongRope must route through a Phi3 builder.
        Some(crate::nn::llama::RopeScaling::LongRope { .. }) => {
            panic!(
                "LongRope rope_scaling reached the Llama builder. \
                 Phi-3 / Phi-3.5 checkpoints must use the Phi3 builder \
                 (M11.B in progress; LongRope executor wire-up still pending)."
            );
        }
    };

    // ---- 5. [b, s, h, d] → [b, h, s, d] ----
    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let new_k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let new_v_perm = gb.permute(v, vec![0, 2, 1, 3]);

    // ---- 6. Concat against cache (cache-aware path only) ----
    //
    // When `kv_cache` is `None`, K_full/V_full = new K/V
    // and the attention path is bit-exact identical to
    // `build_llama`. When `Some(spec)`, we register
    // graph-local cache_K/V parameters (zero-init at build
    // time; runtime overwrites them with the resident cache
    // before each forward) and Concat-axis=2 them with the
    // new K/V projections.
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

    // ---- 7. Attention scores ----
    let k_full_t = gb.transpose_last_two(k_full);
    let scores = gb.batch_matmul(q_perm, k_full_t);
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);
    let attn_weights = gb.softmax(scores_masked);
    let attn_out = gb.batch_matmul(attn_weights, v_full);
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    // ---- 8. Output projection ----
    let attn_out_flat = gb.reshape(attn_out_back, vec![bs, hidden as isize]);
    let o_proj_w = register_param_from_store(
        gb,
        store,
        &format!("{prefix}.self_attn.o_proj.weight"),
        vec![hidden, hidden],
        param_ids,
        param_names,
    )?;
    let attn_proj_flat = gb.matmul(attn_out_flat, o_proj_w);
    let attn_proj = gb.reshape(
        attn_proj_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );
    let x_residual_1 = gb.add(x, attn_proj);

    // ---- 9. Post-attention layernorm ----
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

    // ---- 10. SwiGLU FFN ----
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

/// **M5.c.2.c** — Build a Llama-family graph backed by an
/// [`Arc`-shared `WeightStore`](crate::amg::weight_store::WeightStore),
/// optionally wired for cache-aware decode.
///
/// See the module-level doc-comment for the full contract.
pub fn build_llama_with_store(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
    store: &WeightStore,
    kv_cache: Option<&KvCacheBuildSpec>,
) -> Result<LlamaHandlesShared, BuildError> {
    config.validate().expect("invalid LlamaConfig");

    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();
    let mut kv_handles_inner: Vec<KvLayerHandle> = Vec::new();

    // ---- Causal mask ----
    //
    // Sized `[1, 1, seq, total_kv_seq]` where
    // `total_kv_seq = seq + cached_len`. With cached_len=0
    // (no-cache path) this collapses to the classic
    // [1, 1, seq, seq] upper-triangular form bit-identical
    // to `build_llama`.
    let cached_len = kv_cache.map(|s| s.cached_len).unwrap_or(0);
    let seq = runtime.seq;
    let total_kv_seq = seq + cached_len;
    let mut mask_data = vec![0.0_f32; seq * total_kv_seq];
    for i in 0..seq {
        for j in 0..total_kv_seq {
            // Cached positions (j < cached_len) are always
            // visible. New positions: j - cached_len > i is
            // forbidden (future position).
            if j >= cached_len && (j - cached_len) > i {
                mask_data[i * total_kv_seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, total_kv_seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    // ---- Embedding lookup ----
    let embed_w = register_param_from_store(
        gb,
        store,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    )?;
    let mut x = gb.index_select(embed_w, token_input_id);

    // ---- Transformer blocks ----
    for layer_idx in 0..config.num_hidden_layers {
        x = build_block_shared(
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
            if kv_cache.is_some() {
                Some(&mut kv_handles_inner)
            } else {
                None
            },
        )?;
    }

    // ---- Final RMSNorm ----
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

    // ---- LM head ----
    let bs = (runtime.batch * runtime.seq) as isize;
    let x_flat = gb.reshape(x_final, vec![bs, config.hidden_size as isize]);
    let logits_flat = if config.tie_word_embeddings {
        gb.matmul_rhs_transposed(x_flat, embed_w)
    } else {
        let lm_head_input = register_param_from_store(
            gb,
            store,
            "lm_head.weight",
            vec![config.hidden_size, config.vocab_size],
            &mut param_ids,
            &mut param_names,
        )?;
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
