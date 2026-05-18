//! Llama-family WeightMapper builder.
//!
//! Encapsulates the per-tensor transform list needed to load a
//! HuggingFace Llama-family checkpoint (TinyLlama, SmolLM2,
//! Llama 3.x, Qwen 2.5, ...) into an Atenia
//! graph. The builder graph layer (M4.5-b1, Paso 3) registers
//! parameters with the names this helper expects; this helper does
//! not touch the graph itself.
//!
//! Transformations applied:
//!
//! - **Embedding tables** (`model.embed_tokens.weight`): identity.
//!   The embed table's HF layout `[vocab, hidden]` is consumed
//!   directly by `IndexSelect`.
//! - **RMSNorm gammas** (`*.layernorm.weight`, `model.norm.weight`):
//!   identity. 1D `[hidden]`, used as-is.
//! - **`q_proj` / `o_proj` / MLP `gate_proj` / `up_proj` /
//!   `down_proj` / `lm_head`**: `[Transpose2D]` only. Converts HF
//!   `[out, in]` → Atenia `[in, out]`.
//! - **`k_proj`**: `[TileGroupedDim, Transpose2D, Scale]`. The tile
//!   expands GQA K to MHA-equivalent shape; the scale absorbs
//!   `1/sqrt(head_dim)` into the weight, eliminating a runtime
//!   scaling node.
//! - **`v_proj`**: `[TileGroupedDim, Transpose2D]`. Same GQA
//!   expansion, no scale (V is not part of the QK product).
//!
//! ## Phase B (Qwen 2.5 family) — QKV biases
//!
//! Activated when `LlamaConfig::effective_attention_bias()` returns
//! true (currently: Qwen2 model_type without an explicit
//! `attention_bias` field). Three additional 1D tensor families
//! per layer:
//!
//! - **`q_proj.bias`** (`[hidden]`): `[Reshape([1, hidden])]`. The
//!   rank-2 layout matches the same-rank rule of `BroadcastAdd`
//!   when added to the `[batch*seq, hidden]` projection output.
//! - **`k_proj.bias`** (`[kv_heads * head_dim]`):
//!   `[TileGroupedDim, Reshape([1, hidden]), Scale]`. Mirrors the
//!   K weight transform: GQA expansion to MHA shape, plus the
//!   `1/sqrt(head_dim)` absorption that eliminates a runtime
//!   scale. The scale must apply to the bias too so that
//!   `Q @ (h W_k + b_k)^T / sqrt(d_k)` ≡ `Q @ k_atenia^T` holds
//!   exactly.
//! - **`v_proj.bias`** (`[kv_heads * head_dim]`):
//!   `[TileGroupedDim, Reshape([1, hidden])]`. GQA expansion only;
//!   V never enters the QK product, so no scale.

use crate::model_adapters::tensor_spec::{LLAMA_SPEC, TransformParams, resolve_transforms};
use crate::nn::llama::LlamaConfig;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::{LoadTransform, WeightMapper};

/// Build a `WeightMapper` pre-configured for a Llama-family
/// HuggingFace checkpoint. The caller supplies the parameter
/// names/ids produced by the graph builder; this function attaches
/// the appropriate transforms by inspecting each name.
///
/// Returns `LoaderError::InvalidFormat` only if the mapper itself
/// cannot be constructed (length mismatch or duplicate names —
/// caller bug). Unknown parameter names get an empty transform list
/// (defensive default), so the mapper is permissive about extra
/// names that future architectures may add.
pub fn llama_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;

    let head_dim = config.head_dim();
    let kv_groups = config.kv_groups();
    let attention_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms =
            compute_transforms_for_name(name, hidden_size, head_dim, kv_groups, attention_scale);
        if !transforms.is_empty() {
            // Cannot fail: every name in `param_names` was just
            // inserted into the mapper above.
            mapper.set_transforms(name, transforms)?;
        }
    }

    Ok(mapper)
}

/// Compute the transform list for a single HF parameter name.
/// `pub` so integration tests can verify the per-name dispatch
/// directly without going through `llama_weight_mapper`.
///
/// **AT-1c**: the per-name ladder is now declarative data in
/// [`crate::model_adapters::tensor_spec`] (`LLAMA_SPEC.hf_transforms`).
/// Signature and behaviour are unchanged — `resolve_transforms` is
/// first-match-wins with an identity (`Vec::new()`) default,
/// exactly reproducing the previous embed / norm / k / v /
/// {q,o,gate,up,down,lm_head} / k_bias / v_bias / q_bias ladder.
/// Equivalence is pinned by the AT-1a golden
/// `golden_llama_hf_matches_live_compute_transforms` and the AT-2
/// conformance suite.
pub fn compute_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    resolve_transforms(
        LLAMA_SPEC.hf_transforms,
        name,
        &TransformParams {
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        },
    )
}
