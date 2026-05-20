//! **AT-1a — `FamilyTensorSpec`: declarative tensor-mapping data.**
//!
//! This module is the data layer for the Adapter Toolkit (ADR-006).
//! It introduces, as pure Rust data, the GGUF->HF name tables, the
//! per-name load-transform rules, and the non-weight tensor set that
//! today live as imperative `match` arms / `if name.contains(...)`
//! ladders scattered across `gguf_to_hf_naming.rs`,
//! `weight_loading.rs`, `phi3.rs`, `gemma2.rs` and
//! `gguf_weight_loading.rs`.
//!
//! **AT-1a scope:** introduce the data + lookups + an equivalence
//! oracle ONLY. No production call-site is rewired here — every
//! existing function keeps its current body. AT-1b rewires the
//! name-map functions; AT-1c rewires the transform functions. The
//! refactor is behaviour-preserving by contract; the golden tests
//! below prove the lookups reproduce the *current* functions
//! byte-for-byte before any rewire happens.
//!
//! **G-1c / GAP-T1** — Gemma 2 GGUF has a *correct*
//! family-specific transform table [`GEMMA2_GGUF_TRANSFORMS`]: it
//! equals the HF table except the RMSNorm rule drops the `+1`
//! fold, because llama.cpp pre-folds `1+γ` into the Gemma 2 GGUF
//! norm weights (measured exactly +1.0 element-wise vs the HF
//! safetensors). The old buggy pre-`.rev()` `GEMMA2_GGUF_GAP1`
//! table and the `TileKvDim1` recipe were removed here.

use crate::v17::loader::gguf_reader::GgufTensorType;
use crate::v17::loader::weight_mapper::LoadTransform;

// ============================================================
// Types
// ============================================================

/// Runtime parameters needed to materialize the parameterized
/// transform recipes. Mirrors the four scalars every current
/// `*_transforms_for_name` function already threads through.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TransformParams {
    pub hidden_size: usize,
    pub head_dim: usize,
    pub kv_groups: usize,
    pub attention_scale: f32,
}

/// Param-free transform recipe. Each variant materializes 1:1 to
/// the exact [`LoadTransform`] the current code emits — see
/// [`materialize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransformRecipe {
    /// `LoadTransform::Transpose2D`
    Transpose2D,
    /// `LoadTransform::Reshape { target: [1, 1, hidden_size] }`
    ReshapeHidden3D,
    /// `LoadTransform::Reshape { target: [1, hidden_size] }` (Qwen2 biases)
    ReshapeHidden2D,
    /// `LoadTransform::AddScalar { scalar: 1.0 }`
    AddScalarOne,
    /// `LoadTransform::Scale { factor: attention_scale }`
    ScaleAttn,
    /// `LoadTransform::TileGroupedDim { dim: 0, group_size: head_dim, repeats: kv_groups }`
    TileKvDim0,
    /// `LoadTransform::Reshape { target: [1, 1, 1, head_dim] }`
    ///
    /// Per-head broadcast for QK-Norm γ (Qwen3 family). Reshapes
    /// a flat `[head_dim]` γ tensor to a rank-4 shape that
    /// broadcasts naturally against `[batch, seq, n_heads,
    /// head_dim]`.
    ReshapeHeadDim4D,
}

/// A single name predicate. The current code uses exactly these
/// three shapes (`name == s`, `name.ends_with(s)`,
/// `name.contains(s)`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum NameMatch {
    Exact(&'static str),
    EndsWith(&'static str),
    Contains(&'static str),
}

impl NameMatch {
    fn is_match(&self, name: &str) -> bool {
        match self {
            NameMatch::Exact(s) => name == *s,
            NameMatch::EndsWith(s) => name.ends_with(s),
            NameMatch::Contains(s) => name.contains(s),
        }
    }
}

/// One transform rule. `any_of` reproduces the `||` groups of the
/// current ladders; rules are evaluated **first-match-wins** in
/// declaration order, and *no* match yields an empty `Vec`
/// (identity) — exactly the current `Vec::new()` fall-through.
pub(crate) struct TransformRule {
    pub any_of: &'static [NameMatch],
    pub recipes: &'static [TransformRecipe],
}

/// A GGUF->HF name table: exact top-level pairs plus
/// `blk.<N>.<suffix>` block-suffix pairs (the HF side is the
/// fragment that follows `model.layers.<N>.`).
pub(crate) struct NameTable {
    pub top_level: &'static [(&'static str, &'static str)],
    pub block_suffix: &'static [(&'static str, &'static str)],
}

/// The declarative bundle for one model family.
pub(crate) struct FamilyTensorSpec {
    /// Diagnostic anchor (matches the adapter `id()`); reserved for
    /// future spec-driven diagnostics, not read on the hot path.
    #[allow(dead_code)]
    pub id: &'static str,
    /// Family-specific *extra* GGUF->HF block names (Phi-3 fused,
    /// Gemma 2 post-norms). Composed with [`COMMON_NAME_TABLE`] by
    /// the adapter's `GgufNameMapper` (unchanged by AT-1). Empty
    /// for the Llama family.
    pub name_extra: NameTable,
    /// HF (safetensors) per-name transform rules.
    pub hf_transforms: &'static [TransformRule],
    /// `Some` when the family needs a GGUF-specific transform
    /// table distinct from `hf_transforms`; `None` when the GGUF
    /// transforms equal the HF table (Llama / Phi-3, handled by
    /// their thin GGUF wrappers). Gemma 2 sets this to
    /// [`GEMMA2_GGUF_TRANSFORMS`] — the HF table minus the RMSNorm
    /// `+1` fold (llama.cpp pre-folds `1+γ` into the Gemma 2 GGUF
    /// norm weights; G-1c / GAP-T1).
    pub gguf_transforms: Option<&'static [TransformRule]>,
    /// **AT-3b** — the union of GGUF tensor dtypes any of this
    /// family's certified GGUF checkpoints actually contains. The
    /// conformance test
    /// `every_adapter_required_dtypes_are_decodable` asserts every
    /// entry is in `decode_tensor`'s supported set, so a future
    /// family declaring an undecodable dtype (e.g. Q2_K) fails at
    /// test time rather than at runtime as `UnsupportedDType` —
    /// the prevention loop ADR-006 wanted for the Phi-3.5 Q5_K
    /// class of bug. Declarative-only: never read on the hot
    /// path; same staging pattern as `FamilyTensorSpec::id`.
    #[allow(dead_code)]
    pub required_gguf_dtypes: &'static [GgufTensorType],
}

// ============================================================
// Lookups (pure)
// ============================================================

/// Materialize a recipe with the runtime params. This is the
/// single 1:1 recipe -> `LoadTransform` mapping; behaviour
/// equivalence rests on this match being exact.
fn materialize(r: TransformRecipe, p: &TransformParams) -> LoadTransform {
    match r {
        TransformRecipe::Transpose2D => LoadTransform::Transpose2D,
        TransformRecipe::ReshapeHidden3D => LoadTransform::Reshape {
            target: vec![1, 1, p.hidden_size],
        },
        TransformRecipe::ReshapeHidden2D => LoadTransform::Reshape {
            target: vec![1, p.hidden_size],
        },
        TransformRecipe::AddScalarOne => LoadTransform::AddScalar { scalar: 1.0 },
        TransformRecipe::ScaleAttn => LoadTransform::Scale {
            factor: p.attention_scale,
        },
        TransformRecipe::TileKvDim0 => LoadTransform::TileGroupedDim {
            dim: 0,
            group_size: p.head_dim,
            repeats: p.kv_groups,
        },
        TransformRecipe::ReshapeHeadDim4D => LoadTransform::Reshape {
            target: vec![1, 1, 1, p.head_dim],
        },
    }
}

/// First-match-wins over `rules`; no match -> empty `Vec`
/// (identity). Reproduces every current `*_transforms_for_name`
/// dispatch when fed that function's rule list.
pub(crate) fn resolve_transforms(
    rules: &[TransformRule],
    name: &str,
    p: &TransformParams,
) -> Vec<LoadTransform> {
    for rule in rules {
        if rule.any_of.iter().any(|m| m.is_match(name)) {
            return rule.recipes.iter().map(|r| materialize(*r, p)).collect();
        }
    }
    Vec::new()
}

/// Resolve a GGUF tensor name to its HF name via `table`. The
/// block-name parsing is byte-identical to
/// `gguf_to_hf_naming::split_blk` (strip `blk.`, split once on
/// `.`, layer must be a non-empty all-ASCII-digit run). Reproduces
/// `gguf_to_hf_name_common` (with [`COMMON_NAME_TABLE`]) and
/// `phi3_gguf_extra` / `gemma2_gguf_extra` (with the family
/// `name_extra`, whose `top_level` is empty).
pub(crate) fn resolve_name(table: &NameTable, gguf_name: &str) -> Option<String> {
    for (g, h) in table.top_level {
        if *g == gguf_name {
            return Some((*h).to_string());
        }
    }
    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    for (s, hf) in table.block_suffix {
        if *s == suffix {
            return Some(format!("model.layers.{layer}.{hf}"));
        }
    }
    None
}

/// GGUF config-input tensors with no HF graph parameter. Reproduces
/// `gguf_to_hf_naming::is_gguf_non_weight_tensor` exactly.
pub(crate) static NON_WEIGHT_TENSORS: &[&str] =
    &["rope_factors_short.weight", "rope_factors_long.weight"];

// ============================================================
// Name tables
// ============================================================

/// Architecture-agnostic GGUF->HF names (the common Llama layout).
/// Mirrors `gguf_to_hf_name_common` verbatim.
pub(crate) static COMMON_NAME_TABLE: NameTable = NameTable {
    top_level: &[
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
        ("rope_freqs.weight", "rope_freqs"),
    ],
    block_suffix: &[
        ("attn_norm.weight", "input_layernorm.weight"),
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
        ("ffn_down.weight", "mlp.down_proj.weight"),
    ],
};

// ============================================================
// Transform rule tables (one per current dispatch function)
// ============================================================

/// Reproduces `weight_loading::compute_transforms_for_name`
/// (Llama / Qwen2 / Mistral HF). Order = the current ladder;
/// embed and any unmatched name fall through to identity.
static LLAMA_HF_TRANSFORMS: &[TransformRule] = &[
    TransformRule {
        any_of: &[
            NameMatch::Exact("model.norm.weight"),
            NameMatch::EndsWith("layernorm.weight"),
        ],
        recipes: &[TransformRecipe::ReshapeHidden3D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.k_proj.weight")],
        recipes: &[
            TransformRecipe::TileKvDim0,
            TransformRecipe::Transpose2D,
            TransformRecipe::ScaleAttn,
        ],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.v_proj.weight")],
        recipes: &[TransformRecipe::TileKvDim0, TransformRecipe::Transpose2D],
    },
    TransformRule {
        any_of: &[
            NameMatch::Contains(".self_attn.q_proj.weight"),
            NameMatch::Contains(".self_attn.o_proj.weight"),
            NameMatch::Contains(".mlp.gate_proj.weight"),
            NameMatch::Contains(".mlp.up_proj.weight"),
            NameMatch::Contains(".mlp.down_proj.weight"),
            NameMatch::Exact("lm_head.weight"),
        ],
        recipes: &[TransformRecipe::Transpose2D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.k_proj.bias")],
        recipes: &[
            TransformRecipe::TileKvDim0,
            TransformRecipe::ReshapeHidden2D,
            TransformRecipe::ScaleAttn,
        ],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.v_proj.bias")],
        recipes: &[TransformRecipe::TileKvDim0, TransformRecipe::ReshapeHidden2D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.q_proj.bias")],
        recipes: &[TransformRecipe::ReshapeHidden2D],
    },
];

/// Reproduces `phi3::phi3_transforms_for_name` (Phi-3 HF; also the
/// Phi-3 GGUF table, since `phi3_gguf_transforms_for_name`
/// delegates to it).
static PHI3_HF_TRANSFORMS: &[TransformRule] = &[
    TransformRule {
        any_of: &[
            NameMatch::Exact("model.norm.weight"),
            NameMatch::EndsWith("layernorm.weight"),
        ],
        recipes: &[TransformRecipe::ReshapeHidden3D],
    },
    TransformRule {
        any_of: &[
            NameMatch::Contains(".self_attn.qkv_proj.weight"),
            NameMatch::Contains(".self_attn.o_proj.weight"),
            NameMatch::Contains(".mlp.gate_up_proj.weight"),
            NameMatch::Contains(".mlp.down_proj.weight"),
            NameMatch::Exact("lm_head.weight"),
        ],
        recipes: &[TransformRecipe::Transpose2D],
    },
];

/// Reproduces `gemma2::gemma2_transforms_for_name` (Gemma 2 HF).
/// Same shape as Llama but the norm rule adds the `+1` fold and
/// there are no QKV biases.
static GEMMA2_HF_TRANSFORMS: &[TransformRule] = &[
    TransformRule {
        any_of: &[
            NameMatch::Exact("model.norm.weight"),
            NameMatch::EndsWith("layernorm.weight"),
        ],
        recipes: &[
            TransformRecipe::ReshapeHidden3D,
            TransformRecipe::AddScalarOne,
        ],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.k_proj.weight")],
        recipes: &[
            TransformRecipe::TileKvDim0,
            TransformRecipe::Transpose2D,
            TransformRecipe::ScaleAttn,
        ],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.v_proj.weight")],
        recipes: &[TransformRecipe::TileKvDim0, TransformRecipe::Transpose2D],
    },
    TransformRule {
        any_of: &[
            NameMatch::Contains(".self_attn.q_proj.weight"),
            NameMatch::Contains(".self_attn.o_proj.weight"),
            NameMatch::Contains(".mlp.gate_proj.weight"),
            NameMatch::Contains(".mlp.up_proj.weight"),
            NameMatch::Contains(".mlp.down_proj.weight"),
            NameMatch::Exact("lm_head.weight"),
        ],
        recipes: &[TransformRecipe::Transpose2D],
    },
];

/// **G-1c / GAP-T1 — the correct Gemma 2 GGUF transform table.**
///
/// Identical to [`GEMMA2_HF_TRANSFORMS`] **except the RMSNorm rule
/// drops the `AddScalarOne` (`+1`) fold**. llama.cpp pre-folds
/// `1 + γ` into the Gemma 2 GGUF norm weights — measured exactly
/// `+1.0` element-wise vs the HF safetensors of the same model
/// across every norm class (`attn_norm`, `ffn_norm`,
/// `post_attention_norm`, `output_norm`). Applying the HF table's
/// `+1` on top of an already-folded GGUF tensor double-folds to
/// `(2+γ)` on every norm → total garbage (this was the real root
/// cause; it masked the RopeUnpermute question, which is decided
/// empirically by the G-1c smoke — primary path = no
/// RopeUnpermute). embed and the Linear/k/v classes are byte
/// identical to the HF table (post-`.rev()` orientation == HF).
static GEMMA2_GGUF_TRANSFORMS: &[TransformRule] = &[
    // RMSNorm: reshape only — NO `+1` (llama.cpp pre-folded it).
    TransformRule {
        any_of: &[
            NameMatch::Exact("model.norm.weight"),
            NameMatch::EndsWith("layernorm.weight"),
        ],
        recipes: &[TransformRecipe::ReshapeHidden3D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.k_proj.weight")],
        recipes: &[
            TransformRecipe::TileKvDim0,
            TransformRecipe::Transpose2D,
            TransformRecipe::ScaleAttn,
        ],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.v_proj.weight")],
        recipes: &[TransformRecipe::TileKvDim0, TransformRecipe::Transpose2D],
    },
    TransformRule {
        any_of: &[
            NameMatch::Contains(".self_attn.q_proj.weight"),
            NameMatch::Contains(".self_attn.o_proj.weight"),
            NameMatch::Contains(".mlp.gate_proj.weight"),
            NameMatch::Contains(".mlp.up_proj.weight"),
            NameMatch::Contains(".mlp.down_proj.weight"),
            NameMatch::Exact("lm_head.weight"),
        ],
        recipes: &[TransformRecipe::Transpose2D],
    },
];

/// **Phase Q (Qwen3 family support).** Same shape as Llama but
/// with **per-head QK-Norm γ rules** for `*.self_attn.q_norm.weight`
/// / `*.self_attn.k_norm.weight` (reshape `[head_dim]` -> `[1, 1, 1,
/// head_dim]` so it broadcasts against `[batch, seq, n_heads,
/// head_dim]`), and the `1/√head_dim` attention scale **moved from
/// k_proj to k_norm γ** (it would be stripped by the K-Norm RMSNorm
/// if left on k_proj; absorbed into the post-norm γ where it
/// survives). The Linear class loses ScaleAttn on k_proj.weight.
/// No QKV biases. Rule order: QK-Norm specifics first
/// (`q_norm.weight` / `k_norm.weight`) so the generic norm rule
/// below them doesn't accidentally match (it won't — neither name
/// ends with "layernorm.weight" — but the ordering documents
/// intent and is safe against future rule additions).
static QWEN3_HF_TRANSFORMS: &[TransformRule] = &[
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.q_norm.weight")],
        recipes: &[TransformRecipe::ReshapeHeadDim4D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.k_norm.weight")],
        recipes: &[
            TransformRecipe::ReshapeHeadDim4D,
            TransformRecipe::ScaleAttn,
        ],
    },
    TransformRule {
        any_of: &[
            NameMatch::Exact("model.norm.weight"),
            NameMatch::EndsWith("layernorm.weight"),
        ],
        recipes: &[TransformRecipe::ReshapeHidden3D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.k_proj.weight")],
        // **NO ScaleAttn here** (the K-Norm RMSNorm would strip a
        // pre-normalize multiplicative scale; the 1/√d attention
        // scale lives in k_norm γ for Qwen3).
        recipes: &[TransformRecipe::TileKvDim0, TransformRecipe::Transpose2D],
    },
    TransformRule {
        any_of: &[NameMatch::Contains(".self_attn.v_proj.weight")],
        recipes: &[TransformRecipe::TileKvDim0, TransformRecipe::Transpose2D],
    },
    TransformRule {
        any_of: &[
            NameMatch::Contains(".self_attn.q_proj.weight"),
            NameMatch::Contains(".self_attn.o_proj.weight"),
            NameMatch::Contains(".mlp.gate_proj.weight"),
            NameMatch::Contains(".mlp.up_proj.weight"),
            NameMatch::Contains(".mlp.down_proj.weight"),
            NameMatch::Exact("lm_head.weight"),
        ],
        recipes: &[TransformRecipe::Transpose2D],
    },
];

// ============================================================
// Per-family specs
// ============================================================

/// Llama / Qwen2 / Mistral. No GGUF name extras (default common
/// mapping); no bespoke GGUF transform table (the GGUF wrapper
/// prepends the rope-unpermute and reuses the HF rules).
pub(crate) static LLAMA_SPEC: FamilyTensorSpec = FamilyTensorSpec {
    id: "llama",
    name_extra: NameTable {
        top_level: &[],
        block_suffix: &[],
    },
    hf_transforms: LLAMA_HF_TRANSFORMS,
    gguf_transforms: None,
    // TinyLlama Q4_K_M / Q8_0, SmolLM2 1.7B Q4_K_M,
    // Llama-3.2-1B Q4_K_M: F32 norms, F16 / Q8_0 / Q4_K / Q6_K
    // weight quants.
    required_gguf_dtypes: &[
        GgufTensorType::F32,
        GgufTensorType::F16,
        GgufTensorType::Q4_K,
        GgufTensorType::Q6_K,
        GgufTensorType::Q8_0,
    ],
};

/// **Phase Q (Qwen3 family support).** Llama-like name layout
/// (no GGUF extras; the canonical Qwen3 GGUF conversion mirrors
/// the common Llama-layout table). HF transforms add per-head
/// QK-Norm γ rules and move the attention scale from k_proj to
/// k_norm γ. GGUF support is out of scope this phase:
/// `gguf_transforms: None`, `required_gguf_dtypes` empty.
pub(crate) static QWEN3_SPEC: FamilyTensorSpec = FamilyTensorSpec {
    id: "qwen3",
    name_extra: NameTable {
        top_level: &[],
        block_suffix: &[],
    },
    hf_transforms: QWEN3_HF_TRANSFORMS,
    gguf_transforms: None,
    required_gguf_dtypes: &[],
};

/// Phi-3: fused QKV + fused gate_up name extras.
pub(crate) static PHI3_SPEC: FamilyTensorSpec = FamilyTensorSpec {
    id: "phi3",
    name_extra: NameTable {
        top_level: &[],
        block_suffix: &[
            ("attn_qkv.weight", "self_attn.qkv_proj.weight"),
            ("ffn_up.weight", "mlp.gate_up_proj.weight"),
        ],
    },
    hf_transforms: PHI3_HF_TRANSFORMS,
    gguf_transforms: None,
    // Phi-3.5-mini Q4_K_M: F32 norms + Q4_K / Q5_K / Q6_K (Q5_K
    // was the dtype the Phi-3.5 phase added to decode_tensor).
    required_gguf_dtypes: &[
        GgufTensorType::F32,
        GgufTensorType::Q4_K,
        GgufTensorType::Q5_K,
        GgufTensorType::Q6_K,
    ],
};

/// Gemma 2: post/pre norm name extras, plus the GGUF-specific
/// transform table (HF table minus the RMSNorm `+1` fold —
/// llama.cpp pre-folds it; G-1c / GAP-T1).
pub(crate) static GEMMA2_SPEC: FamilyTensorSpec = FamilyTensorSpec {
    id: "gemma2",
    name_extra: NameTable {
        top_level: &[],
        block_suffix: &[
            ("attn_post_norm.weight", "post_attention_layernorm.weight"),
            ("post_attention_norm.weight", "post_attention_layernorm.weight"),
            ("ffn_post_norm.weight", "post_feedforward_layernorm.weight"),
            // **G-1a / GAP-N1** — the real llama.cpp Gemma 2 GGUF
            // post-FFN norm tensor is `post_ffw_norm.weight`
            // (verified on bartowski/gemma-2-2b-it Q4_K_M). The
            // pre-existing `attn_post_norm.weight` /
            // `ffn_post_norm.weight` entries were authored from
            // guessed names and never matched a real checkpoint;
            // they are kept for now (conformance-pinned) and this
            // additive entry unblocks the Gemma 2 GGUF load.
            ("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
            // **G-1b2 / GAP-N2** — Gemma 2 has FOUR per-layer
            // norms. In llama.cpp `ffn_norm` is the *pre-FFN* norm
            // = HF `pre_feedforward_layernorm` (the common
            // Llama-layout table maps `ffn_norm.weight` to
            // `post_attention_layernorm.weight`, correct only for
            // the 2-norm Llama layout). This family override must
            // win over the common table, which requires the
            // `Gemma2Adapter` GgufNameMapper to compose extra-first
            // (mirrors `Phi3Adapter`; see model_adapters/gemma2.rs).
            ("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
        ],
    },
    hf_transforms: GEMMA2_HF_TRANSFORMS,
    gguf_transforms: Some(GEMMA2_GGUF_TRANSFORMS),
    // gemma-2-2b-it Q4_K_M (inspector: F32 norms + Q4_K weights +
    // Q6_K embed / v_proj / down_proj).
    required_gguf_dtypes: &[
        GgufTensorType::F32,
        GgufTensorType::Q4_K,
        GgufTensorType::Q6_K,
    ],
};

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::llama::gemma2::gemma2_transforms_for_name;
    use crate::nn::llama::gguf_weight_loading::gemma2_gguf_transforms_for_name;
    use crate::nn::llama::phi3::phi3_transforms_for_name;
    use crate::nn::llama::weight_loading::compute_transforms_for_name;
    use crate::v17::loader::gguf_to_hf_naming::{
        gemma2_gguf_extra, gguf_to_hf_name_common, is_gguf_non_weight_tensor, phi3_gguf_extra,
    };

    fn p() -> TransformParams {
        TransformParams {
            hidden_size: 10,
            head_dim: 4,
            kv_groups: 3,
            attention_scale: 0.5,
        }
    }

    // ---- materialize: each recipe -> exact LoadTransform ----

    #[test]
    fn materialize_maps_each_recipe_exactly() {
        let p = p();
        assert_eq!(
            materialize(TransformRecipe::Transpose2D, &p),
            LoadTransform::Transpose2D
        );
        assert_eq!(
            materialize(TransformRecipe::ReshapeHidden3D, &p),
            LoadTransform::Reshape {
                target: vec![1, 1, 10]
            }
        );
        assert_eq!(
            materialize(TransformRecipe::ReshapeHidden2D, &p),
            LoadTransform::Reshape { target: vec![1, 10] }
        );
        assert_eq!(
            materialize(TransformRecipe::AddScalarOne, &p),
            LoadTransform::AddScalar { scalar: 1.0 }
        );
        assert_eq!(
            materialize(TransformRecipe::ScaleAttn, &p),
            LoadTransform::Scale { factor: 0.5 }
        );
        assert_eq!(
            materialize(TransformRecipe::TileKvDim0, &p),
            LoadTransform::TileGroupedDim {
                dim: 0,
                group_size: 4,
                repeats: 3
            }
        );
    }

    // ---- resolve_transforms: ordering / OR / default ----

    #[test]
    fn resolve_transforms_is_first_match_wins_with_identity_default() {
        static RULES: &[TransformRule] = &[
            TransformRule {
                any_of: &[NameMatch::Contains("foo")],
                recipes: &[TransformRecipe::Transpose2D],
            },
            TransformRule {
                any_of: &[NameMatch::Contains("foo"), NameMatch::Exact("bar")],
                recipes: &[TransformRecipe::AddScalarOne],
            },
        ];
        // "xfoox" matches rule 0 first (first-match-wins).
        assert_eq!(
            resolve_transforms(RULES, "xfoox", &p()),
            vec![LoadTransform::Transpose2D]
        );
        // "bar" matches only rule 1 (OR over any_of).
        assert_eq!(
            resolve_transforms(RULES, "bar", &p()),
            vec![LoadTransform::AddScalar { scalar: 1.0 }]
        );
        // no match -> identity (empty).
        assert_eq!(
            resolve_transforms(RULES, "nothing", &p()),
            Vec::<LoadTransform>::new()
        );
    }

    // ---- resolve_name: top-level / block / None ----

    #[test]
    fn resolve_name_top_level_block_and_none() {
        assert_eq!(
            resolve_name(&COMMON_NAME_TABLE, "token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            resolve_name(&COMMON_NAME_TABLE, "blk.7.attn_q.weight").as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        // malformed layer / non-blk / unknown suffix -> None
        assert_eq!(resolve_name(&COMMON_NAME_TABLE, "blk.x.attn_q.weight"), None);
        assert_eq!(resolve_name(&COMMON_NAME_TABLE, "general.foo"), None);
        assert_eq!(
            resolve_name(&COMMON_NAME_TABLE, "blk.0.unknown.weight"),
            None
        );
        // family extra (block-only): Phi-3 fused.
        assert_eq!(
            resolve_name(&PHI3_SPEC.name_extra, "blk.3.attn_qkv.weight").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        assert_eq!(
            resolve_name(&PHI3_SPEC.name_extra, "token_embd.weight"),
            None,
            "phi3 extra has no top-level entries"
        );
    }

    // ---- GOLDEN equivalence: spec == live current functions ----
    //
    // This is the AT-1a oracle. It proves the declarative lookups
    // reproduce the CURRENT production functions byte-for-byte,
    // BEFORE any call-site is rewired (AT-1b/c). The reference side
    // is the live `*_transforms_for_name` / `*_gguf_*` / name fns,
    // imported read-only (no file is modified, no call-site changed).

    const CORPUS: &[&str] = &[
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.3.pre_feedforward_layernorm.weight",
        "model.layers.3.post_feedforward_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "totally.unknown.weight",
    ];

    #[test]
    fn golden_llama_hf_matches_live_compute_transforms() {
        let (h, hd, kvg, s) = (2048usize, 64usize, 8usize, 0.125f32);
        let tp = TransformParams {
            hidden_size: h,
            head_dim: hd,
            kv_groups: kvg,
            attention_scale: s,
        };
        for n in CORPUS {
            assert_eq!(
                resolve_transforms(LLAMA_HF_TRANSFORMS, n, &tp),
                compute_transforms_for_name(n, h, hd, kvg, s),
                "llama HF spec/live divergence for '{n}'"
            );
        }
    }

    #[test]
    fn golden_phi3_hf_matches_live_phi3_transforms() {
        let h = 3072usize;
        let tp = TransformParams {
            hidden_size: h,
            head_dim: 96,
            kv_groups: 1,
            attention_scale: 0.1,
        };
        for n in CORPUS {
            assert_eq!(
                resolve_transforms(PHI3_HF_TRANSFORMS, n, &tp),
                phi3_transforms_for_name(n, h),
                "phi3 HF spec/live divergence for '{n}'"
            );
        }
    }

    #[test]
    fn golden_gemma2_hf_matches_live_gemma2_transforms() {
        let (h, hd, kvg, s) = (2304usize, 256usize, 2usize, 1.0f32 / 16.0f32);
        let tp = TransformParams {
            hidden_size: h,
            head_dim: hd,
            kv_groups: kvg,
            attention_scale: s,
        };
        for n in CORPUS {
            assert_eq!(
                resolve_transforms(GEMMA2_HF_TRANSFORMS, n, &tp),
                gemma2_transforms_for_name(n, h, hd, kvg, s),
                "gemma2 HF spec/live divergence for '{n}'"
            );
        }
    }

    /// **G-1c / GAP-T1** — the corrected Gemma 2 GGUF table. The
    /// live `gemma2_gguf_transforms_for_name` delegates to
    /// `GEMMA2_GGUF_TRANSFORMS` (primary path, no RopeUnpermute).
    /// The defining property: the GGUF norm rule is
    /// `[ReshapeHidden3D]` ONLY (llama.cpp pre-folds the Gemma 2
    /// `+1` into the GGUF norm weights — measured exactly +1.0
    /// element-wise vs HF), whereas the HF table keeps
    /// `[ReshapeHidden3D, AddScalarOne]`. Every non-norm class is
    /// identical between the GGUF and HF tables.
    #[test]
    fn golden_gemma2_gguf_norm_fold() {
        let (h, hd, kvg, s) = (2304usize, 256usize, 2usize, 1.0f32 / 16.0f32);
        let tp = TransformParams {
            hidden_size: h,
            head_dim: hd,
            kv_groups: kvg,
            attention_scale: s,
        };
        for n in CORPUS {
            assert_eq!(
                resolve_transforms(GEMMA2_GGUF_TRANSFORMS, n, &tp),
                gemma2_gguf_transforms_for_name(n, h, hd, kvg, s),
                "gemma2 GGUF spec/live divergence for '{n}'"
            );
        }
        let norm = "model.layers.0.post_feedforward_layernorm.weight";
        assert_eq!(
            resolve_transforms(GEMMA2_GGUF_TRANSFORMS, norm, &tp),
            vec![LoadTransform::Reshape {
                target: vec![1, 1, h]
            }],
            "gemma2 GGUF norm = Reshape only (llama.cpp pre-folded the +1)"
        );
        assert_eq!(
            resolve_transforms(GEMMA2_HF_TRANSFORMS, norm, &tp),
            vec![
                LoadTransform::Reshape {
                    target: vec![1, 1, h]
                },
                LoadTransform::AddScalar { scalar: 1.0 },
            ],
            "gemma2 HF norm keeps the +1 (raw gamma in safetensors)"
        );
        for n in [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ] {
            assert_eq!(
                resolve_transforms(GEMMA2_GGUF_TRANSFORMS, n, &tp),
                resolve_transforms(GEMMA2_HF_TRANSFORMS, n, &tp),
                "gemma2 GGUF/HF must be identical for non-norm '{n}'"
            );
        }
    }

    #[test]
    fn golden_name_maps_match_live_functions() {
        let name_corpus = [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "rope_freqs.weight",
            "blk.0.attn_norm.weight",
            "blk.5.attn_q.weight",
            "blk.5.attn_k.weight",
            "blk.5.attn_v.weight",
            "blk.5.attn_output.weight",
            "blk.5.ffn_norm.weight",
            "blk.5.ffn_gate.weight",
            "blk.5.ffn_up.weight",
            "blk.5.ffn_down.weight",
            "blk.3.attn_qkv.weight",
            "blk.3.attn_post_norm.weight",
            "blk.3.post_attention_norm.weight",
            "blk.3.ffn_post_norm.weight",
            "blk.x.attn_q.weight",
            "general.foo",
            "blk.0.unknown.weight",
            "not_a_tensor",
        ];
        for n in name_corpus {
            assert_eq!(
                resolve_name(&COMMON_NAME_TABLE, n),
                gguf_to_hf_name_common(n),
                "common name spec/live divergence for '{n}'"
            );
            assert_eq!(
                resolve_name(&PHI3_SPEC.name_extra, n),
                phi3_gguf_extra(n),
                "phi3 extra spec/live divergence for '{n}'"
            );
            assert_eq!(
                resolve_name(&GEMMA2_SPEC.name_extra, n),
                gemma2_gguf_extra(n),
                "gemma2 extra spec/live divergence for '{n}'"
            );
        }
    }

    #[test]
    fn golden_non_weight_set_matches_live_function() {
        for n in [
            "rope_factors_short.weight",
            "rope_factors_long.weight",
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "rope_freqs.weight",
            "output.weight",
            "unknown",
        ] {
            assert_eq!(
                NON_WEIGHT_TENSORS.contains(&n),
                is_gguf_non_weight_tensor(n),
                "non-weight set spec/live divergence for '{n}'"
            );
        }
    }

    /// **G-1a / GAP-N1** — the real llama.cpp Gemma 2 GGUF
    /// post-norm tensor names (verified on bartowski/gemma-2-2b-it
    /// Q4_K_M: `post_attention_norm.weight` + `post_ffw_norm.weight`)
    /// must resolve through the production `gemma2_gguf_extra`. The
    /// `post_ffw_norm.weight` mapping is the additive G-1a fix that
    /// unblocks the Gemma 2 GGUF load; `post_attention_norm.weight`
    /// was already covered. The phantom entries kept for now must
    /// still map (regression — they are conformance-pinned).
    #[test]
    fn gemma2_real_gguf_post_norm_names_resolve() {
        // Real checkpoint tensor names.
        assert_eq!(
            gemma2_gguf_extra("blk.7.post_attention_norm.weight").as_deref(),
            Some("model.layers.7.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.7.post_ffw_norm.weight").as_deref(),
            Some("model.layers.7.post_feedforward_layernorm.weight"),
            "G-1a: real Gemma 2 GGUF post-FFN norm must map (was the GAP-N1 load blocker)"
        );
        // Phantom entries (kept for now, conformance-pinned) still map.
        assert_eq!(
            gemma2_gguf_extra("blk.7.attn_post_norm.weight").as_deref(),
            Some("model.layers.7.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.7.ffn_post_norm.weight").as_deref(),
            Some("model.layers.7.post_feedforward_layernorm.weight")
        );
        // Unrelated / unknown still None.
        assert_eq!(gemma2_gguf_extra("blk.7.attn_q.weight"), None);
        assert_eq!(gemma2_gguf_extra("post_ffw_norm.weight"), None);
    }
}
