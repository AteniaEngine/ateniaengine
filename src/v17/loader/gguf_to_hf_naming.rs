//! **AT-1b** — the GGUF→HF name maps are now driven by the
//! declarative [`crate::model_adapters::tensor_spec`]
//! `FamilyTensorSpec` data (ADR-006). The public functions below
//! keep their exact signatures and behaviour; only their bodies
//! changed from hand-written `match` arms to a `resolve_name`
//! lookup over the spec tables. The `split_blk` block parser was
//! removed: its logic is byte-identically embedded in
//! `tensor_spec::resolve_name` (proved by the AT-1a golden
//! `golden_name_maps_match_live_functions`, kept green here too).

use crate::model_adapters::tensor_spec::{
    COMMON_NAME_TABLE, GEMMA2_SPEC, GEMMA3_SPEC, NON_WEIGHT_TENSORS, PHI3_SPEC, QWEN3_SPEC,
    resolve_name,
};

/// **Phase 16.1 / AT-1b** — architecture-agnostic GGUF → HF name
/// mapping: the top-level tensors plus the common Llama-layout
/// block suffixes shared by every supported family. Returns `None`
/// for family-specific suffixes (handled by the `*_gguf_extra`
/// functions) and for unknown / malformed names. Body delegates to
/// [`resolve_name`] over [`COMMON_NAME_TABLE`].
pub fn gguf_to_hf_name_common(gguf_name: &str) -> Option<String> {
    resolve_name(&COMMON_NAME_TABLE, gguf_name)
}

/// **Phase 16 / AT-1b** — Phi-3-specific GGUF suffixes that the
/// common Llama-layout table maps differently. Phi-3 fuses both
/// the QKV projection (`attn_qkv` → `qkv_proj`) and the MLP
/// gate/up projection: llama.cpp stores the latter as a single
/// `blk.N.ffn_up.weight` of width `2 * intermediate_size`, which
/// the common table would mis-map to the separate `up_proj` —
/// Phi-3's graph expects the fused `gate_up_proj`. These overrides
/// must be tried *before* `gguf_to_hf_name_common` (the common
/// table also matches `ffn_up.weight`), see `Phi3Adapter`'s
/// `GgufNameMapper`. Body delegates to [`resolve_name`] over
/// `PHI3_SPEC.name_extra`.
pub fn phi3_gguf_extra(gguf_name: &str) -> Option<String> {
    resolve_name(&PHI3_SPEC.name_extra, gguf_name)
}

/// **Phase 16.1 / AT-1b** — Gemma 2-specific GGUF suffix(es) not
/// covered by [`gguf_to_hf_name_common`]: the post-attention /
/// post-FFN norms. Body delegates to [`resolve_name`] over
/// `GEMMA2_SPEC.name_extra`.
pub fn gemma2_gguf_extra(gguf_name: &str) -> Option<String> {
    resolve_name(&GEMMA2_SPEC.name_extra, gguf_name)
}

/// **Qwen GGUF support** — Qwen3-specific GGUF suffixes not
/// covered by [`gguf_to_hf_name_common`]: the per-head QK-Norm γ
/// tensors (`blk.N.attn_{q,k}_norm.weight`). Body delegates to
/// [`resolve_name`] over `QWEN3_SPEC.name_extra`.
pub fn qwen3_gguf_extra(gguf_name: &str) -> Option<String> {
    resolve_name(&QWEN3_SPEC.name_extra, gguf_name)
}

/// **Gemma 3 GGUF support** — Gemma 3-specific GGUF suffixes: the
/// four per-layer norms (as Gemma 2) plus the per-head QK-Norm γ
/// tensors (`blk.N.attn_{q,k}_norm.weight`). Body delegates to
/// [`resolve_name`] over `GEMMA3_SPEC.name_extra`. Must be tried
/// **before** [`gguf_to_hf_name_common`] (the `ffn_norm` override
/// shadows the common 2-norm mapping — same composition class as
/// Gemma 2).
pub fn gemma3_gguf_extra(gguf_name: &str) -> Option<String> {
    resolve_name(&GEMMA3_SPEC.name_extra, gguf_name)
}

// **Phase 16.3** — the arch-branching composing free function
// `gguf_to_hf_name(name, arch)` was removed. The core no longer
// owns GGUF→HF name structure: `pipeline::build_gguf_name_map`
// calls `adapter.gguf_to_hf_name(name)` (the `GgufNameMapper`
// trait), which composes `gguf_to_hf_name_common` with the
// family extras. The structure now lives as data in
// `tensor_spec`; the family dispatch still lives behind the
// adapter.

/// GGUF tensors that are **config inputs, not graph weights**:
/// they carry no HuggingFace parameter equivalent (HF keeps the
/// same data in `config.json`). The canonical llama.cpp Phi-3
/// LongRope GGUF stores its scaling factors as the top-level
/// tensors `rope_factors_{short,long}.weight`; these are consumed
/// at config-parse time by
/// `gguf_config::gguf_rope_scaling_json` and must be **skipped**
/// by the weight-name mapping (`build_gguf_name_map` /
/// `gguf_tensor_metas`) instead of hard-erroring as "no known HF
/// name mapping". Name-based (not family-dispatched) because the
/// convention is fixed across any longrope GGUF; no real graph
/// parameter is ever named this way. **AT-1b**: backed by the
/// declarative [`NON_WEIGHT_TENSORS`] set.
pub fn is_gguf_non_weight_tensor(gguf_name: &str) -> bool {
    NON_WEIGHT_TENSORS.contains(&gguf_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_top_level_names() {
        assert_eq!(
            gguf_to_hf_name_common("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            gguf_to_hf_name_common("output_norm.weight").as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(
            gguf_to_hf_name_common("output.weight").as_deref(),
            Some("lm_head.weight")
        );
    }

    #[test]
    fn maps_llama_block_names() {
        let cases = [
            (
                "blk.7.attn_norm.weight",
                "model.layers.7.input_layernorm.weight",
            ),
            (
                "blk.7.attn_q.weight",
                "model.layers.7.self_attn.q_proj.weight",
            ),
            (
                "blk.7.attn_k.weight",
                "model.layers.7.self_attn.k_proj.weight",
            ),
            (
                "blk.7.attn_v.weight",
                "model.layers.7.self_attn.v_proj.weight",
            ),
            (
                "blk.7.attn_output.weight",
                "model.layers.7.self_attn.o_proj.weight",
            ),
            (
                "blk.7.ffn_norm.weight",
                "model.layers.7.post_attention_layernorm.weight",
            ),
            (
                "blk.7.ffn_gate.weight",
                "model.layers.7.mlp.gate_proj.weight",
            ),
            ("blk.7.ffn_up.weight", "model.layers.7.mlp.up_proj.weight"),
            (
                "blk.7.ffn_down.weight",
                "model.layers.7.mlp.down_proj.weight",
            ),
        ];
        for (gguf, hf) in cases {
            assert_eq!(gguf_to_hf_name_common(gguf).as_deref(), Some(hf));
        }
    }

    #[test]
    fn maps_architecture_specific_names() {
        assert_eq!(
            phi3_gguf_extra("blk.3.attn_qkv.weight").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.3.attn_post_norm.weight").as_deref(),
            Some("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.3.ffn_post_norm.weight").as_deref(),
            Some("model.layers.3.post_feedforward_layernorm.weight")
        );
    }

    #[test]
    fn rejects_unknown_or_malformed_names() {
        assert_eq!(gguf_to_hf_name_common("general.foo"), None);
        assert_eq!(gguf_to_hf_name_common("blk.x.attn_q.weight"), None);
        assert_eq!(gguf_to_hf_name_common("blk.0.unknown.weight"), None);
        // `attn_qkv` is Phi-3-specific: not in the common set.
        assert_eq!(gguf_to_hf_name_common("blk.0.attn_qkv.weight"), None);
    }

    // ----- Phase 16.1: the split functions, exercised directly -----

    #[test]
    fn common_covers_top_level_and_llama_block_only() {
        assert_eq!(
            gguf_to_hf_name_common("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            gguf_to_hf_name_common("blk.7.attn_q.weight").as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        // Family-specific suffixes are NOT common.
        assert_eq!(gguf_to_hf_name_common("blk.3.attn_qkv.weight"), None);
        assert_eq!(gguf_to_hf_name_common("blk.3.ffn_post_norm.weight"), None);
        // Malformed still rejected.
        assert_eq!(gguf_to_hf_name_common("blk.x.attn_q.weight"), None);
    }

    #[test]
    fn phi3_extra_only_handles_fused_qkv() {
        assert_eq!(
            phi3_gguf_extra("blk.3.attn_qkv.weight").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        // Phi-3 fuses gate/up into a single `ffn_up.weight`; it
        // must map to the fused `gate_up_proj`, NOT the common
        // `up_proj`.
        assert_eq!(
            phi3_gguf_extra("blk.3.ffn_up.weight").as_deref(),
            Some("model.layers.3.mlp.gate_up_proj.weight")
        );
        assert_eq!(phi3_gguf_extra("blk.3.attn_q.weight"), None);
        assert_eq!(phi3_gguf_extra("blk.3.ffn_down.weight"), None);
        assert_eq!(phi3_gguf_extra("token_embd.weight"), None);
        // Regression: the common table is unchanged — `ffn_up`
        // still maps to the separate `up_proj` there (llama path).
        assert_eq!(
            gguf_to_hf_name_common("blk.3.ffn_up.weight").as_deref(),
            Some("model.layers.3.mlp.up_proj.weight")
        );
    }

    #[test]
    fn gemma2_extra_only_handles_post_norms() {
        assert_eq!(
            gemma2_gguf_extra("blk.3.attn_post_norm.weight").as_deref(),
            Some("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.3.post_attention_norm.weight").as_deref(),
            Some("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.3.ffn_post_norm.weight").as_deref(),
            Some("model.layers.3.post_feedforward_layernorm.weight")
        );
        assert_eq!(gemma2_gguf_extra("blk.3.attn_q.weight"), None);
    }

    #[test]
    fn non_weight_tensors_are_the_longrope_factors_only() {
        assert!(is_gguf_non_weight_tensor("rope_factors_short.weight"));
        assert!(is_gguf_non_weight_tensor("rope_factors_long.weight"));
        // Real graph weights / unknowns must NOT be treated as skippable.
        assert!(!is_gguf_non_weight_tensor("token_embd.weight"));
        assert!(!is_gguf_non_weight_tensor("blk.0.attn_q.weight"));
        assert!(!is_gguf_non_weight_tensor("blk.3.attn_qkv.weight"));
        assert!(!is_gguf_non_weight_tensor("output.weight"));
    }
}
