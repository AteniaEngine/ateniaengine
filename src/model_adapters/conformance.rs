//! **AT-2 — Adapter conformance harness.**
//!
//! Test-only. This module introduces *no* production logic. It is an
//! executable freeze of the adapter layer's *current* behaviour, to
//! act as the oracle for the AT-1 declarative refactor (ADR-006).
//!
//! Every assertion below encodes behaviour the system exhibits
//! **today** (commit `20bc651`). Per the AT-2 contract: if any test
//! here fails after AT-1, the refactor changed observable behaviour —
//! **the test must not be adapted**; the divergence is reported as a
//! gap.
//!
//! It only calls already-public helpers
//! (`gguf_to_hf_name_common` / `phi3_gguf_extra` / `gemma2_gguf_extra`
//! / `is_gguf_non_weight_tensor`, the per-name transform fns, the
//! adapter registry). It does not re-declare any production list
//! (e.g. it uses `is_gguf_non_weight_tensor`, never a local copy of
//! its name set).

use super::*;

use crate::nn::llama::gemma2::gemma2_transforms_for_name;
use crate::nn::llama::gguf_weight_loading::{
    gemma2_gguf_transforms_for_name, llama_gguf_transforms_for_name, phi3_gguf_transforms_for_name,
};
use crate::nn::llama::phi3::phi3_transforms_for_name;
use crate::nn::llama::weight_loading::compute_transforms_for_name;
use crate::v17::loader::gguf_reader::GgufTensorType;
use crate::v17::loader::gguf_to_hf_naming::{
    gguf_to_hf_name_common, is_gguf_non_weight_tensor, phi3_gguf_extra,
};
use crate::v17::loader::weight_mapper::LoadTransform;

// ============================================================
// 1) Name-mapping
// ============================================================

/// Every registered adapter resolves the architecture-agnostic
/// common names (top-level + Llama-layout block suffixes)
/// identically. Freezes the common contract across the whole
/// registry.
#[test]
fn every_adapter_resolves_common_names() {
    let common: &[(&str, &str)] = &[
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
        (
            "blk.5.attn_norm.weight",
            "model.layers.5.input_layernorm.weight",
        ),
        (
            "blk.5.attn_q.weight",
            "model.layers.5.self_attn.q_proj.weight",
        ),
        (
            "blk.5.attn_k.weight",
            "model.layers.5.self_attn.k_proj.weight",
        ),
        (
            "blk.5.attn_v.weight",
            "model.layers.5.self_attn.v_proj.weight",
        ),
        (
            "blk.5.attn_output.weight",
            "model.layers.5.self_attn.o_proj.weight",
        ),
        (
            "blk.5.ffn_norm.weight",
            "model.layers.5.post_attention_layernorm.weight",
        ),
        ("blk.5.ffn_gate.weight", "model.layers.5.mlp.gate_proj.weight"),
        ("blk.5.ffn_down.weight", "model.layers.5.mlp.down_proj.weight"),
    ];
    for adapter in ADAPTERS.iter() {
        for (gguf, hf) in common {
            // **G-1b2 / GAP-N2 (intentional behaviour change)** —
            // `ffn_norm.weight` is the one "common" name that
            // diverges by family. Gemma 2 has a 4-norm layout
            // where llama.cpp `ffn_norm` is the *pre-FFN* norm
            // (`pre_feedforward_layernorm`), NOT the 2-norm-Llama
            // `post_attention_layernorm`. This was validated
            // against a real Gemma 2 GGUF checkpoint (the load
            // hard-errored on a missing `pre_feedforward_layernorm`
            // before the fix). The conformance set is updated here
            // to assert the family-correct value for gemma2 rather
            // than freeze the wrong shared one.
            if *gguf == "blk.5.ffn_norm.weight" && adapter.id() == "gemma2" {
                assert_eq!(
                    adapter.gguf_to_hf_name(gguf).as_deref(),
                    Some("model.layers.5.pre_feedforward_layernorm.weight"),
                    "gemma2 maps ffn_norm -> pre_feedforward_layernorm (4-norm layout)"
                );
                continue;
            }
            assert_eq!(
                adapter.gguf_to_hf_name(gguf).as_deref(),
                Some(*hf),
                "adapter '{}' must map common '{gguf}' -> '{hf}'",
                adapter.id()
            );
        }
    }
}

/// Unknown / malformed GGUF names resolve to `None` on every
/// adapter (no adapter invents a mapping for garbage).
#[test]
fn every_adapter_rejects_unknown_names() {
    for adapter in ADAPTERS.iter() {
        for bad in [
            "general.foo",
            "blk.x.attn_q.weight",
            "blk.0.totally_unknown.weight",
            "not_a_tensor",
        ] {
            assert_eq!(
                adapter.gguf_to_hf_name(bad),
                None,
                "adapter '{}' must NOT map unknown '{bad}'",
                adapter.id()
            );
        }
    }
}

/// **Family override wins over common.** The Phi-3 fused mappings
/// must shadow the common Llama-layout table; the Llama-family
/// adapters must keep the common (separate) mapping. This is the
/// exact invariant the Phi-3 `345482d` / #5a composition-order bug
/// violated.
#[test]
fn phi3_family_override_wins_over_common() {
    // Phi-3: fused QKV + fused gate_up.
    assert_eq!(
        PHI3_ADAPTER
            .gguf_to_hf_name("blk.3.attn_qkv.weight")
            .as_deref(),
        Some("model.layers.3.self_attn.qkv_proj.weight")
    );
    assert_eq!(
        PHI3_ADAPTER
            .gguf_to_hf_name("blk.3.ffn_up.weight")
            .as_deref(),
        Some("model.layers.3.mlp.gate_up_proj.weight"),
        "Phi-3 fused gate_up override must win over common ffn_up->up_proj"
    );
    // Non-overridden Phi-3 names still fall through to common.
    assert_eq!(
        PHI3_ADAPTER
            .gguf_to_hf_name("blk.3.ffn_down.weight")
            .as_deref(),
        Some("model.layers.3.mlp.down_proj.weight")
    );

    // Llama family keeps the separate (common) mapping and has no
    // fused QKV at all.
    let llama_family: [&dyn AteniaModelAdapter; 3] =
        [&LLAMA_FAMILY_ADAPTER, &QWEN2_ADAPTER, &MISTRAL_ADAPTER];
    for a in llama_family {
        assert_eq!(
            a.gguf_to_hf_name("blk.3.ffn_up.weight").as_deref(),
            Some("model.layers.3.mlp.up_proj.weight"),
            "llama-family adapter must keep common ffn_up->up_proj"
        );
        assert_eq!(
            a.gguf_to_hf_name("blk.3.attn_qkv.weight"),
            None,
            "llama-family adapter must NOT map fused QKV"
        );
    }
}

/// Gemma 2 adds the post-attention / post-FFN norms on top of the
/// common set; the Llama default must not resolve those.
#[test]
fn gemma2_family_override_adds_post_norms() {
    assert_eq!(
        GEMMA2_ADAPTER
            .gguf_to_hf_name("blk.3.attn_post_norm.weight")
            .as_deref(),
        Some("model.layers.3.post_attention_layernorm.weight")
    );
    assert_eq!(
        GEMMA2_ADAPTER
            .gguf_to_hf_name("blk.3.ffn_post_norm.weight")
            .as_deref(),
        Some("model.layers.3.post_feedforward_layernorm.weight")
    );
    assert_eq!(
        LLAMA_FAMILY_ADAPTER.gguf_to_hf_name("blk.3.ffn_post_norm.weight"),
        None,
        "post-FFN norm is gemma2-only; llama default must not resolve it"
    );
}

/// **G-1b2 / GAP-N2** — the four real llama.cpp Gemma 2 per-layer
/// norm tensors (verified on bartowski/gemma-2-2b-it Q4_K_M) must
/// each resolve to the matching HF norm through `Gemma2Adapter`
/// (extra-first composition). The decisive case is `ffn_norm`,
/// which the common Llama-layout table maps to
/// `post_attention_layernorm` (correct only for the 2-norm Llama
/// layout); Gemma 2's 4-norm layout requires
/// `pre_feedforward_layernorm`, and the family override must win.
/// Before this fix the Gemma 2 GGUF load hard-errored with 26
/// missing `pre_feedforward_layernorm.weight`.
#[test]
fn gemma2_four_per_layer_norms_resolve() {
    let cases: &[(&str, &str)] = &[
        ("blk.4.attn_norm.weight", "model.layers.4.input_layernorm.weight"),
        (
            "blk.4.ffn_norm.weight",
            "model.layers.4.pre_feedforward_layernorm.weight",
        ),
        (
            "blk.4.post_attention_norm.weight",
            "model.layers.4.post_attention_layernorm.weight",
        ),
        (
            "blk.4.post_ffw_norm.weight",
            "model.layers.4.post_feedforward_layernorm.weight",
        ),
    ];
    for (gguf, hf) in cases {
        assert_eq!(
            GEMMA2_ADAPTER.gguf_to_hf_name(gguf).as_deref(),
            Some(*hf),
            "Gemma 2 norm '{gguf}' must map to '{hf}'"
        );
    }
    // The same four HF norm names are exactly what build_gemma2
    // registers per layer — no Gemma 2 GGUF norm tensor is left
    // unmapped (the GAP-N2 completeness failure).
    let resolved: std::collections::HashSet<_> = cases
        .iter()
        .map(|(g, _)| GEMMA2_ADAPTER.gguf_to_hf_name(g).unwrap())
        .collect();
    assert_eq!(resolved.len(), 4, "the four Gemma 2 norms must map 1:1");

    // Regression: the llama-family default must NOT divert
    // `ffn_norm` (it stays the 2-norm post_attention mapping).
    assert_eq!(
        LLAMA_FAMILY_ADAPTER
            .gguf_to_hf_name("blk.4.ffn_norm.weight")
            .as_deref(),
        Some("model.layers.4.post_attention_layernorm.weight"),
        "llama ffn_norm stays post_attention_layernorm (2-norm layout)"
    );
}

// ============================================================
// 2) HF vs GGUF transform parity (per tensor class)
// ============================================================

/// Phi-3 GGUF transforms are *defined* by delegation to the
/// safetensors table (`b423f56`). Freeze that they are
/// byte-identical for every Phi-3 tensor class — this is the
/// single-source-of-truth invariant the #4 HF/GGUF divergence
/// violated.
#[test]
fn phi3_hf_and_gguf_transforms_are_identical_per_class() {
    let h = 3072;
    let names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ];
    for n in names {
        assert_eq!(
            phi3_gguf_transforms_for_name(n, h),
            phi3_transforms_for_name(n, h),
            "Phi-3 HF vs GGUF transform divergence for '{n}'"
        );
    }
    // Per-class snapshot (catches a regression in the shared table).
    assert_eq!(
        phi3_transforms_for_name("model.embed_tokens.weight", h),
        Vec::<LoadTransform>::new(),
        "Phi-3 embed must NOT be transposed"
    );
    for n in [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ] {
        assert_eq!(
            phi3_transforms_for_name(n, h),
            vec![LoadTransform::Transpose2D],
            "Phi-3 Linear '{n}' must be a single Transpose2D"
        );
    }
    assert_eq!(
        phi3_transforms_for_name("model.layers.0.input_layernorm.weight", h),
        vec![LoadTransform::Reshape {
            target: vec![1, 1, h],
        }]
    );
}

/// Llama GGUF transforms == a `LlamaRopeUnpermuteRows` prefix on
/// the q/k projections, then the HF table verbatim; for every
/// other class the GGUF list equals the HF list exactly. Freezes
/// the precise current relationship (not naive equality — q/k
/// genuinely differ by the rope-unpermute prefix).
#[test]
fn llama_gguf_is_rope_unpermute_prefix_over_hf_table() {
    let (h, head_dim, kv_groups, scale) = (2048usize, 64usize, 8usize, 0.125f32);
    let q_k = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    ];
    let others = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
    ];
    for n in q_k {
        let hf = compute_transforms_for_name(n, h, head_dim, kv_groups, scale);
        let mut expected = vec![LoadTransform::LlamaRopeUnpermuteRows { head_dim }];
        expected.extend(hf);
        assert_eq!(
            llama_gguf_transforms_for_name(n, h, head_dim, kv_groups, scale),
            expected,
            "llama GGUF q/k '{n}' must be rope-unpermute prefix + HF table"
        );
    }
    for n in others {
        assert_eq!(
            llama_gguf_transforms_for_name(n, h, head_dim, kv_groups, scale),
            compute_transforms_for_name(n, h, head_dim, kv_groups, scale),
            "llama GGUF non-q/k '{n}' must equal the HF table exactly"
        );
    }
}

/// **G-1c / GAP-T1 (updated, intentional behaviour change).**
/// The Gemma 2 GGUF table is now the **corrected** one: identical
/// to the HF table **except the RMSNorm rule drops the `+1`
/// fold**, because llama.cpp pre-folds `1+γ` into the Gemma 2
/// GGUF norm weights (measured exactly +1.0 element-wise vs the
/// HF safetensors of the same model). This replaces the old
/// pre-`.rev()` buggy snapshot (embed transposed, dim:1 tiles,
/// identity Linears) that the AT-2 version froze verbatim. The
/// change was validated end-to-end against the real
/// bartowski/gemma-2-2b-it Q4_K_M checkpoint — not adapted to
/// hide a regression.
#[test]
fn gemma2_hf_and_gguf_tables_are_frozen_snapshots() {
    let (h, head_dim, kv_groups, scale) = (2304usize, 256usize, 2usize, 1.0f32 / 16.0f32);

    // --- HF table (crate::nn::llama::gemma2::gemma2_transforms_for_name) ---
    assert_eq!(
        gemma2_transforms_for_name("model.embed_tokens.weight", h, head_dim, kv_groups, scale),
        Vec::<LoadTransform>::new(),
        "gemma2 HF embed = identity"
    );
    assert_eq!(
        gemma2_transforms_for_name(
            "model.layers.0.post_feedforward_layernorm.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        vec![
            LoadTransform::Reshape {
                target: vec![1, 1, h],
            },
            LoadTransform::AddScalar { scalar: 1.0 },
        ],
        "gemma2 HF norm = Reshape + AddScalar(1.0)"
    );
    assert_eq!(
        gemma2_transforms_for_name(
            "model.layers.0.self_attn.k_proj.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        vec![
            LoadTransform::TileGroupedDim {
                dim: 0,
                group_size: head_dim,
                repeats: kv_groups,
            },
            LoadTransform::Transpose2D,
            LoadTransform::Scale { factor: scale },
        ],
        "gemma2 HF k_proj snapshot"
    );
    assert_eq!(
        gemma2_transforms_for_name(
            "model.layers.0.self_attn.o_proj.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        vec![LoadTransform::Transpose2D],
        "gemma2 HF Linear (o_proj) = Transpose2D"
    );

    // --- GGUF table (gguf_weight_loading::gemma2_gguf_transforms_for_name) ---
    // **Corrected (GAP-T1)**: identical to the HF table EXCEPT the
    // RMSNorm rule has NO `+1` (llama.cpp pre-folds it into the
    // GGUF weights). embed / k / Linear now match HF exactly.
    assert_eq!(
        gemma2_gguf_transforms_for_name(
            "model.embed_tokens.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        Vec::<LoadTransform>::new(),
        "gemma2 GGUF embed = identity (post-.rev() == HF; the old +Transpose2D was the bug)"
    );
    assert_eq!(
        gemma2_gguf_transforms_for_name(
            "model.layers.0.post_feedforward_layernorm.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        vec![LoadTransform::Reshape {
            target: vec![1, 1, h],
        }],
        "gemma2 GGUF norm = Reshape ONLY (no +1; llama.cpp pre-folded 1+gamma)"
    );
    assert_eq!(
        gemma2_gguf_transforms_for_name(
            "model.layers.0.self_attn.k_proj.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        vec![
            LoadTransform::TileGroupedDim {
                dim: 0,
                group_size: head_dim,
                repeats: kv_groups,
            },
            LoadTransform::Transpose2D,
            LoadTransform::Scale { factor: scale },
        ],
        "gemma2 GGUF k_proj = HF (dim:0 tile + Transpose2D + Scale)"
    );
    assert_eq!(
        gemma2_gguf_transforms_for_name(
            "model.layers.0.self_attn.o_proj.weight",
            h,
            head_dim,
            kv_groups,
            scale
        ),
        vec![LoadTransform::Transpose2D],
        "gemma2 GGUF Linear (o_proj) = Transpose2D (== HF)"
    );
}

// ============================================================
// 3) Non-weight (config-input) tensors
// ============================================================

/// `is_gguf_non_weight_tensor` recognises **exactly** the two
/// Phi-3 LongRope factor tensors and nothing else. Uses the
/// production helper directly (no local name list).
#[test]
fn non_weight_tensor_set_is_exactly_the_longrope_factors() {
    assert!(is_gguf_non_weight_tensor("rope_factors_short.weight"));
    assert!(is_gguf_non_weight_tensor("rope_factors_long.weight"));

    // Real graph weights / unknowns / the other rope tensor
    // (`rope_freqs`, handled by the separate gate predicate, NOT
    // by this helper) must all be false.
    for not_skipped in [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_q.weight",
        "blk.3.attn_qkv.weight",
        "blk.0.ffn_up.weight",
        "rope_freqs.weight",
        "rope_factors_short",     // missing `.weight` suffix
        "blk.0.rope_factors_long.weight", // block-scoped, not top-level
        "totally.unknown",
    ] {
        assert!(
            !is_gguf_non_weight_tensor(not_skipped),
            "'{not_skipped}' must NOT be treated as a skippable non-weight tensor"
        );
    }
}

// ============================================================
// 4) Fused tensors (critical Phi-3 cases)
// ============================================================

/// The fused-tensor extras, exercised directly on the free fn AND
/// through the resolved Phi-3 adapter. Llama-family adapters must
/// not produce these mappings.
#[test]
fn fused_qkv_and_gate_up_are_phi3_only() {
    // Free fn: phi3_gguf_extra owns both fused tensors.
    assert_eq!(
        phi3_gguf_extra("blk.7.attn_qkv.weight").as_deref(),
        Some("model.layers.7.self_attn.qkv_proj.weight")
    );
    assert_eq!(
        phi3_gguf_extra("blk.7.ffn_up.weight").as_deref(),
        Some("model.layers.7.mlp.gate_up_proj.weight")
    );
    // phi3_gguf_extra is fused-only: it does NOT handle the common
    // names (those compose via the adapter).
    assert_eq!(phi3_gguf_extra("blk.7.attn_q.weight"), None);
    assert_eq!(phi3_gguf_extra("token_embd.weight"), None);

    // Common table is untouched: ffn_up still maps to the separate
    // up_proj there (the llama path).
    assert_eq!(
        gguf_to_hf_name_common("blk.7.ffn_up.weight").as_deref(),
        Some("model.layers.7.mlp.up_proj.weight")
    );
    assert_eq!(gguf_to_hf_name_common("blk.7.attn_qkv.weight"), None);

    // Through the registry: Phi-3 fuses, the rest do not.
    assert_eq!(
        PHI3_ADAPTER
            .gguf_to_hf_name("blk.7.attn_qkv.weight")
            .as_deref(),
        Some("model.layers.7.self_attn.qkv_proj.weight")
    );
    let non_phi3: [&dyn AteniaModelAdapter; 4] = [
        &LLAMA_FAMILY_ADAPTER,
        &QWEN2_ADAPTER,
        &MISTRAL_ADAPTER,
        &GEMMA2_ADAPTER,
    ];
    for a in non_phi3 {
        assert_eq!(
            a.gguf_to_hf_name("blk.7.attn_qkv.weight"),
            None,
            "only Phi-3 maps fused QKV"
        );
    }
}

// ============================================================
// 5) GGUF dtype coverage
// ============================================================

/// Freezes the `GgufTensorType` raw-id <-> variant table — the
/// routing key `decode_tensor`'s `match` switches on. The
/// supported set {F32,F16,Q8_0,Q4_K,Q5_K,Q6_K} and the
/// known-but-unsupported set (which hit the `UnsupportedDType`
/// arm) must all round-trip stably; an unknown id must fall to
/// `Unknown(v)`.
///
/// NOTE / reported gap: this freezes the *selector*, not the
/// `decode_tensor` Ok/Err routing itself. The Ok / UnsupportedDType
/// behaviour of `decode_tensor` for each variant remains covered
/// only by the existing fixture tests in `gguf_decode.rs` — there
/// is no public reusable GGUF fixture builder, so a consolidated
/// decode-routing assertion is intentionally NOT added here (a
/// hand-rolled GGUF writer would risk format-fragility false
/// failures, violating the "do not adapt a failing test" rule).
#[test]
fn gguf_dtype_id_table_is_frozen() {
    // Supported by decode_tensor today.
    let supported = [
        (0u32, GgufTensorType::F32),
        (1, GgufTensorType::F16),
        (8, GgufTensorType::Q8_0),
        (12, GgufTensorType::Q4_K),
        (13, GgufTensorType::Q5_K),
        (14, GgufTensorType::Q6_K),
    ];
    // Known to GGUF, NOT decoded today -> decode_tensor's
    // `other => UnsupportedDType` arm.
    let unsupported = [
        (2u32, GgufTensorType::Q4_0),
        (3, GgufTensorType::Q4_1),
        (6, GgufTensorType::Q5_0),
        (7, GgufTensorType::Q5_1),
        (9, GgufTensorType::Q8_1),
        (10, GgufTensorType::Q2_K),
        (11, GgufTensorType::Q3_K),
        (15, GgufTensorType::Q8_K),
        (30, GgufTensorType::BF16),
    ];
    for (raw, variant) in supported.into_iter().chain(unsupported) {
        assert_eq!(
            GgufTensorType::from_u32(raw),
            variant,
            "from_u32({raw}) must map to {variant:?}"
        );
        assert_eq!(
            variant.as_u32(),
            raw,
            "{variant:?}.as_u32() must round-trip to {raw}"
        );
    }
    // Unknown id falls through to the catch-all (decode_tensor
    // treats this as UnsupportedDType too).
    assert_eq!(GgufTensorType::from_u32(9999), GgufTensorType::Unknown(9999));
    assert_eq!(GgufTensorType::Unknown(9999).as_u32(), 9999);
}

// ============================================================
// 6) Completeness gate (no false positives / no false negatives)
// ============================================================

/// Mirrors the GGUF completeness predicate verbatim from
/// `pipeline.rs` (the two identical sites:
/// `!s.contains("rope_freqs") && !is_gguf_non_weight_tensor(s)`).
/// This is *not* a new production helper — it is the test stating
/// the gate's semantics, calling the real
/// `is_gguf_non_weight_tensor`.
fn is_unexpected_skip(skipped_name: &str) -> bool {
    !skipped_name.contains("rope_freqs") && !is_gguf_non_weight_tensor(skipped_name)
}

#[test]
fn completeness_gate_has_no_false_positives() {
    // Legitimately-skippable tensors must NOT be flagged as an
    // unexpected skip (false positive == spurious load failure).
    for ok_to_skip in [
        "rope_freqs.weight",
        "blk.0.rope_freqs.weight",
        "rope_factors_short.weight",
        "rope_factors_long.weight",
    ] {
        assert!(
            !is_unexpected_skip(ok_to_skip),
            "'{ok_to_skip}' is a known config-input/aux tensor; \
             flagging it would be a false positive"
        );
    }
}

#[test]
fn completeness_gate_has_no_false_negatives() {
    // A genuinely missing graph weight MUST be flagged (a false
    // negative == silent wrong/partial load).
    for real_weight in [
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.3.attn_qkv.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "bogus.tensor.name",
    ] {
        assert!(
            is_unexpected_skip(real_weight),
            "'{real_weight}' is a real weight; skipping it MUST be flagged"
        );
    }
}
