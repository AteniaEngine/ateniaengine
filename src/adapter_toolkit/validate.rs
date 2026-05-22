//! **Adapter Toolkit v2 — Part 6: declarative validators.**
//!
//! Structural validation of an [`AdapterDsl`] *before* it is turned
//! into a live adapter. The checks here are declarative — they read
//! the DSL fields and report inconsistent combinations — and
//! fail-loud: every problem is collected into a [`ValidationReport`]
//! whose [`ValidationReport::into_result`] turns a non-empty error
//! list into a single [`ToolkitError::Validation`].
//!
//! Two severities:
//! - **errors** — the spec is inconsistent and must not be used
//!   (`gqa` without `kv_heads`, `fused_qkv` without `split_strategy`,
//!   `partial_rotary_factor` contradicting an explicit `rope`, an
//!   unknown family/architecture).
//! - **warnings** — the spec resolves, but a field is unusual for
//!   the resolved family (e.g. `fused_qkv` declared on a llama-family
//!   model whose v1 builder has no fused-QKV path).
//!
//! Validation never mutates and never constructs a builder; it is
//! safe to run on a spec that would later fail to resolve.

use crate::model_adapters::ModelFamily;

use super::dsl::{AdapterDsl, KvHeads};
use super::spec::ResolvedAdapterSpec;
use super::ToolkitError;

/// The outcome of validating an [`AdapterDsl`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    /// Blocking inconsistencies — the spec must not be used.
    pub errors: Vec<String>,
    /// Non-blocking oddities — the spec resolves but a field looks
    /// wrong for the resolved family.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// `true` when there are no blocking errors (warnings allowed).
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Collapse the report into a `Result`. Errors become a single
    /// [`ToolkitError::Validation`] joining every message; warnings
    /// never block.
    pub fn into_result(self) -> Result<(), ToolkitError> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(ToolkitError::Validation(self.errors.join("; ")))
        }
    }
}

/// Validate a DSL document. Runs the structural rules on the raw
/// DSL, then — if the DSL resolves — the family-aware cross-checks.
/// Resolution failures (unknown family / architecture) are surfaced
/// as errors here so a single `validate` call is the one fail-loud
/// gate the CLI needs.
pub fn validate(dsl: &AdapterDsl) -> ValidationReport {
    let mut report = ValidationReport::default();

    structural_rules(dsl, &mut report);

    // Resolution doubles as a validator: an unknown family or
    // architecture is a hard error. When it succeeds, run the
    // family-aware checks that need the resolved family.
    match ResolvedAdapterSpec::resolve(dsl) {
        Ok(spec) => family_rules(dsl, &spec, &mut report),
        Err(e) => report.errors.push(e.to_string()),
    }

    report
}

/// Rules that read only the raw DSL — no resolution needed.
fn structural_rules(dsl: &AdapterDsl, report: &mut ValidationReport) {
    // Rule: GQA requires an explicit kv_heads declaration.
    if let Some(att) = &dsl.attention {
        let is_gqa = att
            .kind
            .as_deref()
            .is_some_and(|k| k.eq_ignore_ascii_case("gqa"));
        if is_gqa && att.kv_heads.is_none() {
            report.errors.push(
                "attention.type = gqa requires attention.kv_heads (an integer or `auto`)"
                    .to_string(),
            );
        }
        // Rule: MQA with an explicit kv_heads count must be 1.
        let is_mqa = att
            .kind
            .as_deref()
            .is_some_and(|k| k.eq_ignore_ascii_case("mqa"));
        if is_mqa {
            if let Some(KvHeads::Count(n)) = att.kv_heads {
                if n != 1 {
                    report.errors.push(format!(
                        "attention.type = mqa is inconsistent with kv_heads = {n} (mqa implies 1)"
                    ));
                }
            }
        }
    }

    // Rule: fused_qkv requires a split_strategy.
    if let Some(w) = &dsl.weights {
        if w.fused_qkv == Some(true) && w.split_strategy.is_none() {
            report.errors.push(
                "weights.fused_qkv = true requires weights.split_strategy (e.g. `phi_qkv`)"
                    .to_string(),
            );
        }
        // Warn: a split_strategy with no fused_qkv to split is an
        // orphan field — harmless but almost certainly a mistake.
        if w.split_strategy.is_some() && w.fused_qkv != Some(true) {
            report.warnings.push(
                "weights.split_strategy is set but weights.fused_qkv is not true — \
                 the split strategy has nothing to split"
                    .to_string(),
            );
        }
    }

    // Rule: partial_rotary_factor must not contradict an explicit
    // rope variant. Absent rope is fine (it implies `partial`).
    if let Some(cfg) = &dsl.config {
        if cfg.partial_rotary_factor.is_some() {
            if let Some(rope) = &cfg.rope {
                let r = rope.to_ascii_lowercase();
                if r != "partial" && r != "partial_rotary" {
                    report.errors.push(format!(
                        "config.partial_rotary_factor is set but config.rope = `{rope}` \
                         (expected `partial`, or omit `rope` to let it imply partial)"
                    ));
                }
            }
        }
        // Rule: partial_rotary_factor, if present, must be in (0, 1].
        if let Some(f) = cfg.partial_rotary_factor {
            if !(f > 0.0 && f <= 1.0) {
                report.errors.push(format!(
                    "config.partial_rotary_factor = {f} is out of range (expected 0 < f <= 1)"
                ));
            }
        }
    }
}

/// Rules that need the resolved family.
fn family_rules(dsl: &AdapterDsl, spec: &ResolvedAdapterSpec, report: &mut ValidationReport) {
    let fused_capable = matches!(spec.family, ModelFamily::Phi3);

    // Warn: fused QKV/MLP declared on a family whose v1 builder has
    // no fused-weight path. The DSL does not change builders, so
    // this is a metadata inconsistency, not a hard failure.
    if spec.features.fused_qkv && !fused_capable {
        report.warnings.push(format!(
            "weights.fused_qkv is declared but family {:?} has no fused-QKV builder in v1 \
             (only Phi-3 does) — the flag will not change weight loading",
            spec.family
        ));
    }
    if spec.features.fused_mlp && !fused_capable {
        report.warnings.push(format!(
            "weights.fused_mlp is declared but family {:?} has no fused-MLP builder in v1 \
             (only Phi-3 does) — the flag will not change weight loading",
            spec.family
        ));
    }

    // Warn: LongRope on a non-Phi family. Only the Phi-3 adapter
    // parses a `longrope` rope_scaling block in v1.
    if matches!(spec.features.rope, super::spec::RopeKind::LongRope)
        && !matches!(spec.family, ModelFamily::Phi3)
    {
        report.warnings.push(format!(
            "config.rope = longrope is declared but family {:?} has no LongRope parser in v1 \
             (only Phi-3 does)",
            spec.family
        ));
    }

    // Warn: an explicit eos_tokens set that is empty is almost
    // certainly a mistake — generation would never stop on EOS.
    if let Some(tok) = &dsl.tokenizer {
        if tok.eos_tokens.as_ref().is_some_and(|e| e.is_empty()) {
            report
                .warnings
                .push("tokenizer.eos_tokens is an empty list — generation has no EOS stop".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dsl(text: &str) -> AdapterDsl {
        AdapterDsl::from_str(text, true).expect("dsl parses")
    }

    #[test]
    fn valid_simple_spec_passes_clean() {
        let r = validate(&dsl("family: llama\n"));
        assert!(r.is_ok());
        assert!(r.errors.is_empty() && r.warnings.is_empty());
        assert!(r.into_result().is_ok());
    }

    #[test]
    fn gqa_without_kv_heads_is_error() {
        let r = validate(&dsl("family: qwen\nattention:\n  type: gqa\n"));
        assert!(!r.is_ok());
        assert!(r.errors[0].contains("gqa requires attention.kv_heads"));
        assert!(r.into_result().is_err());
    }

    #[test]
    fn gqa_with_kv_heads_is_ok() {
        let r = validate(&dsl("family: qwen\nattention:\n  type: gqa\n  kv_heads: 8\n"));
        assert!(r.is_ok());
    }

    #[test]
    fn fused_qkv_without_split_strategy_is_error() {
        let r = validate(&dsl("family: phi\nweights:\n  fused_qkv: true\n"));
        assert!(!r.is_ok());
        assert!(r.errors[0].contains("split_strategy"));
    }

    #[test]
    fn fused_qkv_with_split_strategy_is_ok() {
        let r = validate(&dsl(
            "family: phi\nweights:\n  fused_qkv: true\n  split_strategy: phi_qkv\n",
        ));
        assert!(r.is_ok());
    }

    #[test]
    fn partial_rotary_factor_contradicting_rope_is_error() {
        let r = validate(&dsl(
            "family: phi\nconfig:\n  rope: standard\n  partial_rotary_factor: 0.4\n",
        ));
        assert!(!r.is_ok());
        assert!(r.errors[0].contains("partial_rotary_factor"));
    }

    #[test]
    fn partial_rotary_factor_alone_implies_partial_and_is_ok() {
        let r = validate(&dsl("family: phi\nconfig:\n  partial_rotary_factor: 0.4\n"));
        assert!(r.is_ok());
    }

    #[test]
    fn partial_rotary_factor_out_of_range_is_error() {
        let r = validate(&dsl("family: phi\nconfig:\n  partial_rotary_factor: 1.5\n"));
        assert!(!r.is_ok());
        assert!(r.errors[0].contains("out of range"));
    }

    #[test]
    fn mqa_with_inconsistent_kv_heads_is_error() {
        let r = validate(&dsl("family: llama\nattention:\n  type: mqa\n  kv_heads: 4\n"));
        assert!(!r.is_ok());
        assert!(r.errors[0].contains("mqa"));
    }

    #[test]
    fn unknown_family_is_validation_error() {
        let r = validate(&dsl("family: falcon\n"));
        assert!(!r.is_ok());
        assert!(r.errors.iter().any(|e| e.contains("falcon")));
    }

    #[test]
    fn fused_qkv_on_llama_family_is_warning_not_error() {
        let r = validate(&dsl(
            "family: llama\nweights:\n  fused_qkv: true\n  split_strategy: phi_qkv\n",
        ));
        assert!(r.is_ok(), "no blocking error");
        assert!(
            r.warnings.iter().any(|w| w.contains("fused-QKV builder")),
            "expected a family-mismatch warning"
        );
    }

    #[test]
    fn longrope_on_non_phi_is_warning() {
        let r = validate(&dsl("family: llama\nconfig:\n  rope: longrope\n"));
        assert!(r.is_ok());
        assert!(r.warnings.iter().any(|w| w.contains("LongRope")));
    }

    #[test]
    fn orphan_split_strategy_is_warning() {
        let r = validate(&dsl("family: phi\nweights:\n  split_strategy: phi_qkv\n"));
        assert!(r.is_ok(), "orphan split_strategy is not blocking");
        assert!(r.warnings.iter().any(|w| w.contains("nothing to split")));
    }

    #[test]
    fn empty_eos_token_list_is_warning() {
        let r = validate(&dsl("family: llama\ntokenizer:\n  eos_tokens: []\n"));
        assert!(r.is_ok());
        assert!(r.warnings.iter().any(|w| w.contains("EOS")));
    }
}
