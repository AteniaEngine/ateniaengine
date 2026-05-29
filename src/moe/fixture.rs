//! **MOE-1** — fixture spec + certification-strategy model (experimental).
//!
//! Pure data + pure functions. No model loading, no forward, no F64
//! generation, no runtime touch. This is the vocabulary a future MoE
//! certification effort will use to describe a fixture and decide how it
//! can be certified against ADR-004.

/// Errors from validating a [`FixtureMoESpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureSpecError {
    /// `model_name` is empty.
    EmptyModelName,
    /// `family` is empty.
    EmptyFamily,
    /// `parameter_count == 0`.
    ZeroParameters,
    /// `expert_count == 0` — a MoE fixture must have at least one expert.
    ZeroExperts,
    /// `experts_per_token == 0`.
    ZeroExpertsPerToken,
    /// `experts_per_token > expert_count` — cannot route to more experts
    /// than exist.
    ExpertsPerTokenExceedsExpertCount { per_token: usize, total: usize },
    /// `active_parameter_count > parameter_count` — the per-token active
    /// share cannot exceed the total.
    ActiveExceedsTotal { active: u64, total: u64 },
}

impl std::fmt::Display for FixtureSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixtureSpecError::EmptyModelName => write!(f, "fixture: model_name is empty"),
            FixtureSpecError::EmptyFamily => write!(f, "fixture: family is empty"),
            FixtureSpecError::ZeroParameters => write!(f, "fixture: parameter_count is 0"),
            FixtureSpecError::ZeroExperts => write!(f, "fixture: expert_count is 0"),
            FixtureSpecError::ZeroExpertsPerToken => {
                write!(f, "fixture: experts_per_token is 0")
            }
            FixtureSpecError::ExpertsPerTokenExceedsExpertCount { per_token, total } => write!(
                f,
                "fixture: experts_per_token ({per_token}) exceeds expert_count ({total})"
            ),
            FixtureSpecError::ActiveExceedsTotal { active, total } => write!(
                f,
                "fixture: active_parameter_count ({active}) exceeds parameter_count ({total})"
            ),
        }
    }
}

impl std::error::Error for FixtureSpecError {}

/// Description of a candidate MoE **certification fixture**. This is
/// metadata only — it never loads or runs the model.
///
/// `parameter_count` is the **total** parameters (all experts);
/// `active_parameter_count` is the per-token active share (router selects
/// `experts_per_token` of `expert_count`). For a synthetic fixture both
/// may be tiny; for a real model they describe the published sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureMoESpec {
    pub model_name: String,
    pub family: String,
    /// Total parameter count across all experts + shared layers.
    pub parameter_count: u64,
    /// Per-token active parameter count (top-k experts + shared layers).
    pub active_parameter_count: u64,
    pub expert_count: usize,
    pub experts_per_token: usize,
}

impl FixtureMoESpec {
    /// Construct and validate. Returns `Err` if the spec is internally
    /// inconsistent.
    pub fn new(
        model_name: impl Into<String>,
        family: impl Into<String>,
        parameter_count: u64,
        active_parameter_count: u64,
        expert_count: usize,
        experts_per_token: usize,
    ) -> Result<Self, FixtureSpecError> {
        let spec = Self {
            model_name: model_name.into(),
            family: family.into(),
            parameter_count,
            active_parameter_count,
            expert_count,
            experts_per_token,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Structural validation. Pure; no I/O.
    pub fn validate(&self) -> Result<(), FixtureSpecError> {
        if self.model_name.is_empty() {
            return Err(FixtureSpecError::EmptyModelName);
        }
        if self.family.is_empty() {
            return Err(FixtureSpecError::EmptyFamily);
        }
        if self.parameter_count == 0 {
            return Err(FixtureSpecError::ZeroParameters);
        }
        if self.expert_count == 0 {
            return Err(FixtureSpecError::ZeroExperts);
        }
        if self.experts_per_token == 0 {
            return Err(FixtureSpecError::ZeroExpertsPerToken);
        }
        if self.experts_per_token > self.expert_count {
            return Err(FixtureSpecError::ExpertsPerTokenExceedsExpertCount {
                per_token: self.experts_per_token,
                total: self.expert_count,
            });
        }
        if self.active_parameter_count > self.parameter_count {
            return Err(FixtureSpecError::ActiveExceedsTotal {
                active: self.active_parameter_count,
                total: self.parameter_count,
            });
        }
        Ok(())
    }

    /// Bytes required to hold this model's **full** weights in F64
    /// (the ADR-004 reference dtype): `parameter_count * 8`.
    pub fn f64_reference_weight_bytes(&self) -> u64 {
        f64_reference_weight_bytes(self.parameter_count)
    }
}

/// Free function form of the F64 weight-byte estimate (`params * 8`).
/// 8 bytes per parameter; verified against the TinyLlama fixture note
/// (1.1B params → ~8.8 GB F64 weights).
pub fn f64_reference_weight_bytes(parameter_count: u64) -> u64 {
    parameter_count.saturating_mul(8)
}

/// How a given MoE fixture can be certified against the ADR-004 F64
/// reference, given a host RAM budget.
///
/// This is the **decision substrate** — it is not wired into any runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoECertificationStrategy {
    /// The full model fits comfortably in F64 within the host budget, so a
    /// complete end-to-end F64 reference can be generated (the same method
    /// the four dense fixtures use). Preferred — strongest guarantee.
    FullReferenceF64,
    /// The full model does NOT fit in F64, but a partial reference is
    /// feasible (e.g. layer-wise / single-expert sub-references, or a
    /// reduced-config variant). Weaker than full; documented as such.
    PartialReferenceF64,
    /// No feasible F64 reference on the given budget — too large to
    /// certify under ADR-004 with the current methodology.
    Unsupported,
}

/// Headroom factor: an F64 reference run needs more than just the weights
/// (activations, F32→F64 intermediates). The TinyLlama note observes
/// ~8.8 GB weights fitting "comfortably within 16 GB", i.e. roughly a 2×
/// peak-over-weights envelope. We require weights to fit within
/// `host_ram_bytes / FULL_REF_HEADROOM_DIVISOR` for `FullReferenceF64`.
const FULL_REF_HEADROOM_DIVISOR: u64 = 2;

/// Below this F64 weight size, a fixture is "trivially F64-certifiable" on
/// essentially any host (e.g. a tiny synthetic MoE). 1 GiB.
const TRIVIAL_F64_BYTES: u64 = 1024 * 1024 * 1024;

/// Recommend a certification strategy for `spec` given a host RAM budget.
///
/// Pure function. Logic:
/// - if the F64 weights are trivially small (< 1 GiB) → `FullReferenceF64`;
/// - else if they fit within `host_ram_bytes / 2` → `FullReferenceF64`;
/// - else if they fit within the full host RAM (no headroom) → a partial
///   reference is the realistic path → `PartialReferenceF64`;
/// - else → `Unsupported`.
pub fn recommend_strategy(
    spec: &FixtureMoESpec,
    host_ram_bytes: u64,
) -> MoECertificationStrategy {
    let f64_bytes = spec.f64_reference_weight_bytes();
    if f64_bytes < TRIVIAL_F64_BYTES {
        return MoECertificationStrategy::FullReferenceF64;
    }
    if host_ram_bytes == 0 {
        return MoECertificationStrategy::Unsupported;
    }
    if f64_bytes <= host_ram_bytes / FULL_REF_HEADROOM_DIVISOR {
        MoECertificationStrategy::FullReferenceF64
    } else if f64_bytes <= host_ram_bytes {
        MoECertificationStrategy::PartialReferenceF64
    } else {
        MoECertificationStrategy::Unsupported
    }
}

// ============================================================================
// Reference candidate catalogue (metadata only — documentation in code).
//
// Published sizes for the candidates the MOE-0 audit evaluated. These are
// const descriptors so the candidate analysis cannot silently drift from
// the docs. NONE of these is loaded or executed.
// ============================================================================

/// Approximate published total / active parameter counts and expert
/// topology for the evaluated MoE candidates. Sizes are rounded to the
/// nearest 0.1B from the public model cards.
pub mod candidates {
    /// Mixtral 8x7B: 8 experts, top-2. ~46.7B total / ~12.9B active.
    pub const MIXTRAL_8X7B: (&str, &str, u64, u64, usize, usize) = (
        "mixtral-8x7b",
        "mixtral",
        46_700_000_000,
        12_900_000_000,
        8,
        2,
    );
    /// Qwen1.5-MoE-A2.7B: 60 experts (+4 shared), top-4. ~14.3B total / ~2.7B active.
    pub const QWEN15_MOE_A2_7B: (&str, &str, u64, u64, usize, usize) = (
        "qwen1.5-moe-a2.7b",
        "qwen-moe",
        14_300_000_000,
        2_700_000_000,
        60,
        4,
    );
    /// DeepSeek-V2-Lite: 64 routed (+2 shared) experts, top-6. ~15.7B total / ~2.4B active.
    pub const DEEPSEEK_V2_LITE: (&str, &str, u64, u64, usize, usize) = (
        "deepseek-v2-lite",
        "deepseek-moe",
        15_700_000_000,
        2_400_000_000,
        64,
        6,
    );
}

// ============================================================================
// Tests (light — no model load, no forward, no F64 generation)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tiny() -> FixtureMoESpec {
        // A tiny synthetic MoE: 4 experts, top-2, ~2M params total. Fits
        // F64 in ~16 MB — trivially certifiable on any host.
        FixtureMoESpec::new("atenia-moe-tiny-synthetic", "synthetic-moe", 2_000_000, 1_000_000, 4, 2)
            .unwrap()
    }

    #[test]
    fn fixture_spec_construction_validates() {
        let s = synthetic_tiny();
        assert_eq!(s.expert_count, 4);
        assert_eq!(s.experts_per_token, 2);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn fixture_spec_rejects_zero_experts() {
        let err = FixtureMoESpec::new("m", "f", 100, 50, 0, 1).unwrap_err();
        assert_eq!(err, FixtureSpecError::ZeroExperts);
    }

    #[test]
    fn fixture_spec_rejects_per_token_over_count() {
        let err = FixtureMoESpec::new("m", "f", 100, 50, 4, 8).unwrap_err();
        assert!(matches!(
            err,
            FixtureSpecError::ExpertsPerTokenExceedsExpertCount { per_token: 8, total: 4 }
        ));
    }

    #[test]
    fn fixture_spec_rejects_active_over_total() {
        let err = FixtureMoESpec::new("m", "f", 100, 200, 4, 2).unwrap_err();
        assert!(matches!(err, FixtureSpecError::ActiveExceedsTotal { .. }));
    }

    #[test]
    fn fixture_spec_rejects_empty_fields_and_zero_params() {
        assert_eq!(
            FixtureMoESpec::new("", "f", 100, 50, 4, 2).unwrap_err(),
            FixtureSpecError::EmptyModelName
        );
        assert_eq!(
            FixtureMoESpec::new("m", "", 100, 50, 4, 2).unwrap_err(),
            FixtureSpecError::EmptyFamily
        );
        assert_eq!(
            FixtureMoESpec::new("m", "f", 0, 0, 4, 2).unwrap_err(),
            FixtureSpecError::ZeroParameters
        );
    }

    #[test]
    fn f64_bytes_is_eight_per_param() {
        assert_eq!(f64_reference_weight_bytes(1), 8);
        // TinyLlama sanity: 1.1B params -> ~8.8 GB.
        let tinyllama = 1_100_000_000u64;
        let bytes = f64_reference_weight_bytes(tinyllama);
        assert_eq!(bytes, 8_800_000_000);
    }

    #[test]
    fn strategy_tiny_synthetic_is_full_reference() {
        // Tiny synthetic fits trivially regardless of host RAM.
        let s = synthetic_tiny();
        assert_eq!(
            recommend_strategy(&s, 8 * 1024 * 1024 * 1024),
            MoECertificationStrategy::FullReferenceF64
        );
    }

    #[test]
    fn strategy_mixtral_is_unsupported_on_dev_host() {
        // Mixtral 8x7B: ~46.7B params -> ~373 GB F64. Far beyond a 32 GB host.
        let (n, f, p, a, e, k) = candidates::MIXTRAL_8X7B;
        let s = FixtureMoESpec::new(n, f, p, a, e, k).unwrap();
        // ~373 GB needed; 32 GB host.
        assert_eq!(
            recommend_strategy(&s, 32 * 1024 * 1024 * 1024),
            MoECertificationStrategy::Unsupported
        );
        // Even a 128 GB workstation cannot full-reference it.
        assert_eq!(
            recommend_strategy(&s, 128u64 * 1024 * 1024 * 1024),
            MoECertificationStrategy::Unsupported
        );
    }

    #[test]
    fn strategy_qwen_moe_is_partial_on_large_host() {
        // Qwen1.5-MoE-A2.7B: ~14.3B -> ~114 GB F64. On a 128 GB host it
        // fits whole RAM but not with 2x headroom -> partial reference.
        let (n, f, p, a, e, k) = candidates::QWEN15_MOE_A2_7B;
        let s = FixtureMoESpec::new(n, f, p, a, e, k).unwrap();
        assert_eq!(
            recommend_strategy(&s, 128u64 * 1024 * 1024 * 1024),
            MoECertificationStrategy::PartialReferenceF64
        );
        // On a 32 GB dev host it is unsupported.
        assert_eq!(
            recommend_strategy(&s, 32 * 1024 * 1024 * 1024),
            MoECertificationStrategy::Unsupported
        );
    }

    #[test]
    fn candidate_descriptors_are_valid_specs() {
        for (n, f, p, a, e, k) in [
            candidates::MIXTRAL_8X7B,
            candidates::QWEN15_MOE_A2_7B,
            candidates::DEEPSEEK_V2_LITE,
        ] {
            let s = FixtureMoESpec::new(n, f, p, a, e, k);
            assert!(s.is_ok(), "candidate {n} must be a valid spec");
        }
    }

    #[test]
    fn zero_ram_host_is_unsupported_for_nontrivial() {
        let (n, f, p, a, e, k) = candidates::QWEN15_MOE_A2_7B;
        let s = FixtureMoESpec::new(n, f, p, a, e, k).unwrap();
        assert_eq!(recommend_strategy(&s, 0), MoECertificationStrategy::Unsupported);
    }
}
