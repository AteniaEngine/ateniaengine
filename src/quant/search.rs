//! **AQS-7** — deterministic policy search engine (experimental).
//!
//! AQS-1..6 produced the pieces; AQS-7 is the orchestrator that ties them
//! into the audit's headline workflow:
//!
//! ```text
//!   generate candidates  (default grid or explicit config)
//!     → build policies    (stable factory)
//!     → evaluate          (end-to-end results, supplied by the harness)
//!     → classify + rank    (AQS-6 certification report)
//!     → best_certified + best_useful_lossy + manifest draft
//! ```
//!
//! ## What this phase does — and does NOT
//!
//! * **Does**: define a deterministic candidate grid + config, a stable
//!   `AqsPolicyKind → Box<dyn QuantizationPolicy>` factory, and a search
//!   that consumes **already-computed** `EndToEndEvalResult`s and runs
//!   them through AQS-6. Optionally a *local* per-tensor search using the
//!   AQS-2 evaluator (clearly marked as a local signal, not certification).
//! * **Does NOT**: run real models from `search.rs` (the heavy TinyLlama
//!   forward stays in the AQS-4 harness — `search.rs` is testable without
//!   any model), implement new quantisation techniques, touch the CLI,
//!   runtime manifests, CUDA, tier-planner, generation, or loaders.
//!
//! Everything is experimental, CPU-only, opt-in, deterministic.
//!
//! See `docs/HANDOFF_AQS_7.md`.

use crate::quant::certification::AqsCertificationReport;
use crate::quant::end_to_end::EndToEndEvalResult;
use crate::quant::policy::{
    AwqPolicy, Bf16Fallback, GptqPolicy, HybridPolicy, PlainInt8, QuantizationPolicy,
};

// ============================================================================
// Candidate specification
// ============================================================================

/// A quantisation policy variant the search can instantiate. Mirrors the
/// existing policies one-to-one; adds no new technique.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AqsPolicyKind {
    Bf16,
    PlainInt8 { group_size: usize },
    Awq { group_size: usize, alpha: f32 },
    Hybrid { group_size: usize, alpha: f32, outlier_k: usize },
    Gptq { group_size: usize, block_size: usize, damp_percent: f32 },
}

impl AqsPolicyKind {
    /// Whether this kind needs a full `[S, K]` activation matrix
    /// (real GPTQ) rather than per-K absmax. Used by a future
    /// model-driving runner to decide if the candidate is supported with
    /// the calibration data on hand.
    pub fn needs_activation_matrix(&self) -> bool {
        matches!(self, AqsPolicyKind::Gptq { .. })
    }

    /// Whether this kind needs per-K activation absmax (AWQ / Hybrid).
    pub fn needs_activation_absmax(&self) -> bool {
        matches!(self, AqsPolicyKind::Awq { .. } | AqsPolicyKind::Hybrid { .. })
    }
}

/// A named candidate: a stable label plus the policy variant.
#[derive(Debug, Clone, PartialEq)]
pub struct AqsCandidateSpec {
    pub name: String,
    pub kind: AqsPolicyKind,
}

impl AqsCandidateSpec {
    pub fn new(name: impl Into<String>, kind: AqsPolicyKind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }

    /// Build the concrete policy. Stable ids are preserved because each
    /// arm delegates to the existing policy constructors.
    pub fn build_policy(&self) -> Box<dyn QuantizationPolicy> {
        policy_for_kind(self.kind)
    }
}

/// **Stable factory** — `AqsPolicyKind → Box<dyn QuantizationPolicy>`.
/// No new logic; delegates to existing constructors so policy `id()`s
/// stay unchanged.
pub fn policy_for_kind(kind: AqsPolicyKind) -> Box<dyn QuantizationPolicy> {
    match kind {
        AqsPolicyKind::Bf16 => Box::new(Bf16Fallback),
        AqsPolicyKind::PlainInt8 { group_size } => Box::new(PlainInt8::new(group_size)),
        AqsPolicyKind::Awq { group_size, alpha } => Box::new(AwqPolicy::new(group_size, alpha)),
        AqsPolicyKind::Hybrid {
            group_size,
            alpha,
            outlier_k,
        } => Box::new(HybridPolicy::new(group_size, alpha, outlier_k)),
        AqsPolicyKind::Gptq {
            group_size,
            block_size,
            damp_percent,
        } => Box::new(GptqPolicy::with_config(
            group_size,
            block_size,
            damp_percent,
            false,
        )),
    }
}

// ============================================================================
// Search configuration + default grid
// ============================================================================

/// Search configuration: the model under test, the ADR-004 gate, the
/// baseline memory used for compression ratios, and the ordered candidate
/// list.
#[derive(Debug, Clone, PartialEq)]
pub struct AqsSearchConfig {
    pub model_name: String,
    pub adr004_gate: f32,
    pub baseline_memory_bytes: Option<u64>,
    pub candidates: Vec<AqsCandidateSpec>,
}

impl AqsSearchConfig {
    /// Build a config using the [`default_candidate_grid`].
    pub fn with_default_grid(
        model_name: impl Into<String>,
        baseline_memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            adr004_gate: crate::quant::certification::ADR_004_GATE,
            baseline_memory_bytes,
            candidates: default_candidate_grid(),
        }
    }
}

/// **Deterministic, conservative default grid.** Small on purpose — this
/// is not the combinatorial explosion the AQS-0 audit warned against.
/// Order is fixed and stable.
///
/// ```text
///   bf16
///   plain_int8         g128
///   awq                g128 alpha=0.20
///   awq                g128 alpha=0.25
///   awq                g128 alpha=0.30
///   hybrid             g128 alpha=0.25 k64
///   gptq               g128 block128 damp=0.01
/// ```
pub fn default_candidate_grid() -> Vec<AqsCandidateSpec> {
    vec![
        AqsCandidateSpec::new("bf16", AqsPolicyKind::Bf16),
        AqsCandidateSpec::new(
            "plain_int8_g128",
            AqsPolicyKind::PlainInt8 { group_size: 128 },
        ),
        AqsCandidateSpec::new(
            "awq_g128_a0.20",
            AqsPolicyKind::Awq {
                group_size: 128,
                alpha: 0.20,
            },
        ),
        AqsCandidateSpec::new(
            "awq_g128_a0.25",
            AqsPolicyKind::Awq {
                group_size: 128,
                alpha: 0.25,
            },
        ),
        AqsCandidateSpec::new(
            "awq_g128_a0.30",
            AqsPolicyKind::Awq {
                group_size: 128,
                alpha: 0.30,
            },
        ),
        AqsCandidateSpec::new(
            "hybrid_g128_a0.25_k64",
            AqsPolicyKind::Hybrid {
                group_size: 128,
                alpha: 0.25,
                outlier_k: 64,
            },
        ),
        AqsCandidateSpec::new(
            "gptq_g128_b128_d0.01",
            AqsPolicyKind::Gptq {
                group_size: 128,
                block_size: 128,
                damp_percent: 0.01,
            },
        ),
    ]
}

// ============================================================================
// Search result
// ============================================================================

/// The result of a search: the config used, the AQS-6 certification
/// report (which carries the ranking + selections), and counts of
/// evaluated vs skipped candidates.
#[derive(Debug, Clone)]
pub struct AqsSearchResult {
    pub config: AqsSearchConfig,
    pub report: AqsCertificationReport,
    pub evaluated_count: usize,
    pub skipped_count: usize,
}

impl AqsSearchResult {
    /// Best ADR-004-certified policy (delegates to the report).
    pub fn best_certified(&self) -> Option<&str> {
        self.report.best_certified()
    }
    /// Best useful-lossy policy (delegates to the report).
    pub fn best_useful_lossy(&self) -> Option<&str> {
        self.report.best_useful_lossy()
    }
    /// Recommended policy — never substitutes lossy for certified.
    pub fn recommended_policy(&self) -> Option<&str> {
        self.report.recommended_policy()
    }
    /// Experimental manifest draft (delegates to the report).
    pub fn manifest_draft(&self) -> String {
        self.report.render_manifest_draft()
    }
    /// Human-readable report table (delegates to the report).
    pub fn human_report(&self) -> String {
        self.report.render_human_report()
    }
}

// ============================================================================
// Search over already-computed end-to-end results
// ============================================================================

/// **Primary AQS-7 entry point.** Run the search over *already-computed*
/// [`EndToEndEvalResult`]s (typically produced by the AQS-4 harness on a
/// real model). This keeps `search.rs` free of any model I/O and fully
/// unit-testable, while reusing the harness output rather than duplicating
/// it.
///
/// The classification + ranking + selection are done entirely by the
/// AQS-6 [`AqsCertificationReport`]. `evaluated_count` is the number of
/// results consumed; `skipped_count` is reported as 0 here (nothing is
/// skipped when results are pre-computed — skipping only happens in a
/// model-driving runner that lacks calibration for some candidate).
pub fn search_from_end_to_end_results(
    model_name: &str,
    adr004_gate: f32,
    baseline_memory_bytes: Option<u64>,
    results: Vec<EndToEndEvalResult>,
) -> AqsSearchResult {
    // Reconstruct a config whose candidate list mirrors the results
    // (by name). This keeps the result self-describing without requiring
    // the caller to pass the original grid.
    let candidates: Vec<AqsCandidateSpec> = results
        .iter()
        .map(|r| {
            // We do not know the exact kind from a bare result; record the
            // name with a Bf16 placeholder kind. The report (not this
            // list) is the source of truth for metrics/status. This field
            // exists only so the result is self-describing.
            AqsCandidateSpec::new(r.candidate_name.clone(), AqsPolicyKind::Bf16)
        })
        .collect();
    let evaluated_count = results.len();
    let report = AqsCertificationReport::build(model_name, &results, baseline_memory_bytes);
    let config = AqsSearchConfig {
        model_name: model_name.to_string(),
        adr004_gate,
        baseline_memory_bytes,
        candidates,
    };
    AqsSearchResult {
        config,
        report,
        evaluated_count,
        skipped_count: 0,
    }
}

// ============================================================================
// Optional: local per-tensor search (LOCAL SIGNAL ONLY — not certification)
// ============================================================================

/// One ranked entry from a local per-tensor search.
#[derive(Debug, Clone, PartialEq)]
pub struct AqsLocalTensorRanking {
    pub candidate_name: String,
    /// AQS-2 *local* weight-buffer drift — NOT end-to-end logit drift.
    pub local_max_abs_diff: f32,
    pub memory_bytes: u64,
}

/// **LOCAL SIGNAL ONLY.** Generate candidates, evaluate each against one
/// F32 tensor with the AQS-2 evaluator, and rank by ascending local
/// drift. This is a cheap pre-filter, **not** certification — AQS-3/AQS-4
/// proved local drift can mis-rank policies end-to-end (GPTQ). Candidates
/// that error (e.g. AWQ/GPTQ without calibration in `cal`) are skipped,
/// never panicked.
///
/// Returns `(rankings_sorted_by_local_drift, skipped_count)`.
pub fn search_tensor_local(
    candidates: &[AqsCandidateSpec],
    values: &[f32],
    shape: &[usize],
    cal: &crate::quant::policy::CalibrationContext<'_>,
) -> (Vec<AqsLocalTensorRanking>, usize) {
    use crate::quant::evaluator::{evaluate_tensor_policy, TensorEvalInput};

    let mut rankings: Vec<AqsLocalTensorRanking> = Vec::new();
    let mut skipped = 0usize;
    for spec in candidates {
        let policy = spec.build_policy();
        let input = TensorEvalInput {
            name: &spec.name,
            values,
            shape,
        };
        match evaluate_tensor_policy(input, policy.as_ref(), cal) {
            Ok(r) => rankings.push(AqsLocalTensorRanking {
                candidate_name: spec.name.clone(),
                local_max_abs_diff: r.max_abs_diff,
                memory_bytes: r.memory_bytes,
            }),
            Err(_) => skipped += 1,
        }
    }
    // Deterministic: ascending local drift, then memory, then name.
    rankings.sort_by(|a, b| {
        a.local_max_abs_diff
            .partial_cmp(&b.local_max_abs_diff)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.memory_bytes.cmp(&b.memory_bytes))
            .then(a.candidate_name.cmp(&b.candidate_name))
    });
    (rankings, skipped)
}

// ============================================================================
// Tests (synthetic — no real model)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::policy::CalibrationContext;

    fn res(name: &str, max: f32, argmax: bool, mem: u64) -> EndToEndEvalResult {
        EndToEndEvalResult {
            candidate_name: name.to_string(),
            max_abs_diff: max,
            mean_abs_diff: max * 0.1,
            rmse: max * 0.15,
            argmax_match: argmax,
            memory_bytes: mem,
        }
    }

    /// Real consolidated TinyLlama results (AQS-5).
    fn tinyllama_results() -> Vec<EndToEndEvalResult> {
        vec![
            res("bf16", 0.000063, true, 2_260_729_856),
            res("plain_int8", 1.260771, false, 1_165_688_832),
            res("awq", 0.889217, true, 1_165_688_832),
            res("hybrid", 0.831786, false, 1_266_614_272),
            res("gptq", 1.405399, false, 1_167_265_792),
        ]
    }
    const BASE: Option<u64> = Some(2_260_729_856);

    #[test]
    fn default_grid_is_deterministic() {
        let a = default_candidate_grid();
        let b = default_candidate_grid();
        assert_eq!(a, b);
        assert_eq!(a.len(), 7);
        assert_eq!(a[0].name, "bf16");
        assert_eq!(a.last().unwrap().name, "gptq_g128_b128_d0.01");
    }

    #[test]
    fn policy_factory_builds_supported_policies() {
        for spec in default_candidate_grid() {
            let p = spec.build_policy();
            // id() is stable and non-empty for every kind.
            assert!(!p.id().is_empty(), "{} produced empty id", spec.name);
        }
        // Spot-check ids map to the existing policies.
        assert_eq!(policy_for_kind(AqsPolicyKind::Bf16).id(), "bf16");
        assert_eq!(
            policy_for_kind(AqsPolicyKind::PlainInt8 { group_size: 128 }).id(),
            "plain_int8"
        );
        assert_eq!(
            policy_for_kind(AqsPolicyKind::Awq { group_size: 128, alpha: 0.25 }).id(),
            "awq"
        );
        assert_eq!(
            policy_for_kind(AqsPolicyKind::Gptq {
                group_size: 128,
                block_size: 128,
                damp_percent: 0.01
            })
            .id(),
            "gptq"
        );
    }

    #[test]
    fn search_from_results_selects_best_certified() {
        let r = search_from_end_to_end_results("tinyllama", 0.5, BASE, tinyllama_results());
        assert_eq!(r.best_certified(), Some("bf16"));
        assert_eq!(r.recommended_policy(), Some("bf16"));
    }

    #[test]
    fn search_from_results_selects_best_useful_lossy() {
        let r = search_from_end_to_end_results("tinyllama", 0.5, BASE, tinyllama_results());
        assert_eq!(r.best_useful_lossy(), Some("awq"));
    }

    #[test]
    fn search_from_results_marks_awq_useful_lossy() {
        use crate::quant::certification::CertificationStatus;
        let r = search_from_end_to_end_results("tinyllama", 0.5, BASE, tinyllama_results());
        assert_eq!(
            r.report.policy("awq").unwrap().status,
            CertificationStatus::UsefulLossy
        );
    }

    #[test]
    fn search_from_results_keeps_failed_candidates() {
        let r = search_from_end_to_end_results("tinyllama", 0.5, BASE, tinyllama_results());
        // All 5 candidates survive in the report (none dropped).
        assert_eq!(r.report.policies.len(), 5);
        assert_eq!(r.evaluated_count, 5);
    }

    #[test]
    fn search_result_manifest_draft_is_stable() {
        let r = search_from_end_to_end_results("tinyllama-1.1b", 0.5, BASE, tinyllama_results());
        let a = r.manifest_draft();
        let b = r.manifest_draft();
        assert_eq!(a, b);
        assert!(a.contains("schema_version: \"3.0.0-draft\""));
        assert!(a.contains("certified_policy: bf16"));
        assert!(a.contains("policy: awq"));
    }

    #[test]
    fn search_does_not_promote_lossy_over_certified() {
        let r = search_from_end_to_end_results("tinyllama", 0.5, BASE, tinyllama_results());
        assert_eq!(r.recommended_policy(), Some("bf16"));
        assert_ne!(r.recommended_policy(), r.best_useful_lossy());
    }

    #[test]
    fn search_handles_empty_results() {
        let r = search_from_end_to_end_results("empty", 0.5, BASE, vec![]);
        assert_eq!(r.evaluated_count, 0);
        assert_eq!(r.best_certified(), None);
        assert_eq!(r.best_useful_lossy(), None);
        assert!(r.report.policies.is_empty());
    }

    #[test]
    fn search_counts_evaluated_and_skipped() {
        let r = search_from_end_to_end_results("tinyllama", 0.5, BASE, tinyllama_results());
        assert_eq!(r.evaluated_count, 5);
        assert_eq!(r.skipped_count, 0);
    }

    #[test]
    fn local_search_ranks_and_skips_uncalibrated() {
        // 16x16 tensor, NO calibration → AWQ/Hybrid/GPTQ skip, bf16 +
        // plain_int8 evaluate. Deterministic ranking by local drift.
        let values: Vec<f32> = (0..256).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
        let shape = [16usize, 16usize];
        let grid = default_candidate_grid();
        let (rankings, skipped) =
            search_tensor_local(&grid, &values, &shape, &CalibrationContext::empty());
        // bf16 (0 drift) + plain_int8 evaluate; 3 awq + 1 hybrid + 1 gptq skip = 5.
        assert_eq!(rankings.len(), 2);
        assert_eq!(skipped, 5);
        // bf16 is a no-op → lowest local drift → ranked first.
        assert_eq!(rankings[0].candidate_name, "bf16");
        assert_eq!(rankings[0].local_max_abs_diff, 0.0);
    }

    #[test]
    fn local_search_is_deterministic() {
        let values: Vec<f32> = (0..256).map(|i| ((i % 7) as f32) * 0.2 - 0.6).collect();
        let shape = [16usize, 16usize];
        let grid = default_candidate_grid();
        let a = search_tensor_local(&grid, &values, &shape, &CalibrationContext::empty());
        let b = search_tensor_local(&grid, &values, &shape, &CalibrationContext::empty());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }
}
