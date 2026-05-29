//! **AQS-8** — callback-based model-driving runner (experimental).
//!
//! AQS-7 can rank *already-computed* end-to-end results. AQS-8 closes the
//! loop: given an [`AqsSearchConfig`], it iterates the candidate grid,
//! asks an **evaluator callback** to produce one [`EndToEndEvalResult`]
//! per candidate, accumulates them, and feeds them straight into
//! [`search_from_end_to_end_results`].
//!
//! ## Why callback-based (Option A)
//!
//! `src/quant/runner.rs` must NOT load or execute a real model — doing so
//! would couple `quant` to the loader / model runtime and duplicate the
//! AQS-4 harness. Instead the runner takes a closure
//! `FnMut(&AqsCandidateSpec) -> Result<EndToEndEvalResult, AqsRunnerError>`
//! plus a declaration of [`AqsEvaluatorCapabilities`]. This keeps the
//! orchestration:
//!
//! * **decoupled** — no model I/O in `quant`;
//! * **deterministic + unit-testable** — fake evaluators, no model needed;
//! * **reusable** — the heavy AQS-4 harness can pass a real evaluator that
//!   runs the forward, without `runner.rs` knowing anything about it.
//!
//! ## Scope (AQS-8)
//!
//! No CLI, no new techniques, no CUDA, no tier-planner, no generation, no
//! loaders, no productive manifest. Experimental, CPU-only, opt-in.
//!
//! See `docs/HANDOFF_AQS_8.md`.

use crate::quant::end_to_end::EndToEndEvalResult;
use crate::quant::search::{
    search_from_end_to_end_results, AqsCandidateSpec, AqsSearchConfig, AqsSearchResult,
};

// ============================================================================
// Evaluator capabilities
// ============================================================================

/// What kinds of calibration the supplied evaluator can provide. The
/// runner uses this to decide which candidates are *supported* before
/// calling the evaluator at all (so unsupported candidates are skipped
/// cleanly, never failed inside the closure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AqsEvaluatorCapabilities {
    /// The evaluator can supply per-K activation absmax (AWQ / Hybrid).
    pub activation_absmax: bool,
    /// The evaluator can supply a full `[S, K]` activation matrix
    /// (real GPTQ).
    pub activation_matrix: bool,
}

impl AqsEvaluatorCapabilities {
    /// An evaluator that supports every calibration kind.
    pub fn all() -> Self {
        Self {
            activation_absmax: true,
            activation_matrix: true,
        }
    }

    /// An evaluator with no calibration (only BF16 / plain INT8 work).
    pub fn none() -> Self {
        Self {
            activation_absmax: false,
            activation_matrix: false,
        }
    }

    /// Whether `candidate` is supported given these capabilities.
    pub fn supports(&self, candidate: &AqsCandidateSpec) -> bool {
        if candidate.kind.needs_activation_matrix() {
            return self.activation_matrix;
        }
        if candidate.kind.needs_activation_absmax() {
            return self.activation_absmax;
        }
        true
    }
}

// ============================================================================
// Error model
// ============================================================================

/// Errors surfaced by the runner / evaluator. No panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AqsRunnerError {
    /// The candidate needs calibration the evaluator does not provide.
    UnsupportedCandidate(String),
    /// The evaluator callback failed for this candidate.
    EvaluationFailed(String),
    /// The search config had no candidates.
    EmptyCandidateList,
}

impl std::fmt::Display for AqsRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AqsRunnerError::UnsupportedCandidate(n) => {
                write!(f, "aqs-runner: candidate `{n}` unsupported by evaluator")
            }
            AqsRunnerError::EvaluationFailed(m) => {
                write!(f, "aqs-runner: evaluation failed: {m}")
            }
            AqsRunnerError::EmptyCandidateList => {
                write!(f, "aqs-runner: search config has no candidates")
            }
        }
    }
}

impl std::error::Error for AqsRunnerError {}

// ============================================================================
// Runner config + result
// ============================================================================

/// Runner configuration: the search to run plus error/skip policy.
#[derive(Debug, Clone)]
pub struct AqsRunnerConfig {
    pub search: AqsSearchConfig,
    /// If `true`, abort the whole run on the first evaluator error.
    pub stop_on_first_error: bool,
    /// If `true`, candidates unsupported by the evaluator's capabilities
    /// are skipped; if `false`, they become `UnsupportedCandidate` errors.
    pub skip_unsupported: bool,
}

impl AqsRunnerConfig {
    pub fn new(search: AqsSearchConfig) -> Self {
        Self {
            search,
            stop_on_first_error: false,
            skip_unsupported: true,
        }
    }
}

/// A candidate that was skipped, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AqsSkippedCandidate {
    pub candidate_name: String,
    pub reason: String,
}

/// A candidate whose evaluation errored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AqsRunnerErrorRecord {
    pub candidate_name: String,
    pub error: AqsRunnerError,
}

/// The full runner output: the AQS-7 search result over the evaluated
/// candidates, plus bookkeeping of what was evaluated / skipped / errored.
#[derive(Debug, Clone)]
pub struct AqsRunnerResult {
    pub search_result: AqsSearchResult,
    pub evaluated: Vec<String>,
    pub skipped: Vec<AqsSkippedCandidate>,
    pub errors: Vec<AqsRunnerErrorRecord>,
}

// ============================================================================
// The runner
// ============================================================================

/// **AQS-8 entry point.** Drive the candidate grid through an evaluator
/// callback and rank the results via AQS-7.
///
/// Behaviour:
/// 1. iterate `config.search.candidates` in order (deterministic);
/// 2. for each, check support against `capabilities` — skip (if
///    `skip_unsupported`) or record an `UnsupportedCandidate` error;
/// 3. call `evaluator(spec)`; on `Ok`, accumulate the result; on `Err`,
///    record it and either stop (`stop_on_first_error`) or continue;
/// 4. feed the accumulated results to [`search_from_end_to_end_results`];
/// 5. return the [`AqsRunnerResult`].
///
/// An empty candidate list yields a result whose `search_result` has an
/// empty report (and an `EmptyCandidateList` error recorded) — never a
/// panic.
pub fn run_aqs_with_evaluator<F>(
    config: AqsRunnerConfig,
    capabilities: AqsEvaluatorCapabilities,
    mut evaluator: F,
) -> AqsRunnerResult
where
    F: FnMut(&AqsCandidateSpec) -> Result<EndToEndEvalResult, AqsRunnerError>,
{
    let search = &config.search;
    let mut results: Vec<EndToEndEvalResult> = Vec::new();
    let mut evaluated: Vec<String> = Vec::new();
    let mut skipped: Vec<AqsSkippedCandidate> = Vec::new();
    let mut errors: Vec<AqsRunnerErrorRecord> = Vec::new();

    if search.candidates.is_empty() {
        errors.push(AqsRunnerErrorRecord {
            candidate_name: String::new(),
            error: AqsRunnerError::EmptyCandidateList,
        });
    }

    for spec in &search.candidates {
        // (2) Support gate — decided before calling the evaluator.
        if !capabilities.supports(spec) {
            if config.skip_unsupported {
                skipped.push(AqsSkippedCandidate {
                    candidate_name: spec.name.clone(),
                    reason: "evaluator lacks required calibration capability".to_string(),
                });
                continue;
            } else {
                errors.push(AqsRunnerErrorRecord {
                    candidate_name: spec.name.clone(),
                    error: AqsRunnerError::UnsupportedCandidate(spec.name.clone()),
                });
                if config.stop_on_first_error {
                    break;
                }
                continue;
            }
        }

        // (3) Evaluate.
        match evaluator(spec) {
            Ok(r) => {
                evaluated.push(spec.name.clone());
                results.push(r);
            }
            Err(e) => {
                errors.push(AqsRunnerErrorRecord {
                    candidate_name: spec.name.clone(),
                    error: e,
                });
                if config.stop_on_first_error {
                    break;
                }
            }
        }
    }

    // (4) Rank via AQS-7 — single source of truth for classification.
    let search_result = search_from_end_to_end_results(
        &search.model_name,
        search.adr004_gate,
        search.baseline_memory_bytes,
        results,
    );

    AqsRunnerResult {
        search_result,
        evaluated,
        skipped,
        errors,
    }
}

// ============================================================================
// Tests (fake evaluators — no real model)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::certification::CertificationStatus;
    use crate::quant::search::{AqsPolicyKind, AqsSearchConfig};

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

    /// Map the real TinyLlama numbers by candidate id (the grid uses
    /// labels like `awq_g128_a0.25`; we key on the policy id family).
    fn fake_result_for(spec: &AqsCandidateSpec) -> EndToEndEvalResult {
        match spec.kind {
            AqsPolicyKind::Bf16 => res(&spec.name, 0.000063, true, 2_260_729_856),
            AqsPolicyKind::PlainInt8 { .. } => res(&spec.name, 1.260771, false, 1_165_688_832),
            AqsPolicyKind::Awq { .. } => res(&spec.name, 0.889217, true, 1_165_688_832),
            AqsPolicyKind::Hybrid { .. } => res(&spec.name, 0.831786, false, 1_266_614_272),
            AqsPolicyKind::Gptq { .. } => res(&spec.name, 1.405399, false, 1_167_265_792),
        }
    }

    const BASE: Option<u64> = Some(2_260_729_856);

    fn cfg() -> AqsRunnerConfig {
        AqsRunnerConfig::new(AqsSearchConfig::with_default_grid("tinyllama", BASE))
    }

    #[test]
    fn runner_evaluates_candidates_in_order() {
        let mut seen = Vec::new();
        let out = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            seen.push(spec.name.clone());
            Ok(fake_result_for(spec))
        });
        // Default grid order preserved.
        assert_eq!(seen[0], "bf16");
        assert_eq!(seen.last().unwrap(), "gptq_g128_b128_d0.01");
        assert_eq!(out.evaluated.len(), 7);
    }

    #[test]
    fn runner_accumulates_results() {
        let out = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            Ok(fake_result_for(spec))
        });
        assert_eq!(out.search_result.report.policies.len(), 7);
        assert_eq!(out.search_result.evaluated_count, 7);
    }

    #[test]
    fn runner_skips_unsupported_matrix_candidate() {
        // Capabilities: absmax yes, matrix no → GPTQ skipped, AWQ/Hybrid ok.
        let caps = AqsEvaluatorCapabilities {
            activation_absmax: true,
            activation_matrix: false,
        };
        let out = run_aqs_with_evaluator(cfg(), caps, |spec| Ok(fake_result_for(spec)));
        assert!(out.skipped.iter().any(|s| s.candidate_name == "gptq_g128_b128_d0.01"));
        assert_eq!(out.skipped.len(), 1);
        // bf16, int8, 3x awq, hybrid evaluated = 6.
        assert_eq!(out.evaluated.len(), 6);
    }

    #[test]
    fn runner_errors_on_unsupported_when_skip_false() {
        let mut c = cfg();
        c.skip_unsupported = false;
        // No calibration at all → awq/hybrid/gptq all unsupported.
        let out = run_aqs_with_evaluator(c, AqsEvaluatorCapabilities::none(), |spec| {
            Ok(fake_result_for(spec))
        });
        // 3 awq + 1 hybrid + 1 gptq = 5 unsupported errors.
        assert_eq!(out.errors.len(), 5);
        assert!(out
            .errors
            .iter()
            .all(|e| matches!(e.error, AqsRunnerError::UnsupportedCandidate(_))));
        // bf16 + plain_int8 still evaluated.
        assert_eq!(out.evaluated.len(), 2);
    }

    #[test]
    fn runner_calls_search_from_results() {
        let out = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            Ok(fake_result_for(spec))
        });
        // The search result is a real AqsSearchResult built by AQS-7.
        assert_eq!(out.search_result.config.model_name, "tinyllama");
        assert!(out.search_result.report.policy("bf16").is_some());
    }

    #[test]
    fn runner_preserves_best_certified_and_best_lossy() {
        let out = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            Ok(fake_result_for(spec))
        });
        assert_eq!(out.search_result.best_certified(), Some("bf16"));
        // best useful lossy is an AWQ variant (argmax true, compresses).
        let lossy = out.search_result.best_useful_lossy().unwrap();
        assert!(lossy.starts_with("awq"));
        assert_eq!(
            out.search_result.report.policy(lossy).unwrap().status,
            CertificationStatus::UsefulLossy
        );
        // Recommendation never substitutes lossy for certified.
        assert_eq!(out.search_result.recommended_policy(), Some("bf16"));
    }

    #[test]
    fn runner_handles_empty_candidate_list() {
        let mut search = AqsSearchConfig::with_default_grid("empty", BASE);
        search.candidates.clear();
        let out = run_aqs_with_evaluator(
            AqsRunnerConfig::new(search),
            AqsEvaluatorCapabilities::all(),
            |spec| Ok(fake_result_for(spec)),
        );
        assert_eq!(out.evaluated.len(), 0);
        assert!(out
            .errors
            .iter()
            .any(|e| e.error == AqsRunnerError::EmptyCandidateList));
        assert!(out.search_result.report.policies.is_empty());
    }

    #[test]
    fn runner_records_evaluation_error_and_continues_when_configured() {
        // stop_on_first_error = false (default). Fail plain_int8 only.
        let out = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            if matches!(spec.kind, AqsPolicyKind::PlainInt8 { .. }) {
                Err(AqsRunnerError::EvaluationFailed("boom".into()))
            } else {
                Ok(fake_result_for(spec))
            }
        });
        assert_eq!(out.errors.len(), 1);
        assert_eq!(out.errors[0].candidate_name, "plain_int8_g128");
        // The other 6 still evaluated.
        assert_eq!(out.evaluated.len(), 6);
    }

    #[test]
    fn runner_stops_on_first_error_when_configured() {
        let mut c = cfg();
        c.stop_on_first_error = true;
        // bf16 is first and succeeds; fail the 2nd (plain_int8).
        let out = run_aqs_with_evaluator(c, AqsEvaluatorCapabilities::all(), |spec| {
            if matches!(spec.kind, AqsPolicyKind::PlainInt8 { .. }) {
                Err(AqsRunnerError::EvaluationFailed("boom".into()))
            } else {
                Ok(fake_result_for(spec))
            }
        });
        // bf16 evaluated, then plain_int8 errors and we stop.
        assert_eq!(out.evaluated, vec!["bf16".to_string()]);
        assert_eq!(out.errors.len(), 1);
    }

    #[test]
    fn runner_result_is_deterministic() {
        let a = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            Ok(fake_result_for(spec))
        });
        let b = run_aqs_with_evaluator(cfg(), AqsEvaluatorCapabilities::all(), |spec| {
            Ok(fake_result_for(spec))
        });
        assert_eq!(a.evaluated, b.evaluated);
        assert_eq!(a.skipped, b.skipped);
        assert_eq!(
            a.search_result.manifest_draft(),
            b.search_result.manifest_draft()
        );
    }
}
