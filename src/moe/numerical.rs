//! **MOE-16** — numerical-equivalence metrics + report for MoE outputs.
//!
//! Compares an Atenia MoE-block output (f32) against a reference output and
//! reports the same metric family AQS / ADR-004 use: `max_abs_diff`,
//! `mean_abs_diff`, `rmse`, `argmax_match`. All accumulation is in **f64**.
//!
//! This module is pure math over two slices — it does not load models,
//! fixtures, or run forwards. The integration test
//! (`tests/moe_numerical_equivalence_test.rs`) loads committed reference
//! fixtures (generated offline by `fixtures/moe/generate_reference.py`, never
//! in CI), runs Atenia's `RealMoeLayer::forward`, and feeds both vectors here.
//!
//! ## Two references (ADR-004 structure)
//!
//! * **Primary (gates pass/fail):** an independent f64 reimplementation of
//!   the EXACT operation Atenia performs. Atenia's f32 output must match it
//!   within `MOE_NUMERICAL_TOLERANCE`. This catches indexing / transpose /
//!   packed-split bugs.
//! * **Secondary (informative):** the real HuggingFace `transformers` MoE
//!   block in f64. Differences reflect convention gaps Atenia does not (yet)
//!   implement (e.g. `norm_topk_prob`, sigmoid-gated shared expert) and do
//!   not gate the test — they are reported as a finding.

/// ADR-004 tolerance for the primary (same-operation) comparison.
pub const MOE_NUMERICAL_TOLERANCE: f64 = 0.5;

/// Numerical comparison metrics between two equal-length vectors. All fields
/// are computed with f64 accumulation regardless of input precision.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericalMetrics {
    pub len: usize,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub rmse: f64,
    /// Whether `argmax(atenia) == argmax(reference)`.
    pub argmax_match: bool,
}

/// Index of the maximum element (first on ties). `None` if empty.
fn argmax(v: &[f32]) -> Option<usize> {
    if v.is_empty() {
        return None;
    }
    let mut best = 0usize;
    let mut best_v = v[0] as f64;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if (x as f64) > best_v {
            best_v = x as f64;
            best = i;
        }
    }
    Some(best)
}

impl NumericalMetrics {
    /// Compute metrics between an Atenia output and a reference. Returns
    /// `None` if the lengths differ or are zero. f64 accumulation.
    pub fn compute(atenia: &[f32], reference: &[f32]) -> Option<NumericalMetrics> {
        if atenia.is_empty() || atenia.len() != reference.len() {
            return None;
        }
        let n = atenia.len();
        let mut max_abs = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let d = (atenia[i] as f64) - (reference[i] as f64);
            let ad = d.abs();
            if ad > max_abs {
                max_abs = ad;
            }
            sum_abs += ad;
            sum_sq += d * d;
        }
        Some(NumericalMetrics {
            len: n,
            max_abs_diff: max_abs,
            mean_abs_diff: sum_abs / n as f64,
            rmse: (sum_sq / n as f64).sqrt(),
            argmax_match: argmax(atenia) == argmax(reference),
        })
    }

    /// Whether these metrics pass the ADR-004 gate: `max_abs_diff <
    /// tolerance` AND argmax agrees.
    pub fn passes(&self, tolerance: f64) -> bool {
        self.max_abs_diff < tolerance && self.argmax_match
    }
}

/// A numerical-validation report for one model's MoE block.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeNumericalReport {
    pub model: String,
    pub layers: usize,
    pub experts: usize,
    pub metrics: NumericalMetrics,
    /// Pass against the primary tolerance ([`MOE_NUMERICAL_TOLERANCE`]).
    pub pass: bool,
}

impl MoeNumericalReport {
    /// Build a report from computed metrics, applying the primary tolerance.
    pub fn new(
        model: impl Into<String>,
        layers: usize,
        experts: usize,
        metrics: NumericalMetrics,
    ) -> Self {
        let pass = metrics.passes(MOE_NUMERICAL_TOLERANCE);
        Self { model: model.into(), layers, experts, metrics, pass }
    }

    /// One-line human summary.
    pub fn summary(&self) -> String {
        format!(
            "{}: layers={}, experts={}, max_abs_diff={:.3e}, mean_abs_diff={:.3e}, rmse={:.3e}, argmax_match={}, pass={}",
            self.model,
            self.layers,
            self.experts,
            self.metrics.max_abs_diff,
            self.metrics.mean_abs_diff,
            self.metrics.rmse,
            self.metrics.argmax_match,
            self.pass
        )
    }
}

// ============================================================================
// Tests (pure math — no fixtures, no model)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_diff() {
        let a = vec![1.0_f32, -2.0, 3.0, 0.5];
        let m = NumericalMetrics::compute(&a, &a).unwrap();
        assert_eq!(m.max_abs_diff, 0.0);
        assert_eq!(m.mean_abs_diff, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert!(m.argmax_match);
        assert!(m.passes(MOE_NUMERICAL_TOLERANCE));
    }

    #[test]
    fn known_difference_metrics() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, -1.0, 0.0];
        let m = NumericalMetrics::compute(&a, &b).unwrap();
        assert!((m.max_abs_diff - 1.0).abs() < 1e-12);
        assert!((m.mean_abs_diff - (2.0 / 3.0)).abs() < 1e-12);
        assert!((m.rmse - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn length_mismatch_returns_none() {
        assert!(NumericalMetrics::compute(&[1.0, 2.0], &[1.0]).is_none());
        assert!(NumericalMetrics::compute(&[], &[]).is_none());
    }

    #[test]
    fn argmax_mismatch_is_detected() {
        let a = vec![0.1_f32, 0.9];
        let b = vec![0.9_f32, 0.1];
        let m = NumericalMetrics::compute(&a, &b).unwrap();
        assert!(!m.argmax_match);
        assert!(!m.passes(MOE_NUMERICAL_TOLERANCE)); // argmax gate fails
    }

    #[test]
    fn metrics_are_deterministic() {
        let a = vec![0.25_f32, -0.5, 0.75, 1.5, -2.25];
        let b = vec![0.24_f32, -0.51, 0.76, 1.49, -2.26];
        let m1 = NumericalMetrics::compute(&a, &b).unwrap();
        let m2 = NumericalMetrics::compute(&a, &b).unwrap();
        assert_eq!(m1, m2);
    }

    #[test]
    fn validation_report_builds() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0001_f32, 2.0, 2.9999];
        let m = NumericalMetrics::compute(&a, &b).unwrap();
        let r = MoeNumericalReport::new("test-model", 1, 4, m);
        assert_eq!(r.model, "test-model");
        assert_eq!(r.layers, 1);
        assert_eq!(r.experts, 4);
        assert!(r.pass);
        assert!(r.summary().contains("pass=true"));
    }

    #[test]
    fn large_diff_fails_tolerance() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let m = NumericalMetrics::compute(&a, &b).unwrap();
        let r = MoeNumericalReport::new("m", 1, 2, m);
        assert!(!r.pass, "max_abs_diff 1.0 must fail the 0.5 tolerance");
    }
}
