//! **AQS-4** — end-to-end policy evaluation orchestration primitives.
//!
//! AQS-2 measures drift *locally* on a single weight buffer. AQS-3
//! showed that local drift is the wrong instrument for GPTQ: GPTQ trades
//! per-element drift for column-level error cancellation, which only
//! shows up once the weights are run through a real forward. AQS-4 is the
//! first orchestration that closes that gap — it perturbs *real* model
//! weights with a [`QuantizationPolicy`], runs a *real* forward, and
//! compares the resulting **logits** against the certified F64 fixture.
//!
//! ## What lives here vs. in the harness
//!
//! This module holds the **model-agnostic, reusable** pieces:
//!
//! * [`PolicyEvalCandidate`] — a named `&dyn QuantizationPolicy`.
//! * [`EndToEndEvalResult`] — the per-candidate logit-drift report.
//! * [`logit_drift_metrics`] — the pure logits-vs-reference metric.
//! * [`render_result_table`] — a plain-text comparison table.
//!
//! The **model-specific orchestration** (loading TinyLlama, the
//! calibration forward, weight perturbation, the real forward) lives in
//! `tests/aqs4_end_to_end_test.rs`, reusing the β / β-pivot harness
//! pattern and the new `WeightStore::perturb_all_proj_with_policy`
//! helper. Keeping env-vars / fixtures / safetensors paths out of `src/`
//! is deliberate — the library stays free of test-only I/O.
//!
//! ## Scope contract (AQS-4)
//!
//! * **No automatic search** — this evaluates an explicit, ordered list
//!   of candidates and reports; it never selects, prunes, or optimises.
//! * **No CUDA, no tier-planner, no CLI, no generation, no loader /
//!   manifest changes.** CPU-only, opt-in, experimental.
//! * Nothing here is reachable from the productive path.
//!
//! See `docs/HANDOFF_AQS_4.md` for the TinyLlama results.

use crate::quant::policy::QuantizationPolicy;

/// A named candidate policy to evaluate end-to-end.
pub struct PolicyEvalCandidate<'a> {
    /// Human-readable label for the result table (e.g. `"bf16"`,
    /// `"awq_a0.25"`, `"gptq_d0.01"`).
    pub name: &'a str,
    /// The policy to apply to every eligible weight.
    pub policy: &'a dyn QuantizationPolicy,
}

/// End-to-end logit-drift report for one candidate.
///
/// Unlike [`crate::quant::evaluator::TensorEvalResult`], every metric
/// here is computed on **real forward logits vs. the certified F64
/// fixture** — not on a local weight buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct EndToEndEvalResult {
    /// Echoes [`PolicyEvalCandidate::name`].
    pub candidate_name: String,
    /// L∞ logit drift: `max |logit[i] − reference_f64[i]|`. This is the
    /// metric ADR-004 gates on (`< 0.5`).
    pub max_abs_diff: f32,
    /// Mean absolute logit drift.
    pub mean_abs_diff: f32,
    /// Root-mean-square logit drift.
    pub rmse: f32,
    /// Whether the argmax token matched the reference at **every**
    /// evaluated sequence position (true logits argmax, not the AQS-2
    /// local value argmax).
    pub argmax_match: bool,
    /// Approximate target memory cost reported by the policy
    /// (`Σ policy.memory_bytes(shape)` over perturbed params, supplied by
    /// the harness — this struct just carries it).
    pub memory_bytes: u64,
}

impl EndToEndEvalResult {
    /// Whether this candidate passes the ADR-004 strict logit gate
    /// (`max_abs_diff < 0.5`).
    pub fn passes_adr_004(&self) -> bool {
        (self.max_abs_diff as f64) < 0.5
    }
}

/// Compute logit-drift metrics of `logits` (F32, from a real forward)
/// against the certified `reference` (F64 fixture).
///
/// `vocab_size` and `positions` describe the logit layout: the buffer is
/// `positions × vocab_size` row-major. `argmax_match` is `true` iff the
/// argmax token agrees with the reference at every position.
///
/// Returns `None` if the buffers' lengths disagree or are not a clean
/// `positions × vocab_size`.
pub fn logit_drift_metrics(
    logits: &[f32],
    reference: &[f64],
    vocab_size: usize,
    positions: usize,
) -> Option<(f32, f32, f32, bool)> {
    if logits.len() != reference.len() {
        return None;
    }
    if vocab_size == 0 || logits.len() != vocab_size * positions {
        return None;
    }

    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for (&a, &t) in logits.iter().zip(reference.iter()) {
        let d = ((a as f64) - t).abs();
        if d as f32 > max_abs {
            max_abs = d as f32;
        }
        sum_abs += d;
        sum_sq += d * d;
    }
    let n = logits.len() as f64;
    let mean_abs = (sum_abs / n) as f32;
    let rmse = (sum_sq / n).sqrt() as f32;

    let mut argmax_match = true;
    for pos in 0..positions {
        let s = pos * vocab_size;
        let e = s + vocab_size;
        let a_id = argmax_f32(&logits[s..e]);
        let t_id = argmax_f64(&reference[s..e]);
        if a_id != t_id {
            argmax_match = false;
            break;
        }
    }

    Some((max_abs, mean_abs, rmse, argmax_match))
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

fn argmax_f64(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Render a plain-text comparison table for a slice of results, in the
/// given order (no sorting — AQS-4 reports, it does not rank).
pub fn render_result_table(results: &[EndToEndEvalResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<18} {:>12} {:>12} {:>12} {:>8} {:>14} {:>8}\n",
        "candidate", "max_diff", "mean_diff", "rmse", "argmax", "memory_bytes", "ADR-004"
    ));
    out.push_str(&"-".repeat(88));
    out.push('\n');
    for r in results {
        out.push_str(&format!(
            "{:<18} {:>12.6} {:>12.6} {:>12.6} {:>8} {:>14} {:>8}\n",
            r.candidate_name,
            r.max_abs_diff,
            r.mean_abs_diff,
            r.rmse,
            if r.argmax_match { "true" } else { "false" },
            r.memory_bytes,
            if r.passes_adr_004() { "PASS" } else { "FAIL" },
        ));
    }
    out
}

// ============================================================================
// Tests (light — no model, no I/O)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_metrics_zero_drift_on_identical() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5, 9.0, 0.1];
        let reference: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
        let (max, mean, rmse, argmax) =
            logit_drift_metrics(&logits, &reference, 3, 2).unwrap();
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(rmse, 0.0);
        assert!(argmax);
    }

    #[test]
    fn logit_metrics_detect_argmax_flip() {
        // pos 0: logits argmax = idx2 (3.0); reference argmax = idx0 (9.0)
        let logits = vec![1.0_f32, 2.0, 3.0];
        let reference = vec![9.0_f64, 2.0, 3.0];
        let (max, _, _, argmax) = logit_drift_metrics(&logits, &reference, 3, 1).unwrap();
        assert!((max - 8.0).abs() < 1e-5);
        assert!(!argmax, "argmax should not match");
    }

    #[test]
    fn logit_metrics_reject_bad_layout() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let reference = vec![1.0_f64, 2.0, 3.0];
        // vocab*positions != len
        assert!(logit_drift_metrics(&logits, &reference, 2, 2).is_none());
        // length mismatch
        assert!(logit_drift_metrics(&logits, &[1.0, 2.0], 3, 1).is_none());
    }

    #[test]
    fn result_table_is_stable_and_ordered() {
        let results = vec![
            EndToEndEvalResult {
                candidate_name: "bf16".into(),
                max_abs_diff: 0.0,
                mean_abs_diff: 0.0,
                rmse: 0.0,
                argmax_match: true,
                memory_bytes: 200,
            },
            EndToEndEvalResult {
                candidate_name: "int8".into(),
                max_abs_diff: 1.26,
                mean_abs_diff: 0.1,
                rmse: 0.2,
                argmax_match: false,
                memory_bytes: 100,
            },
        ];
        let table = render_result_table(&results);
        let bf16_pos = table.find("bf16").unwrap();
        let int8_pos = table.find("int8").unwrap();
        assert!(bf16_pos < int8_pos, "order must follow input");
        assert!(table.contains("PASS"));
        assert!(table.contains("FAIL"));
    }

    #[test]
    fn passes_adr_004_boundary() {
        let mut r = EndToEndEvalResult {
            candidate_name: "x".into(),
            max_abs_diff: 0.49,
            mean_abs_diff: 0.0,
            rmse: 0.0,
            argmax_match: true,
            memory_bytes: 0,
        };
        assert!(r.passes_adr_004());
        r.max_abs_diff = 0.5;
        assert!(!r.passes_adr_004(), "0.5 is the strict threshold");
    }
}
