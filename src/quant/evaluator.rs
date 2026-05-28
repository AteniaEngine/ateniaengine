//! **AQS-2** — cheap per-tensor drift evaluator.
//!
//! Given a single F32 weight tensor and a [`QuantizationPolicy`], this
//! module measures how far the policy's in-place perturbation moves
//! the buffer away from the original. It is the base brick the future
//! AQS search (AQS-4) will call millions of times to rank candidate
//! policies per layer.
//!
//! ## What this is — and what it is NOT
//!
//! This evaluator does **NOT** run a model forward. It measures only:
//!
//! ```text
//!   F32 tensor original  vs  F32 tensor perturbed by the policy
//! ```
//!
//! That is a **cheap local signal, not final certification.** A tensor
//! with low local drift can still blow past ADR-004 once the error
//! cascades through dozens of matmuls + softmaxes (the β.4 → β.5
//! projection was off by ~60× for exactly this reason — see
//! `docs/HANDOFF_INT8_OUTLIER_BETA.md`). End-to-end ADR-004 validation
//! stays the job of the F64 forward harness; this is a pre-filter.
//!
//! ## Scope contract (AQS-2)
//!
//! * **No search** — `evaluate_tensor_policies` runs a list of policies
//!   and returns results *in input order*; it does not select, sort, or
//!   optimise. Selection logic belongs to AQS-4.
//! * **No GPTQ, no CUDA, no tier-planner, no CLI, no generation, no
//!   loader / manifest changes.**
//! * **Experimental, opt-in, CPU-only, isolated.** Only depends on
//!   [`crate::quant::policy`].
//!
//! See `docs/HANDOFF_AQS_2.md` for the milestone summary.

use crate::quant::policy::{CalibrationContext, PolicyError, QuantizationPolicy};

// ============================================================================
// Input / output types
// ============================================================================

/// Borrow-only description of one tensor to evaluate.
#[derive(Debug, Clone, Copy)]
pub struct TensorEvalInput<'a> {
    /// Human-readable identifier (e.g. the loader param name). Carried
    /// through to the result for reporting; not interpreted.
    pub name: &'a str,
    /// Row-major F32 values, length `product(shape)`.
    pub values: &'a [f32],
    /// Tensor shape. AQS-2 policies operate on `[K_in, N_out]`.
    pub shape: &'a [usize],
}

/// Drift + cost metrics for one (tensor, policy) pair.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEvalResult {
    /// Echoes [`TensorEvalInput::name`].
    pub tensor_name: String,
    /// [`QuantizationPolicy::id`] of the evaluated policy.
    pub policy_id: String,
    /// Number of elements (`product(shape)`).
    pub numel: usize,
    /// Target memory cost from [`QuantizationPolicy::memory_bytes`].
    pub memory_bytes: u64,
    /// L∞ drift: `max |original[i] - perturbed[i]|`. This is the same
    /// metric ADR-004 gates on, but measured *locally* on the weight
    /// buffer rather than end-to-end on logits.
    pub max_abs_diff: f32,
    /// L1 drift normalised by `numel`: `mean |Δ|`.
    pub mean_abs_diff: f32,
    /// Root-mean-square error: `sqrt(mean(Δ²))`.
    pub rmse: f32,
    /// **Local argmax over tensor values** — `Some(true)` when the flat
    /// index of `max(original)` equals the flat index of
    /// `max(perturbed)`. This is NOT logits argmax and says nothing
    /// about token-level behaviour; it is a coarse "did the dominant
    /// element survive quantisation" probe. `None` when the tensor is
    /// empty (no argmax to compare).
    pub argmax_match: Option<bool>,
}

impl TensorEvalResult {
    /// F32 baseline byte count for this tensor (`numel * 4`). Handy for
    /// computing a compression ratio against [`Self::memory_bytes`].
    pub fn baseline_f32_bytes(&self) -> u64 {
        (self.numel as u64) * 4
    }

    /// `memory_bytes / baseline_f32_bytes`. Lower is smaller. Returns
    /// `f64::NAN` for an empty tensor.
    pub fn compression_ratio(&self) -> f64 {
        let base = self.baseline_f32_bytes();
        if base == 0 {
            f64::NAN
        } else {
            (self.memory_bytes as f64) / (base as f64)
        }
    }
}

// ============================================================================
// Error model
// ============================================================================

/// Errors raised by [`evaluate_tensor_policy`].
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// `values` is empty — nothing to measure.
    EmptyInput,
    /// `values.len()` does not match `product(shape)`.
    ShapeMismatch { expected: usize, actual: usize },
    /// [`QuantizationPolicy::validate`] rejected the shape.
    PolicyValidationFailed(PolicyError),
    /// [`QuantizationPolicy::apply_inplace`] failed.
    PolicyApplicationFailed(PolicyError),
    /// The perturbed buffer contains a NaN / inf — the policy produced
    /// a numerically invalid result.
    NonFiniteResult { index: usize },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::EmptyInput => write!(f, "evaluator: input values are empty"),
            EvalError::ShapeMismatch { expected, actual } => write!(
                f,
                "evaluator: values.len() = {actual} does not match product(shape) = {expected}"
            ),
            EvalError::PolicyValidationFailed(e) => {
                write!(f, "evaluator: policy validation failed: {e}")
            }
            EvalError::PolicyApplicationFailed(e) => {
                write!(f, "evaluator: policy application failed: {e}")
            }
            EvalError::NonFiniteResult { index } => write!(
                f,
                "evaluator: policy produced a non-finite value at index {index}"
            ),
        }
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Core evaluation
// ============================================================================

/// Evaluate one [`QuantizationPolicy`] against one tensor.
///
/// Flow:
/// 1. validate input (`values` non-empty, `len == product(shape)`);
/// 2. `policy.validate(shape)`;
/// 3. clone `values` into a working buffer;
/// 4. `policy.apply_inplace(&mut buf, shape, cal)`;
/// 5. compute `max_abs_diff` / `mean_abs_diff` / `rmse` / `argmax_match`;
/// 6. read `policy.memory_bytes(shape)`.
///
/// The original `values` slice is never mutated.
pub fn evaluate_tensor_policy(
    input: TensorEvalInput<'_>,
    policy: &dyn QuantizationPolicy,
    cal: &CalibrationContext<'_>,
) -> Result<TensorEvalResult, EvalError> {
    // (1) Structural input validation.
    if input.values.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    let expected: usize = input.shape.iter().product();
    if input.values.len() != expected {
        return Err(EvalError::ShapeMismatch {
            expected,
            actual: input.values.len(),
        });
    }

    // (2) Policy-level structural validation (shape rank, group_size,
    //     alpha range, outlier_k bound, …). Surfaced as a distinct
    //     error variant from application failures.
    policy
        .validate(input.shape)
        .map_err(EvalError::PolicyValidationFailed)?;

    // (3) Work on a copy; the original is read-only.
    let mut perturbed = input.values.to_vec();

    // (4) Apply the policy in place.
    policy
        .apply_inplace(&mut perturbed, input.shape, cal)
        .map_err(EvalError::PolicyApplicationFailed)?;

    // (5) Drift metrics. Accumulate sums in f64 to avoid precision loss
    //     on large tensors; report f32 to match the rest of the surface.
    let numel = input.values.len();
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for (i, (&orig, &pert)) in input.values.iter().zip(perturbed.iter()).enumerate() {
        if !pert.is_finite() {
            return Err(EvalError::NonFiniteResult { index: i });
        }
        let d = (orig - pert).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        sum_abs += d as f64;
        sum_sq += (d as f64) * (d as f64);
    }
    let mean_abs_diff = (sum_abs / numel as f64) as f32;
    let rmse = (sum_sq / numel as f64).sqrt() as f32;

    // Local argmax-over-values probe (NOT logits argmax).
    let argmax_match = Some(argmax_by_value(input.values) == argmax_by_value(&perturbed));

    // (6) Memory estimate from the policy cost model.
    let memory_bytes = policy.memory_bytes(input.shape);

    Ok(TensorEvalResult {
        tensor_name: input.name.to_string(),
        policy_id: policy.id().to_string(),
        numel,
        memory_bytes,
        max_abs_diff,
        mean_abs_diff,
        rmse,
        argmax_match,
    })
}

/// Flat index of the maximum value (not abs). Ties resolve to the
/// lowest index. Returns 0 for an empty slice (never reached here — the
/// caller guards emptiness).
fn argmax_by_value(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Evaluate several policies against the same tensor and return the
/// results **in input order**.
///
/// This is a convenience batch wrapper — it is explicitly *not* a
/// search: it neither sorts, selects, nor prunes. AQS-4 owns selection.
///
/// The whole batch fails fast on the first policy that errors, so a
/// caller that wants per-policy resilience should call
/// [`evaluate_tensor_policy`] in a loop and handle each `Result`.
pub fn evaluate_tensor_policies(
    input: TensorEvalInput<'_>,
    policies: &[&dyn QuantizationPolicy],
    cal: &CalibrationContext<'_>,
) -> Result<Vec<TensorEvalResult>, EvalError> {
    let mut out = Vec::with_capacity(policies.len());
    for policy in policies {
        out.push(evaluate_tensor_policy(input, *policy, cal)?);
    }
    Ok(out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::policy::{
        AwqPolicy, Bf16Fallback, HybridPolicy, PlainInt8, QuantizationPolicy,
    };

    fn rand_weights(k: usize, n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(k * n);
        for _ in 0..(k * n) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as u32;
            let f = (u as f32) / (u32::MAX as f32);
            out.push(f * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn evaluate_bf16_policy_has_zero_drift() {
        let values = rand_weights(16, 32, 1);
        let shape = [16, 32];
        let input = TensorEvalInput {
            name: "w.bf16",
            values: &values,
            shape: &shape,
        };
        let r = evaluate_tensor_policy(input, &Bf16Fallback, &CalibrationContext::empty())
            .unwrap();
        assert_eq!(r.max_abs_diff, 0.0);
        assert_eq!(r.mean_abs_diff, 0.0);
        assert_eq!(r.rmse, 0.0);
        assert_eq!(r.argmax_match, Some(true));
        assert_eq!(r.policy_id, "bf16");
        assert_eq!(r.tensor_name, "w.bf16");
        assert_eq!(r.numel, 16 * 32);
        assert_eq!(r.memory_bytes, (16 * 32) as u64 * 2);
    }

    #[test]
    fn evaluate_plain_int8_reports_positive_drift() {
        let values = rand_weights(32, 16, 2);
        let shape = [32, 16];
        let input = TensorEvalInput {
            name: "w.int8",
            values: &values,
            shape: &shape,
        };
        let r = evaluate_tensor_policy(input, &PlainInt8::new(8), &CalibrationContext::empty())
            .unwrap();
        assert!(r.max_abs_diff > 0.0, "INT8 should introduce drift");
        assert!(r.mean_abs_diff > 0.0);
        assert!(r.rmse > 0.0);
        // RMSE >= mean_abs_diff by Jensen's inequality.
        assert!(r.rmse >= r.mean_abs_diff - 1e-6);
        // INT8 is lighter than F32 baseline.
        assert!(r.compression_ratio() < 1.0);
    }

    #[test]
    fn evaluate_awq_requires_activation_stats() {
        let values = rand_weights(16, 16, 3);
        let shape = [16, 16];
        let input = TensorEvalInput {
            name: "w.awq",
            values: &values,
            shape: &shape,
        };
        let err =
            evaluate_tensor_policy(input, &AwqPolicy::new(8, 0.3), &CalibrationContext::empty())
                .unwrap_err();
        assert!(matches!(
            err,
            EvalError::PolicyApplicationFailed(PolicyError::MissingActivationStats)
        ));
    }

    #[test]
    fn evaluate_awq_with_activation_stats_returns_metrics() {
        let values = rand_weights(16, 16, 4);
        let shape = [16, 16];
        let act = vec![0.5_f32; 16];
        let input = TensorEvalInput {
            name: "w.awq",
            values: &values,
            shape: &shape,
        };
        let r = evaluate_tensor_policy(
            input,
            &AwqPolicy::new(8, 0.3),
            &CalibrationContext::with_activations(&act),
        )
        .unwrap();
        assert!(r.max_abs_diff > 0.0);
        assert!(r.rmse.is_finite());
        assert_eq!(r.policy_id, "awq");
    }

    #[test]
    fn evaluate_hybrid_returns_metrics() {
        let values = rand_weights(16, 16, 5);
        let shape = [16, 16];
        let act = vec![0.5_f32; 16];
        let input = TensorEvalInput {
            name: "w.hybrid",
            values: &values,
            shape: &shape,
        };
        let r = evaluate_tensor_policy(
            input,
            &HybridPolicy::new(8, 0.3, 2),
            &CalibrationContext::with_activations(&act),
        )
        .unwrap();
        assert!(r.max_abs_diff > 0.0);
        assert_eq!(r.policy_id, "hybrid_awq_outlier");
        // Hybrid carries an F32 outlier sidecar -> heavier than plain INT8.
        let plain = evaluate_tensor_policy(
            input,
            &PlainInt8::new(8),
            &CalibrationContext::empty(),
        )
        .unwrap();
        assert!(r.memory_bytes > plain.memory_bytes);
    }

    #[test]
    fn evaluate_reports_memory_bytes_from_policy() {
        let values = rand_weights(64, 64, 6);
        let shape = [64, 64];
        let input = TensorEvalInput {
            name: "w",
            values: &values,
            shape: &shape,
        };
        let policy = PlainInt8::new(16);
        let r = evaluate_tensor_policy(input, &policy, &CalibrationContext::empty()).unwrap();
        assert_eq!(r.memory_bytes, policy.memory_bytes(&shape));
    }

    #[test]
    fn evaluate_rejects_shape_mismatch() {
        let values = rand_weights(4, 4, 7); // 16 elems
        let shape = [4, 8]; // expects 32
        let input = TensorEvalInput {
            name: "w",
            values: &values,
            shape: &shape,
        };
        let err = evaluate_tensor_policy(input, &Bf16Fallback, &CalibrationContext::empty())
            .unwrap_err();
        assert_eq!(
            err,
            EvalError::ShapeMismatch {
                expected: 32,
                actual: 16
            }
        );
    }

    #[test]
    fn evaluate_rejects_empty_values() {
        let values: Vec<f32> = Vec::new();
        let shape = [0, 0];
        let input = TensorEvalInput {
            name: "w",
            values: &values,
            shape: &shape,
        };
        let err = evaluate_tensor_policy(input, &Bf16Fallback, &CalibrationContext::empty())
            .unwrap_err();
        assert_eq!(err, EvalError::EmptyInput);
    }

    /// A policy that deliberately writes a NaN, used only to exercise
    /// the non-finite guard. Lives in the test module so it cannot leak
    /// into the productive surface.
    struct NanPolicy;
    impl QuantizationPolicy for NanPolicy {
        fn id(&self) -> &'static str {
            "nan_test_only"
        }
        fn validate(&self, _shape: &[usize]) -> Result<(), PolicyError> {
            Ok(())
        }
        fn apply_inplace(
            &self,
            weights: &mut [f32],
            _shape: &[usize],
            _cal: &CalibrationContext<'_>,
        ) -> Result<(), PolicyError> {
            if let Some(first) = weights.first_mut() {
                *first = f32::NAN;
            }
            Ok(())
        }
        fn memory_bytes(&self, shape: &[usize]) -> u64 {
            (shape.iter().product::<usize>() as u64) * 4
        }
    }

    #[test]
    fn evaluate_detects_non_finite_output() {
        let values = rand_weights(4, 4, 8);
        let shape = [4, 4];
        let input = TensorEvalInput {
            name: "w",
            values: &values,
            shape: &shape,
        };
        let err = evaluate_tensor_policy(input, &NanPolicy, &CalibrationContext::empty())
            .unwrap_err();
        assert_eq!(err, EvalError::NonFiniteResult { index: 0 });
    }

    #[test]
    fn evaluate_batch_preserves_policy_order() {
        let values = rand_weights(16, 16, 9);
        let shape = [16, 16];
        let input = TensorEvalInput {
            name: "w",
            values: &values,
            shape: &shape,
        };
        let bf16 = Bf16Fallback;
        let int8 = PlainInt8::new(8);
        let policies: [&dyn QuantizationPolicy; 2] = [&bf16, &int8];
        let results =
            evaluate_tensor_policies(input, &policies, &CalibrationContext::empty()).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].policy_id, "bf16");
        assert_eq!(results[1].policy_id, "plain_int8");
        // Order is by input, not by drift: bf16 (0 drift) stays first.
        assert_eq!(results[0].max_abs_diff, 0.0);
    }
}
