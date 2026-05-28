//! **AQS-3** — experimental, simplified GPTQ reconstruction (CPU-only).
//!
//! This is a deliberately *simplified* GPTQ: the goal is to capture the
//! one mechanism that distinguishes GPTQ from plain absmax INT8 and from
//! the β-pivot AWQ family — **error-aware quantisation driven by a
//! diagonal Hessian** — without any of the machinery that only matters
//! for speed or paper-exact reproduction.
//!
//! ## What this implements
//!
//! For a row-major `[K_in, N_out]` weight `W` and a per-input-channel
//! diagonal Hessian `h` (length `K`, derived from calibration
//! activations via [`approximate_hessian_diag`]):
//!
//! 1. **Hessian-regularised scale search.** Per (group, output column),
//!    pick the INT8 group scale that minimises the *Hessian-weighted*
//!    reconstruction error
//!    `Σ_k (h[k] + λ)·(W[k,n] − dequant)²` over a small grid of clip
//!    factors. Channels with large `h` (high activation energy) pull the
//!    scale toward preserving them. This is the error-aware core.
//! 2. **Sequential error diffusion** along the K axis within each
//!    column: the residual of each quantised weight is carried forward to
//!    the next weight. This is the cheap scalar surrogate for GPTQ's
//!    "compensate the not-yet-quantised weights" step (which, with a
//!    purely diagonal Hessian, has no off-diagonal coupling to exploit).
//!
//! The output is a plain F32 reconstruction in place — same contract as
//! every other [`crate::quant::policy::QuantizationPolicy`]; the runtime
//! never learns GPTQ exists.
//!
//! ## What this explicitly does NOT implement
//!
//! * No full `K×K` Hessian, no inverse, no Cholesky, no blockwise update
//!   (`O(N³)` avoided — everything here is `O(K·N·grid)`).
//! * No activation reordering (act-order), no group-wise permutation.
//! * No bit-packing, no INT4, no optimised kernel, no CUDA, no inference
//!   speedup. The reconstruction stays F32 for measurement only.
//! * No claim of paper-exact GPTQ numerics.
//!
//! It is a **local proxy** to answer one question: does error-aware
//! quantisation push local drift below the AWQ/hybrid plateau? End-to-end
//! ADR-004 certification remains the F64 forward harness's job (AQS-4+).

/// Errors returned by [`apply_gptq_reconstruction_inplace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GptqError {
    /// `shape` is not exactly 2 axes.
    NotTwoDimensional { rank: usize },
    /// `weights.len()` does not match `product(shape)`.
    LengthMismatch { expected: usize, actual: usize },
    /// `group_size == 0`.
    InvalidGroupSize,
    /// `damp_percent` is non-finite or outside `[0, 1)`.
    InvalidDampPercent,
    /// `hessian_diag.len()` does not match `K = shape[0]`.
    HessianLengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for GptqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GptqError::NotTwoDimensional { rank } => {
                write!(f, "gptq: shape must be 2D (got rank {rank})")
            }
            GptqError::LengthMismatch { expected, actual } => write!(
                f,
                "gptq: weights.len() = {actual} does not match K * N = {expected}"
            ),
            GptqError::InvalidGroupSize => write!(f, "gptq: group_size must be > 0"),
            GptqError::InvalidDampPercent => {
                write!(f, "gptq: damp_percent must be finite in [0, 1)")
            }
            GptqError::HessianLengthMismatch { expected, actual } => write!(
                f,
                "gptq: hessian_diag.len() = {actual} does not match K = {expected}"
            ),
        }
    }
}

impl std::error::Error for GptqError {}

/// Configuration for the simplified GPTQ reconstruction.
#[derive(Debug, Clone, Copy)]
pub struct GptqConfig {
    /// Quantisation group size along the K axis (matches the M9.4
    /// contract). `128` is a reasonable default.
    pub group_size: usize,
    /// Hessian diagonal damping, expressed as a fraction of the mean
    /// Hessian (`λ = damp_percent · mean(h)`). The classic GPTQ value is
    /// ~`0.01`. Must be in `[0, 1)`.
    pub damp_percent: f32,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            damp_percent: 0.01,
        }
    }
}

/// Strictly-positive floor applied to every Hessian diagonal entry so
/// the regularised weights never collapse to zero.
const HESSIAN_FLOOR: f32 = 1e-12;

/// Clip factors swept by the scale search. `1.0` is plain absmax; values
/// below clip outliers in exchange for tighter resolution on the bulk —
/// the Hessian-weighted cost decides which wins per group.
const CLIP_FACTORS: [f32; 11] = [
    1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
];

/// **AQS-3** — approximate the diagonal of the layer Hessian
/// `H = E[x xᵀ]` from calibration activations.
///
/// `shape = [num_samples, K]` (row-major). Returns a length-`K` vector
/// where `h[k] = (1/S) Σ_s activations[s·K + k]²` — the per-input-channel
/// second moment, floored at [`HESSIAN_FLOOR`] so every entry is strictly
/// positive.
///
/// Passing a single activation vector as `shape = [1, K]` yields
/// `h[k] = x_k²`, which is how [`crate::quant::policy::GptqPolicy`]
/// derives the Hessian from the per-row activation absmax it receives in
/// the calibration context.
///
/// Off-diagonal terms are intentionally discarded — see module docs.
pub fn approximate_hessian_diag(activations: &[f32], shape: &[usize]) -> Vec<f32> {
    // Tolerant: treat anything that is not a clean 2D [S, K] as a single
    // row of length `activations.len()`.
    let (s, k) = if shape.len() == 2 && shape[0] * shape[1] == activations.len() {
        (shape[0], shape[1])
    } else {
        (1, activations.len())
    };
    if k == 0 {
        return Vec::new();
    }
    let mut h = vec![0.0_f64; k];
    for sample in 0..s {
        let base = sample * k;
        for (col, slot) in h.iter_mut().enumerate() {
            let x = activations[base + col] as f64;
            *slot += x * x;
        }
    }
    let inv_s = 1.0 / (s.max(1) as f64);
    h.into_iter()
        .map(|v| ((v * inv_s) as f32).max(HESSIAN_FLOOR))
        .collect()
}

/// **AQS-3** — apply the simplified GPTQ reconstruction in place.
///
/// See the module docs for the two-stage algorithm (Hessian-regularised
/// scale search + sequential error diffusion). Deterministic and
/// CPU-only. `weights` is overwritten with the F32 reconstruction.
pub fn apply_gptq_reconstruction_inplace(
    weights: &mut [f32],
    shape: &[usize],
    hessian_diag: &[f32],
    config: &GptqConfig,
) -> Result<(), GptqError> {
    if shape.len() != 2 {
        return Err(GptqError::NotTwoDimensional { rank: shape.len() });
    }
    let group_size = config.group_size;
    if group_size == 0 {
        return Err(GptqError::InvalidGroupSize);
    }
    if !config.damp_percent.is_finite() || config.damp_percent < 0.0 || config.damp_percent >= 1.0
    {
        return Err(GptqError::InvalidDampPercent);
    }
    let (k_rows, n_cols) = (shape[0], shape[1]);
    let expected = k_rows * n_cols;
    if weights.len() != expected {
        return Err(GptqError::LengthMismatch {
            expected,
            actual: weights.len(),
        });
    }
    if hessian_diag.len() != k_rows {
        return Err(GptqError::HessianLengthMismatch {
            expected: k_rows,
            actual: hessian_diag.len(),
        });
    }
    if k_rows == 0 || n_cols == 0 {
        return Ok(());
    }

    // Hessian damping term λ = damp_percent · mean(h).
    let mean_h: f64 =
        hessian_diag.iter().map(|v| *v as f64).sum::<f64>() / (hessian_diag.len() as f64);
    let lambda = (config.damp_percent as f64) * mean_h;

    let num_groups = (k_rows + group_size - 1) / group_size;

    for n in 0..n_cols {
        for g in 0..num_groups {
            let lo = g * group_size;
            let hi = ((g + 1) * group_size).min(k_rows);

            // Group absmax over the output column.
            let mut absmax = 0.0_f32;
            for k in lo..hi {
                let v = weights[k * n_cols + n].abs();
                if v > absmax {
                    absmax = v;
                }
            }

            // Degenerate (all-zero) group → exact identity (preserves the
            // zero tensor; nothing to quantise).
            if absmax <= 0.0 {
                for k in lo..hi {
                    weights[k * n_cols + n] = 0.0;
                }
                continue;
            }

            let base_scale = absmax / 127.0;

            // (1) Hessian-regularised scale search over clip factors.
            let mut best_scale = base_scale;
            let mut best_cost = f64::INFINITY;
            for &f in CLIP_FACTORS.iter() {
                let s = base_scale * f;
                if s <= 0.0 {
                    continue;
                }
                let mut cost = 0.0_f64;
                for k in lo..hi {
                    let w = weights[k * n_cols + n];
                    let q = (w / s).round().clamp(-127.0, 127.0);
                    let recon = q * s;
                    let d = (w - recon) as f64;
                    let h = (hessian_diag[k] as f64) + lambda;
                    cost += h * d * d;
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_scale = s;
                }
            }

            // (2) Sequential error diffusion down the column within the
            // group, using the chosen scale. The residual of each weight
            // is carried into the next — the scalar surrogate for GPTQ's
            // compensation step.
            let mut carry = 0.0_f32;
            for k in lo..hi {
                let idx = k * n_cols + n;
                let w_adj = weights[idx] + carry;
                let q = (w_adj / best_scale).round().clamp(-127.0, 127.0);
                let recon = q * best_scale;
                carry = w_adj - recon;
                weights[idx] = recon;
            }
        }
    }

    Ok(())
}

/// Approximate target memory cost (bytes) of a GPTQ-quantised
/// `[K, N]` tensor: INT8 payload + per-(group, column) scales +
/// the diagonal Hessian metadata (length `K`).
pub fn gptq_memory_bytes(shape: &[usize], group_size: usize) -> u64 {
    let numel: usize = shape.iter().product();
    let n = *shape.last().unwrap_or(&0);
    let k = if shape.len() >= 2 { shape[0] } else { 0 };
    let gs = group_size.max(1);
    let num_groups = if k == 0 { 0 } else { (k + gs - 1) / gs };
    (numel as u64) + (num_groups as u64) * (n as u64) * 4 + (k as u64) * 4
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::evaluator::{evaluate_tensor_policy, TensorEvalInput};
    use crate::quant::policy::{
        AwqPolicy, CalibrationContext, GptqPolicy, QuantizationPolicy,
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
    fn gptq_hessian_diag_is_positive() {
        let act = rand_weights(1, 32, 11);
        let h = approximate_hessian_diag(&act, &[1, 32]);
        assert_eq!(h.len(), 32);
        for &v in &h {
            assert!(v > 0.0, "hessian diag entry must be strictly positive");
            assert!(v.is_finite());
        }
        // Zero activations still floor to a positive value.
        let h0 = approximate_hessian_diag(&vec![0.0_f32; 16], &[1, 16]);
        for &v in &h0 {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn gptq_hessian_diag_averages_samples() {
        // Two samples, K=2. Column 0: [3, 0] -> mean sq = 4.5.
        //                    Column 1: [0, 4] -> mean sq = 8.0.
        let act = vec![3.0, 0.0, 0.0, 4.0];
        let h = approximate_hessian_diag(&act, &[2, 2]);
        assert!((h[0] - 4.5).abs() < 1e-5);
        assert!((h[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn gptq_reconstruction_is_deterministic_and_finite() {
        let shape = [32, 16];
        let original = rand_weights(32, 16, 12);
        let h = vec![1.0_f32; 32];
        let cfg = GptqConfig {
            group_size: 8,
            damp_percent: 0.01,
        };
        let mut a = original.clone();
        let mut b = original.clone();
        apply_gptq_reconstruction_inplace(&mut a, &shape, &h, &cfg).unwrap();
        apply_gptq_reconstruction_inplace(&mut b, &shape, &h, &cfg).unwrap();
        assert_eq!(a, b, "GPTQ reconstruction must be deterministic");
        for &v in &a {
            assert!(v.is_finite());
        }
        assert_ne!(a, original, "GPTQ must perturb the buffer");
    }

    #[test]
    fn gptq_reconstruction_identity_on_zero_tensor() {
        let shape = [16, 8];
        let mut buf = vec![0.0_f32; 128];
        let h = vec![1.0_f32; 16];
        apply_gptq_reconstruction_inplace(&mut buf, &shape, &h, &GptqConfig::default()).unwrap();
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gptq_reconstruction_rejects_bad_inputs() {
        let h = vec![1.0_f32; 4];
        // Bad group size.
        assert_eq!(
            apply_gptq_reconstruction_inplace(
                &mut vec![0.0; 16],
                &[4, 4],
                &h,
                &GptqConfig { group_size: 0, damp_percent: 0.01 }
            )
            .unwrap_err(),
            GptqError::InvalidGroupSize
        );
        // Bad damp.
        assert_eq!(
            apply_gptq_reconstruction_inplace(
                &mut vec![0.0; 16],
                &[4, 4],
                &h,
                &GptqConfig { group_size: 4, damp_percent: 1.0 }
            )
            .unwrap_err(),
            GptqError::InvalidDampPercent
        );
        // Hessian length mismatch.
        assert!(matches!(
            apply_gptq_reconstruction_inplace(
                &mut vec![0.0; 16],
                &[4, 4],
                &vec![1.0; 3],
                &GptqConfig::default()
            )
            .unwrap_err(),
            GptqError::HessianLengthMismatch { .. }
        ));
    }

    #[test]
    fn gptq_evaluator_returns_metrics() {
        // AQS-2 integration: the evaluator drives GptqPolicy unchanged.
        let shape = [16, 16];
        let values = rand_weights(16, 16, 13);
        let act = vec![0.5_f32; 16];
        let input = TensorEvalInput {
            name: "w.gptq",
            values: &values,
            shape: &shape,
        };
        let r = evaluate_tensor_policy(
            input,
            &GptqPolicy::new(8, 0.01),
            &CalibrationContext::with_activations(&act),
        )
        .unwrap();
        assert_eq!(r.policy_id, "gptq");
        assert!(r.max_abs_diff > 0.0);
        assert!(r.rmse.is_finite());
        assert!(r.memory_bytes > 0);
    }

    /// Informational comparison only — not a hard assertion of GPTQ
    /// superiority. Run with `cargo test ... -- --ignored --nocapture`.
    #[test]
    #[ignore = "informational local-drift comparison; not a pass/fail gate"]
    fn awq_vs_gptq_local_drift() {
        let shape = [64, 64];
        let values = rand_weights(64, 64, 99);
        let act = rand_weights(1, 64, 7).iter().map(|v| v.abs()).collect::<Vec<_>>();
        let input = TensorEvalInput {
            name: "w",
            values: &values,
            shape: &shape,
        };
        let cal = CalibrationContext::with_activations(&act);
        let awq = evaluate_tensor_policy(input, &AwqPolicy::new(16, 0.3), &cal).unwrap();
        let gptq = evaluate_tensor_policy(input, &GptqPolicy::new(16, 0.01), &cal).unwrap();
        eprintln!(
            "AWQ  : max_abs_diff={:.6} mean={:.6} rmse={:.6}",
            awq.max_abs_diff, awq.mean_abs_diff, awq.rmse
        );
        eprintln!(
            "GPTQ : max_abs_diff={:.6} mean={:.6} rmse={:.6}",
            gptq.max_abs_diff, gptq.mean_abs_diff, gptq.rmse
        );
    }
}
