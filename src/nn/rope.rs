//! Rotary Positional Embedding (RoPE) — half-split layout (HuggingFace
//! convention).
//!
//! See [`crate::amg::nodes::NodeType::RoPE`] for the architectural
//! contract (layout, positions, parameters).

use crate::tensor::Tensor;

/// Compute inverse frequencies `theta_i = base_freq^(-2i/head_dim)` for
/// `i in 0..head_dim/2`. Shared by [`apply_rope`] and
/// [`apply_rope_backward`] so that any change to the schedule applies
/// to both directions.
pub(crate) fn compute_inv_freqs(head_dim: usize, base_freq: u32) -> Vec<f32> {
    let half = head_dim / 2;
    let base_freq_f32 = base_freq as f32;
    (0..half)
        .map(|i| {
            let exp = (i as f32) * 2.0 / (head_dim as f32);
            1.0_f32 / base_freq_f32.powf(exp)
        })
        .collect()
}

/// Llama 3 piecewise inverse-frequency scaling.
///
/// Implements the algorithm at
/// `huggingface/transformers::modeling_rope_utils::_compute_llama3_parameters`
/// (lines 728–804). Used by Llama 3.1, 3.2, and 3.3 to extend the
/// effective context window from the pre-training length
/// (`original_max_pos`) up to the configured `max_position_embeddings`
/// without re-training: high-frequency dims stay unchanged, low-frequency
/// dims are divided by `factor`, and a smooth linear interpolation
/// blends the two bands in the middle.
///
/// All math is computed in `f64` and cast to `f32` at the end. With
/// Llama 3.2's `base_freq = 500_000`, the intermediate `b^(2i/d)` term
/// can lose meaningful precision in pure `f32` for indices near
/// `head_dim/2`; routing through `f64` keeps the inverse frequencies
/// faithful to the reference implementation. Legacy
/// [`compute_inv_freqs`] retains its `f32`-only path so non-Llama-3
/// checkpoints (TinyLlama, SmolLM2, Qwen) remain bit-identical.
///
/// Reference (HF, simplified):
/// ```text
///   inv_freq_base[i]  = 1 / base_freq^(2i / head_dim)
///   wavelen[i]        = 2π / inv_freq_base[i]
///   low_wavelen       = original_max_pos / low_freq_factor
///   high_wavelen      = original_max_pos / high_freq_factor
///
///   if   wavelen[i] < high_wavelen : inv_freq[i] = inv_freq_base[i]            // high band
///   elif wavelen[i] > low_wavelen  : inv_freq[i] = inv_freq_base[i] / factor    // low band
///   else                            : s = (original_max_pos / wavelen[i] - low_freq_factor)
///                                         / (high_freq_factor - low_freq_factor)
///                                     inv_freq[i] = (1 - s) * inv_freq_base[i] / factor
///                                                  + s * inv_freq_base[i]      // smooth
/// ```
pub(crate) fn compute_inv_freqs_llama3(
    head_dim: usize,
    base_freq: u32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_pos: u32,
) -> Vec<f32> {
    let half = head_dim / 2;
    let base = base_freq as f64;
    let f = factor as f64;
    let l = low_freq_factor as f64;
    let h = high_freq_factor as f64;
    let m = original_max_pos as f64;
    let two_pi = 2.0_f64 * std::f64::consts::PI;
    let low_wavelen = m / l;
    let high_wavelen = m / h;
    let band_span = h - l;

    (0..half)
        .map(|i| {
            let exp = (i as f64) * 2.0 / (head_dim as f64);
            let inv_freq_base = 1.0_f64 / base.powf(exp);
            let wavelen = two_pi / inv_freq_base;

            let scaled = if wavelen < high_wavelen {
                // High-frequency band: keep as-is.
                inv_freq_base
            } else if wavelen > low_wavelen {
                // Low-frequency band: divide by factor.
                inv_freq_base / f
            } else {
                // Medium band: smooth linear interp between the two
                // extremes, keyed by `original_max_pos / wavelen`.
                let s = (m / wavelen - l) / band_span;
                (1.0 - s) * (inv_freq_base / f) + s * inv_freq_base
            };
            scaled as f32
        })
        .collect()
}

fn validate_shape(shape: &[usize], head_dim: usize, op: &str) {
    assert_eq!(
        shape.len(),
        4,
        "{}: expects 4D tensor [batch, seq_len, n_heads, head_dim], got {:?}",
        op,
        shape
    );
    assert_eq!(
        shape[3], head_dim,
        "{}: tensor last dim ({}) does not match head_dim ({})",
        op, shape[3], head_dim
    );
    assert!(head_dim > 0, "{}: head_dim must be positive", op);
    assert!(
        head_dim % 2 == 0,
        "{}: head_dim must be even, got {}",
        op,
        head_dim
    );
}

/// Apply RoPE (half-split layout) to input tensor.
///
/// Input shape:  `[batch, seq_len, n_heads, head_dim]`.
/// Output shape: same as input.
/// Positions:    implicit `[0..seq_len)`.
///
/// Half-split formula (per `(b, s, h)` slice, for `i in 0..head_dim/2`):
/// ```text
///   angle    = s * base_freq^(-2i/head_dim)
///   y[i]            =  x[i]      * cos(angle) - x[i + half] * sin(angle)
///   y[i + half]     =  x[i + half]* cos(angle) + x[i]       * sin(angle)
/// ```
///
/// Frequencies are computed on the fly. For typical TinyLlama-class
/// shapes the cost is ~1.3 ms per forward pass, which is negligible
/// relative to the matmul-dominated workload around it.
///
/// # Panics
/// See [`validate_shape`].
pub fn apply_rope(x: &Tensor, head_dim: usize, base_freq: u32) -> Tensor {
    let shape = &x.shape;
    validate_shape(shape, head_dim, "apply_rope");

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;
    let inv_freqs = compute_inv_freqs(head_dim, base_freq);

    let x_slice = x.as_cpu_slice();
    let mut output = vec![0.0_f32; x_slice.len()];

    for b in 0..batch {
        for s in 0..seq_len {
            let s_f32 = s as f32;
            for h in 0..n_heads {
                let base_offset =
                    b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim;
                for i in 0..half {
                    let angle = s_f32 * inv_freqs[i];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let x_first = x_slice[base_offset + i];
                    let x_second = x_slice[base_offset + i + half];

                    output[base_offset + i] = x_first * cos_a - x_second * sin_a;
                    output[base_offset + i + half] = x_second * cos_a + x_first * sin_a;
                }
            }
        }
    }

    Tensor::new_cpu(shape.clone(), output)
}

/// Backward pass for RoPE.
///
/// Given the upstream gradient `out_grad` (same shape as the forward
/// output) and the forward configuration, returns `grad_x` (same shape
/// as the input). RoPE has no learnable parameters, so this is the
/// only gradient produced.
///
/// Derivation: forward applies a 2D rotation by `+angle` on each
/// `(i, i+half)` pair. The Jacobian of a rotation matrix is its
/// transpose, i.e. a rotation by `-angle`. Therefore:
/// ```text
///   grad_x[i]      =  out_grad[i]      * cos(angle) + out_grad[i+half] * sin(angle)
///   grad_x[i+half] = -out_grad[i]      * sin(angle) + out_grad[i+half] * cos(angle)
/// ```
/// `head_dim` and `base_freq` are model constants and have no gradient.
///
/// # Panics
/// Panics if `out_grad`'s shape does not satisfy the same constraints
/// as the forward input.
pub fn apply_rope_backward(out_grad: &[f32], shape: &[usize], head_dim: usize, base_freq: u32) -> Vec<f32> {
    validate_shape(shape, head_dim, "apply_rope_backward");
    let expected: usize = shape.iter().product();
    assert_eq!(
        out_grad.len(),
        expected,
        "apply_rope_backward: out_grad length {} does not match shape product {} (shape = {:?})",
        out_grad.len(),
        expected,
        shape
    );

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;
    let inv_freqs = compute_inv_freqs(head_dim, base_freq);

    let mut grad_x = vec![0.0_f32; out_grad.len()];

    for b in 0..batch {
        for s in 0..seq_len {
            let s_f32 = s as f32;
            for h in 0..n_heads {
                let base_offset =
                    b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim;
                for i in 0..half {
                    let angle = s_f32 * inv_freqs[i];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let g_first = out_grad[base_offset + i];
                    let g_second = out_grad[base_offset + i + half];

                    // Rotate grad by -angle (transpose of forward rotation).
                    grad_x[base_offset + i] = g_first * cos_a + g_second * sin_a;
                    grad_x[base_offset + i + half] = -g_first * sin_a + g_second * cos_a;
                }
            }
        }
    }

    grad_x
}

#[cfg(test)]
mod llama3_scaling_tests {
    use super::compute_inv_freqs_llama3;

    /// Llama 3.2 1B parameters: head_dim=64, theta=500_000, factor=32,
    /// low_freq_factor=1, high_freq_factor=4, original_max_pos=8192.
    const HEAD_DIM: usize = 64;
    const BASE: u32 = 500_000;
    const FACTOR: f32 = 32.0;
    const LOW: f32 = 1.0;
    const HIGH: f32 = 4.0;
    const M: u32 = 8192;

    fn inv_freq_base_f64(i: usize) -> f64 {
        let exp = (i as f64) * 2.0 / (HEAD_DIM as f64);
        1.0_f64 / (BASE as f64).powf(exp)
    }

    /// i=0 → wavelen = 2π ≈ 6.28 ≪ high_wavelen = 8192/4 = 2048
    /// → high-frequency band → identity.
    #[test]
    fn llama3_high_frequency_band_is_identity_at_i0() {
        let scaled = compute_inv_freqs_llama3(HEAD_DIM, BASE, FACTOR, LOW, HIGH, M);
        // Reference: plain inv_freq[0] = 1/base^0 = 1.0 (identity).
        assert!((scaled[0] - 1.0_f32).abs() < 1e-7);
    }

    /// i=31 (last index) → wavelen = 2π · 500_000^(31/32) ≈ 2.08e6
    /// ≫ low_wavelen = 8192 → low-frequency band → divide by factor=32.
    #[test]
    fn llama3_low_frequency_band_divides_by_factor_at_i31() {
        let scaled = compute_inv_freqs_llama3(HEAD_DIM, BASE, FACTOR, LOW, HIGH, M);
        let expected = (inv_freq_base_f64(31) / FACTOR as f64) as f32;
        let diff = (scaled[31] - expected).abs();
        assert!(
            diff < 1e-9,
            "low-band scaling mismatch: got {:.10e}, expected {:.10e}, diff={:.3e}",
            scaled[31],
            expected,
            diff
        );
    }

    /// i=16 → wavelen = 2π · 500_000^(0.5) ≈ 4443
    /// → 2048 < 4443 < 8192, mid-band → smooth interp.
    #[test]
    fn llama3_mid_band_uses_smooth_interp_at_i16() {
        let scaled = compute_inv_freqs_llama3(HEAD_DIM, BASE, FACTOR, LOW, HIGH, M);

        // Reproduce the smooth interp in F64 for a hand-checkable
        // reference, mirroring the production code.
        let inv_base = inv_freq_base_f64(16);
        let two_pi = 2.0_f64 * std::f64::consts::PI;
        let wavelen = two_pi / inv_base;
        let low_wavelen = M as f64 / LOW as f64;
        let high_wavelen = M as f64 / HIGH as f64;
        // Sanity: i=16 must indeed sit in the mid band.
        assert!(
            wavelen >= high_wavelen && wavelen <= low_wavelen,
            "i=16 wavelen {:.4} must lie in [{:.4}, {:.4}]",
            wavelen,
            high_wavelen,
            low_wavelen
        );
        let s = (M as f64 / wavelen - LOW as f64) / (HIGH as f64 - LOW as f64);
        let expected = ((1.0 - s) * (inv_base / FACTOR as f64) + s * inv_base) as f32;
        let diff = (scaled[16] - expected).abs();
        assert!(
            diff < 1e-9,
            "mid-band scaling mismatch at i=16: got {:.10e}, expected {:.10e}, diff={:.3e}",
            scaled[16],
            expected,
            diff
        );

        // Sanity: the smooth-interp result must lie strictly between
        // the two band extremes (unscaled and divided-by-factor).
        let unscaled = inv_base as f32;
        let divided = (inv_base / FACTOR as f64) as f32;
        assert!(
            scaled[16] > divided && scaled[16] < unscaled,
            "smooth interp must produce a value between divided ({:.3e}) and unscaled ({:.3e}); got {:.3e}",
            divided,
            unscaled,
            scaled[16]
        );
    }

    /// Boundary check: with factor=1 every band collapses to identity,
    /// so the scaled vector must equal the unscaled inv_freq.
    #[test]
    fn llama3_factor_one_is_identity_for_all_indices() {
        let scaled = compute_inv_freqs_llama3(HEAD_DIM, BASE, 1.0, LOW, HIGH, M);
        for i in 0..HEAD_DIM / 2 {
            let expected = inv_freq_base_f64(i) as f32;
            let diff = (scaled[i] - expected).abs();
            assert!(
                diff < 1e-7,
                "factor=1 must be identity at i={}: got {:.6e}, expected {:.6e}",
                i,
                scaled[i],
                expected
            );
        }
    }
}
