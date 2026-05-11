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
pub fn compute_inv_freqs(head_dim: usize, base_freq: u32) -> Vec<f32> {
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
pub fn compute_inv_freqs_llama3(
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

/// **M11.B** — Microsoft Phi-3 / Phi-3.5 LongRope inverse-frequency
/// schedule.
///
/// Implements the algorithm at
/// `huggingface/transformers::modeling_rope_utils::_compute_longrope_parameters`:
/// pick `short_factor` or `long_factor` per the observed sequence
/// length, then divide the base inverse frequencies element-wise.
///
/// ```text
///   factor[i] = if seq_len > original_max_pos { long_factor[i] }
///               else                          { short_factor[i] }
///   inv_freq[i] = 1 / (factor[i] * base^(2i / head_dim))
/// ```
///
/// Both `short_factor` and `long_factor` must have length
/// `head_dim / 2` — this matches the `inv_freq` indexing used by
/// the half-split RoPE kernel (`apply_rope_with_inv_freqs`).
///
/// All math is computed in `f64` and cast to `f32` at the end. With
/// Phi-3.5's `base = 10_000`, the intermediate `base^(2i/d)` term
/// stays well within `f32` precision, but routing through `f64`
/// matches the reference implementation and keeps the numerics
/// faithful for reproducibility.
///
/// # Panics
/// - `short_factor.len() != head_dim / 2`
/// - `long_factor.len() != head_dim / 2`
pub fn compute_inv_freqs_longrope(
    head_dim: usize,
    base_freq: u32,
    short_factor: &[f32],
    long_factor: &[f32],
    original_max_pos: u32,
    seq_len: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    assert_eq!(
        short_factor.len(),
        half,
        "compute_inv_freqs_longrope: short_factor length {} != head_dim/2 ({})",
        short_factor.len(),
        half
    );
    assert_eq!(
        long_factor.len(),
        half,
        "compute_inv_freqs_longrope: long_factor length {} != head_dim/2 ({})",
        long_factor.len(),
        half
    );

    let factor: &[f32] = if (seq_len as u32) > original_max_pos {
        long_factor
    } else {
        short_factor
    };
    let base = base_freq as f64;

    (0..half)
        .map(|i| {
            let exp = (i as f64) * 2.0 / (head_dim as f64);
            let inv_freq_base = 1.0_f64 / base.powf(exp);
            (inv_freq_base / factor[i] as f64) as f32
        })
        .collect()
}

/// **M11.B** — `attention_factor` derivation for the LongRope
/// scaling. Multiplies cos/sin by this scalar when the scaling
/// is active to compensate the wider context window:
///
/// ```text
///   scale = max_position_embeddings / original_max_position_embeddings
///   if scale > 1.0:
///       attention_factor = sqrt(1 + ln(scale) / ln(original_max_position_embeddings))
///   else:
///       attention_factor = 1.0
/// ```
///
/// The function expects integer-valued context lengths and
/// returns the scalar in `f32`. Computed in `f64` and cast at
/// the end for the same reason as
/// `compute_inv_freqs_longrope`.
pub fn compute_attention_factor_longrope(
    original_max_position_embeddings: u32,
    max_position_embeddings: u32,
) -> f32 {
    let original = original_max_position_embeddings as f64;
    let configured = max_position_embeddings as f64;
    if configured <= original || original <= 1.0 {
        return 1.0;
    }
    let scale = configured / original;
    let attention_factor = (1.0 + scale.ln() / original.ln()).sqrt();
    attention_factor as f32
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
    let inv_freqs = compute_inv_freqs(head_dim, base_freq);
    apply_rope_with_inv_freqs(x, head_dim, &inv_freqs)
}

/// Apply RoPE using a caller-provided inverse-frequency vector.
///
/// Used by Llama 3.x checkpoints whose `rope_scaling` field reshapes
/// the frequency schedule (see [`compute_inv_freqs_llama3`]). For
/// plain RoPE, prefer [`apply_rope`] which encapsulates the standard
/// schedule.
///
/// `inv_freqs.len()` must equal `head_dim / 2`.
/// **M5.c.2.c** — RoPE at a non-zero starting position.
///
/// Identical to [`apply_rope_with_inv_freqs`] but rotates each
/// sequence position `s` against angle `(s + position_offset)
/// * inv_freqs[i]` instead of `s * inv_freqs[i]`. With
/// `position_offset = 0` the function is bit-exact equivalent
/// to [`apply_rope_with_inv_freqs`] — verified by the
/// `rope_offset_zero_matches_no_offset` test.
///
/// Used by decode-step graphs where Q at seq=1 must rotate at
/// the absolute conversation position `cached_len`, not 0.
pub fn apply_rope_with_offset_inv_freqs(
    x: &Tensor,
    head_dim: usize,
    inv_freqs: &[f32],
    position_offset: u32,
) -> Tensor {
    let shape = &x.shape;
    validate_shape(shape, head_dim, "apply_rope_with_offset_inv_freqs");
    assert_eq!(
        inv_freqs.len(),
        head_dim / 2,
        "apply_rope_with_offset_inv_freqs: inv_freqs length {} != head_dim/2 ({})",
        inv_freqs.len(),
        head_dim / 2
    );

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;
    let offset = position_offset as f32;

    let x_slice = x.as_cpu_slice();
    let mut output = vec![0.0_f32; x_slice.len()];

    for b in 0..batch {
        for s in 0..seq_len {
            // M5.c.2.c — absolute position is `s + offset`.
            // offset=0 collapses to `s as f32`, recovering the
            // pre-M5 numeric sequence exactly.
            let abs_pos = s as f32 + offset;
            for h in 0..n_heads {
                let base_offset =
                    b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim;
                for i in 0..half {
                    let angle = abs_pos * inv_freqs[i];
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

pub fn apply_rope_with_inv_freqs(x: &Tensor, head_dim: usize, inv_freqs: &[f32]) -> Tensor {
    let shape = &x.shape;
    validate_shape(shape, head_dim, "apply_rope_with_inv_freqs");
    assert_eq!(
        inv_freqs.len(),
        head_dim / 2,
        "apply_rope_with_inv_freqs: inv_freqs length {} != head_dim/2 ({})",
        inv_freqs.len(),
        head_dim / 2
    );

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;

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
pub fn apply_rope_backward(
    out_grad: &[f32],
    shape: &[usize],
    head_dim: usize,
    base_freq: u32,
) -> Vec<f32> {
    let inv_freqs = compute_inv_freqs(head_dim, base_freq);
    apply_rope_backward_with_inv_freqs(out_grad, shape, head_dim, &inv_freqs)
}

/// Backward pass for RoPE using a caller-provided inverse-frequency
/// vector. Mirrors [`apply_rope_with_inv_freqs`] for the gradient.
pub fn apply_rope_backward_with_inv_freqs(
    out_grad: &[f32],
    shape: &[usize],
    head_dim: usize,
    inv_freqs: &[f32],
) -> Vec<f32> {
    validate_shape(shape, head_dim, "apply_rope_backward_with_inv_freqs");
    let expected: usize = shape.iter().product();
    assert_eq!(
        out_grad.len(),
        expected,
        "apply_rope_backward_with_inv_freqs: out_grad length {} does not match shape product {} (shape = {:?})",
        out_grad.len(),
        expected,
        shape
    );
    assert_eq!(
        inv_freqs.len(),
        head_dim / 2,
        "apply_rope_backward_with_inv_freqs: inv_freqs length {} != head_dim/2 ({})",
        inv_freqs.len(),
        head_dim / 2
    );

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;

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

/// **M11.B** — LongRope scaling tests.
///
/// Validates the per-dimension factor selection (short vs long
/// path keyed by `seq_len`) and the bit-exact identity case
/// (factor = [1.0, ...] reproduces unscaled inv_freq).
#[cfg(test)]
mod longrope_scaling_tests {
    use super::{compute_attention_factor_longrope, compute_inv_freqs, compute_inv_freqs_longrope};

    /// Phi-3.5 Mini parameters: head_dim = 96, theta = 10000,
    /// original_max_pos = 4096, max_pos = 131072.
    const HEAD_DIM: usize = 96;
    const BASE: u32 = 10_000;
    const ORIGINAL_MAX_POS: u32 = 4096;

    fn ones_factor() -> Vec<f32> {
        vec![1.0_f32; HEAD_DIM / 2]
    }

    /// `seq_len <= original_max_pos` selects `short_factor`.
    /// With `short_factor = [1.0, ...]` the result must equal
    /// the unscaled `compute_inv_freqs`.
    #[test]
    fn longrope_short_path_with_unit_factor_matches_unscaled() {
        let short = ones_factor();
        let long: Vec<f32> = (0..HEAD_DIM / 2).map(|i| 2.0 + i as f32).collect();
        let scaled =
            compute_inv_freqs_longrope(HEAD_DIM, BASE, &short, &long, ORIGINAL_MAX_POS, 1024);
        let unscaled = compute_inv_freqs(HEAD_DIM, BASE);
        assert_eq!(scaled.len(), unscaled.len());
        for (i, (s, u)) in scaled.iter().zip(unscaled.iter()).enumerate() {
            let diff = (s - u).abs();
            assert!(
                diff < 1e-7,
                "short-path with unit factor must match unscaled at i={}: \
                 got {:.6e}, expected {:.6e}, diff={:.3e}",
                i,
                s,
                u,
                diff
            );
        }
    }

    /// `seq_len > original_max_pos` selects `long_factor`. With
    /// `long_factor[i] = 2.0` for all i, the result must equal
    /// the unscaled `compute_inv_freqs` divided by 2.
    #[test]
    fn longrope_long_path_divides_by_factor() {
        let short = ones_factor();
        let long: Vec<f32> = vec![2.0_f32; HEAD_DIM / 2];
        let scaled = compute_inv_freqs_longrope(
            HEAD_DIM,
            BASE,
            &short,
            &long,
            ORIGINAL_MAX_POS,
            (ORIGINAL_MAX_POS + 1) as usize,
        );
        let unscaled = compute_inv_freqs(HEAD_DIM, BASE);
        for (i, (s, u)) in scaled.iter().zip(unscaled.iter()).enumerate() {
            let expected = u / 2.0;
            let diff = (s - expected).abs();
            assert!(
                diff < 1e-7,
                "long-path with factor=2 must equal unscaled / 2 at i={}: \
                 got {:.6e}, expected {:.6e}, diff={:.3e}",
                i,
                s,
                expected,
                diff
            );
        }
    }

    /// Boundary: `seq_len == original_max_pos` is the upper edge
    /// of the short range. The function uses strict `>` for the
    /// long-factor switch, so equality stays on the short path.
    #[test]
    fn longrope_boundary_uses_short_factor() {
        let short: Vec<f32> = vec![3.0_f32; HEAD_DIM / 2];
        let long: Vec<f32> = vec![5.0_f32; HEAD_DIM / 2];
        let scaled_at_boundary = compute_inv_freqs_longrope(
            HEAD_DIM,
            BASE,
            &short,
            &long,
            ORIGINAL_MAX_POS,
            ORIGINAL_MAX_POS as usize,
        );
        let unscaled = compute_inv_freqs(HEAD_DIM, BASE);
        // boundary case takes short_factor = 3.0
        for (i, (s, u)) in scaled_at_boundary.iter().zip(unscaled.iter()).enumerate() {
            let expected = u / 3.0;
            let diff = (s - expected).abs();
            assert!(
                diff < 1e-7,
                "boundary at i={i}: got {s} expected {expected}"
            );
        }
    }

    /// `compute_attention_factor_longrope` returns 1.0 when the
    /// configured context window does not exceed the pre-training
    /// window — there is no extension to compensate for.
    #[test]
    fn longrope_attention_factor_is_one_when_no_extension() {
        let af = compute_attention_factor_longrope(4096, 4096);
        assert_eq!(af, 1.0);
        let af_smaller = compute_attention_factor_longrope(8192, 4096);
        assert_eq!(af_smaller, 1.0);
    }

    /// Phi-3.5 Mini case: original=4096, max=131072 → scale=32.
    /// Reference value computed in pure f64:
    ///   sqrt(1 + ln(32) / ln(4096)) ≈ 1.190.
    #[test]
    fn longrope_attention_factor_phi35_mini() {
        let af = compute_attention_factor_longrope(4096, 131072);
        let scale = 131072.0_f64 / 4096.0_f64;
        let expected = (1.0_f64 + scale.ln() / (4096.0_f64).ln()).sqrt() as f32;
        let diff = (af - expected).abs();
        assert!(
            diff < 1e-6,
            "attention_factor mismatch: got {af}, expected {expected}, diff={diff:.3e}"
        );
        // Sanity: must be > 1 (we are extending the context).
        assert!(af > 1.0);
        assert!(af < 1.5, "Phi-3.5 attention_factor should sit around 1.19");
    }
}

/// **M5.c.2.c** — position-offset RoPE tests.
///
/// The headline guarantee for M5.c.2.c is that
/// `apply_rope_with_offset_inv_freqs(..., offset = 0)` is
/// bit-exact equivalent to `apply_rope_with_inv_freqs(...)`.
/// Without this, every M4.6 / M4.7 / M4.8 forward fixture
/// would shift slightly when the new kernel landed.
#[cfg(test)]
mod offset_tests {
    use super::*;
    use crate::tensor::Tensor;

    fn arange_tensor(shape: Vec<usize>, start: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|i| start + i as f32 * 0.01).collect();
        Tensor::new_cpu(shape, data)
    }

    #[test]
    fn rope_offset_zero_matches_no_offset() {
        // Bit-exact contract: offset=0 must produce the same
        // bytes as the offset-less kernel. Locks the invariant
        // that landing this kernel doesn't shift any existing
        // forward fixture.
        let head_dim = 64;
        let base_freq: u32 = 10_000;
        let inv_freqs = compute_inv_freqs(head_dim, base_freq);

        // Shape: [batch=2, seq=8, n_heads=4, head_dim=64]
        // — covers a non-trivial cross-section of the
        // multi-head reshape pattern the builder emits.
        let x = arange_tensor(vec![2, 8, 4, 64], 0.5);

        let baseline = apply_rope_with_inv_freqs(&x, head_dim, &inv_freqs);
        let offset_zero = apply_rope_with_offset_inv_freqs(&x, head_dim, &inv_freqs, 0);

        assert_eq!(baseline.shape, offset_zero.shape);
        let a = baseline.copy_to_cpu_vec();
        let b = offset_zero.copy_to_cpu_vec();
        assert_eq!(a.len(), b.len());
        // Bit-exact (same arithmetic on the same operands).
        for (i, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                lhs.to_bits(),
                rhs.to_bits(),
                "offset=0 must match no-offset bit-exactly at index {i}: \
                 {lhs:.10} vs {rhs:.10}"
            );
        }
    }

    #[test]
    fn rope_offset_n_matches_seq_starting_at_n() {
        // Conceptual contract: rotating a seq=1 tensor at
        // offset=N must produce the same numerics as the row
        // at position N of a seq=N+1 tensor rotated at offset
        // 0. This is the property the M5.c.2.c decode-step
        // attention path leans on — Q at the new token,
        // rotated at cached_len, must match the rotation it
        // would have received as the (cached_len)th row of a
        // full prefill.
        let head_dim = 32;
        let base_freq: u32 = 10_000;
        let inv_freqs = compute_inv_freqs(head_dim, base_freq);

        // Build a prefill-style tensor at seq=4. Row index 3
        // is the "current decode token" we'd be processing in
        // a hypothetical decode at cached_len=3.
        let seq = 4usize;
        let n_heads = 2usize;
        let prefill = arange_tensor(vec![1, seq, n_heads, head_dim], 1.0);
        let prefill_rot = apply_rope_with_inv_freqs(&prefill, head_dim, &inv_freqs);

        // Slice out the row at position 3 from `prefill`'s
        // ORIGINAL data and treat it as a seq=1 tensor.
        let prefill_data = prefill.copy_to_cpu_vec();
        let row_stride = n_heads * head_dim;
        let row_off = 3 * row_stride;
        let row_data: Vec<f32> = prefill_data[row_off..row_off + row_stride].to_vec();
        let single_row = Tensor::new_cpu(vec![1, 1, n_heads, head_dim], row_data);

        // Rotate it at offset=3 — should match the row 3 of
        // the prefill rotation.
        let single_rot = apply_rope_with_offset_inv_freqs(&single_row, head_dim, &inv_freqs, 3);

        // Compare against `prefill_rot[..., row=3, :, :]`.
        let prefill_rot_data = prefill_rot.copy_to_cpu_vec();
        let row3_rot = &prefill_rot_data[row_off..row_off + row_stride];
        let single_rot_data = single_rot.copy_to_cpu_vec();
        assert_eq!(single_rot_data.len(), row3_rot.len());
        for (i, (lhs, rhs)) in single_rot_data.iter().zip(row3_rot.iter()).enumerate() {
            // Floating-point equality with a tiny tolerance —
            // the two paths perform the same arithmetic in the
            // same order (single nested loop, same inv_freqs),
            // so they should be byte-identical, but we allow
            // 1 ULP slack defensively.
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "decode-step rotation at offset=3 must match \
                 prefill row 3 at index {i}: {lhs} vs {rhs}"
            );
        }
    }
}
