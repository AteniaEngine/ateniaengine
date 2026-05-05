//! M9.1 — INT8 weight quantiser.
//!
//! Produces the `(Vec<i8>, Vec<f32>)` pair the
//! [`crate::tensor::TensorStorage::CpuInt8`] variant carries. The
//! design is intentionally minimal: per-output-channel symmetric
//! absmax, no calibration, no zero-point, no fused dequant kernel
//! preview. The microbench in `examples/bench_int8_w8a16.rs`
//! validated this is sufficient for the M9 W8A16 path on Llama 2
//! 13B (H2 PASS, ~2× over Path B M8.4c on RTX 4070).
//!
//! # The math (one column at a time)
//!
//! For a weight matrix `W ∈ R^{K × N}` (row-major) and each output
//! column `n ∈ [0, N)`:
//!
//! ```text
//!     scale[n] = max(|W[:, n]|) / 127
//!     q[k, n]  = clamp(round(W[k, n] / scale[n]), -127, 127)   ∈ [−127, 127] ⊂ i8
//! ```
//!
//! Dequantisation reconstructs `W'[k, n] = (q[k, n] as f32) * scale[n]`.
//! On values that don't exceed BF16's 8-bit mantissa (powers of two,
//! small integers, etc.) the round-trip is bit-exact when the
//! divider scale is also exact; in general the per-element worst-
//! case error is `0.5 × scale[n] ≈ |W[:, n]|_max / 254`.
//!
//! # Edge cases
//!
//! - **All-zero column**: `max(|W[:, n]|) = 0` would produce
//!   `scale = 0`, which makes the dequantisation a useless zero
//!   regardless of `q`. We clamp scales from below by `1e-12` so
//!   the dequantisation is still numerically defined and the test
//!   suite can pin the contract. In practice an all-zero column is
//!   pathological for Llama-family models.
//! - **NaN / inf weights**: undefined behaviour. Production safetensors
//!   ingestion validates these upstream; the quantiser does not
//!   re-check.
//!
//! # Why per-channel and not per-tensor
//!
//! Per-tensor absmax shares one scale across the whole matrix; a
//! single outlier column then dominates the scale and crushes the
//! resolution of every other column. Per-channel (one scale per
//! output column) absorbs outliers locally — the standard W8A16
//! quantisation layout for Llama-family weights, validated in the
//! M9.0 microbench.

use crate::tensor::{Tensor, TensorStorage};

/// **M9.1** — per-output-channel symmetric absmax INT8 quantiser.
///
/// Operates on a row-major `[K, N]` F32 weight. For higher-rank
/// tensors the contract is "treat everything but the last axis as
/// rows": `K = product(shape[..-1])`, `N = shape[-1]`. The 2D path
/// is what the M9 loader will use; higher-rank fallthrough is
/// future-proofing.
///
/// Returns `(q, scales)` where:
///   - `q.len() == weights.len()` (same element count, one byte
///     per element instead of four)
///   - `scales.len() == shape[shape.len() - 1]`
///
/// # Panics
/// Panics on any length / shape mismatch:
///   - `shape.is_empty()`
///   - `weights.len() != product(shape)`
pub fn absmax_per_channel_symmetric(
    weights: &[f32],
    shape: &[usize],
) -> (Vec<i8>, Vec<f32>) {
    assert!(!shape.is_empty(),
        "absmax_per_channel_symmetric: shape must be non-empty");
    let total: usize = shape.iter().product();
    assert_eq!(
        weights.len(), total,
        "absmax_per_channel_symmetric: weights.len() = {} does not match product(shape) = {} (shape = {:?})",
        weights.len(), total, shape,
    );

    let n = *shape.last().unwrap();
    let k = if n == 0 { 0 } else { total / n };

    // Per-column absmax sweep. Reading row-major means stride-N
    // hops per (col, row) iteration; the column-major sweep order
    // here is intentional — touches every column's slot but with a
    // predictable stride that the optimiser can vectorise. K=0 or
    // N=0 produce empty outputs cleanly.
    let mut scales: Vec<f32> = vec![0.0; n];
    if k > 0 && n > 0 {
        for col in 0..n {
            let mut max_abs = 0.0_f32;
            for row in 0..k {
                let v = weights[row * n + col].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            scales[col] = (max_abs / 127.0).max(1e-12);
        }
    }

    let mut q: Vec<i8> = vec![0; total];
    if k > 0 && n > 0 {
        for row in 0..k {
            for col in 0..n {
                let s = scales[col];
                let qf = (weights[row * n + col] / s).round();
                let qi = qf.clamp(-127.0, 127.0) as i32;
                q[row * n + col] = qi as i8;
            }
        }
    }

    (q, scales)
}

/// Convenience wrapper that builds a fully-formed
/// [`TensorStorage::CpuInt8`]-backed [`Tensor`] from an F32 source.
///
/// Equivalent to:
///
/// ```ignore
/// let (q, scales) = absmax_per_channel_symmetric(weights, shape);
/// Tensor::new_cpu_int8(shape.to_vec(), q, scales)
/// ```
///
/// Used by the M9.2 loader path; tests can call either form.
pub fn quantize_int8_w8a16(weights: &[f32], shape: &[usize]) -> Tensor {
    let (q, scales) = absmax_per_channel_symmetric(weights, shape);
    Tensor::new_cpu_int8(shape.to_vec(), q, scales)
}

/// Borrow the `(q, scales, shape)` triple out of a
/// [`TensorStorage::CpuInt8`] tensor without consuming it.
///
/// Returns `None` for any other storage variant. Used by the M9
/// CUDA upload path and tests; production matmul callers go
/// through `int8_to_bf16_in_vram` instead.
pub fn as_cpu_int8_view(tensor: &Tensor) -> Option<(&[i8], &[f32], &[usize])> {
    match &tensor.storage {
        TensorStorage::CpuInt8 { q, scales, shape } => {
            Some((q.as_slice(), scales.as_slice(), shape.as_slice()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorStorage;

    /// Round-trip on values that fit cleanly in INT8 with the
    /// derived scale: pick weights so that `max(|w|) = 127 * s`
    /// for some `s`, i.e. every element is an integer multiple of
    /// `s`. The scale's clamp at `1e-12` doesn't kick in here
    /// because `max_abs > 0`. Round-trip should be bit-exact.
    #[test]
    fn round_trip_bit_exact_for_int_multiples_of_scale() {
        // Single-column case: K=4, N=1. Scale becomes
        // max(|[2, -4, 6, -127]|) / 127 = 1.0. Quantised values
        // should equal the original ints; dequant returns them
        // verbatim.
        let w = vec![2.0_f32, -4.0, 6.0, -127.0];
        let (q, scales) = absmax_per_channel_symmetric(&w, &[4, 1]);
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 1.0).abs() < 1e-9, "expected scale 1.0, got {}", scales[0]);
        assert_eq!(q, vec![2_i8, -4, 6, -127]);

        // Round-trip through TensorStorage::CpuInt8 → copy_to_cpu_vec.
        let t = Tensor::new_cpu_int8(vec![4, 1], q, scales);
        let dequant = t.copy_to_cpu_vec();
        assert_eq!(dequant, w, "bit-exact round-trip failed: {:?} vs {:?}", dequant, w);
    }

    /// The 1/127 envelope: for an arbitrary F32 weight bounded by
    /// `M = max(|W[:, n]|)` per column, the per-element absolute
    /// error of a round-trip is at most `0.5 × scale[n]` ≤
    /// `M / 254`. We use `M = 1.0` so the envelope is `< 1/127`
    /// (a slightly looser bound than `1/254` to absorb round-half-
    /// to-even tiebreakers).
    #[test]
    fn round_trip_error_bounded_by_one_over_127_for_unit_columns() {
        let k = 8usize;
        let n = 3usize;
        // Column 0: linear ramp in [-1, 1]. Column 1: alternating
        // small values. Column 2: a single dominant outlier so the
        // scale is large but most elements are tiny.
        let mut w = vec![0.0_f32; k * n];
        for row in 0..k {
            w[row * n + 0] = -1.0 + 2.0 * (row as f32) / ((k - 1) as f32);
            w[row * n + 1] = if row % 2 == 0 { 0.7 } else { -0.3 };
            w[row * n + 2] = if row == 3 { -1.0 } else { 0.0 };
        }
        let (q, scales) = absmax_per_channel_symmetric(&w, &[k, n]);
        assert_eq!(q.len(), k * n);
        assert_eq!(scales.len(), n);

        let t = Tensor::new_cpu_int8(vec![k, n], q, scales);
        let dequant = t.copy_to_cpu_vec();

        for row in 0..k {
            for col in 0..n {
                let orig = w[row * n + col];
                let back = dequant[row * n + col];
                let err = (orig - back).abs();
                // |W[:,col]|_max ≤ 1.0 in every column → envelope
                // ≤ 1/127 ≈ 7.87e-3. Allow a tiny absolute margin
                // for round-half ties.
                assert!(err < 1.0 / 127.0 + 1e-6,
                    "round-trip error {err:e} at ({row}, {col}) exceeds 1/127");
            }
        }
    }

    /// Scales vector length contract — must equal N (last axis).
    #[test]
    fn scales_length_equals_last_axis() {
        let (_q, scales) = absmax_per_channel_symmetric(
            &vec![0.5_f32; 5 * 7],
            &[5, 7],
        );
        assert_eq!(scales.len(), 7, "scales.len() must equal N = 7");

        let (_q, scales) = absmax_per_channel_symmetric(
            &vec![0.25_f32; 11 * 13],
            &[11, 13],
        );
        assert_eq!(scales.len(), 13);
    }

    /// All-zero column edge case: the scale is clamped to `1e-12`
    /// (not zero) so dequantisation is well-defined; the dequant
    /// values themselves are zero (q is zero, anything × 0 = 0).
    #[test]
    fn all_zero_column_produces_safe_scale_and_zero_dequant() {
        let w = vec![0.0_f32; 4 * 2];
        let (q, scales) = absmax_per_channel_symmetric(&w, &[4, 2]);
        for s in &scales {
            assert!(*s >= 1e-12, "scale must be clamped above zero, got {}", s);
        }
        assert!(q.iter().all(|&x| x == 0));

        let t = Tensor::new_cpu_int8(vec![4, 2], q, scales);
        let dequant = t.copy_to_cpu_vec();
        assert!(dequant.iter().all(|&x| x == 0.0));
    }

    /// `quantize_int8_w8a16` and `Tensor::new_cpu_int8` produce
    /// the same result; the convenience wrapper is just sugar.
    #[test]
    fn convenience_wrapper_matches_direct_construction() {
        let w = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        let shape = vec![3, 2];

        let direct = {
            let (q, scales) = absmax_per_channel_symmetric(&w, &shape);
            Tensor::new_cpu_int8(shape.clone(), q, scales)
        };
        let via_wrapper = quantize_int8_w8a16(&w, &shape);

        assert_eq!(direct.shape, via_wrapper.shape);
        assert_eq!(direct.copy_to_cpu_vec(), via_wrapper.copy_to_cpu_vec());

        let (q_d, s_d, _) = as_cpu_int8_view(&direct).unwrap();
        let (q_w, s_w, _) = as_cpu_int8_view(&via_wrapper).unwrap();
        assert_eq!(q_d, q_w);
        assert_eq!(s_d, s_w);
    }

    /// `as_cpu_int8_view` returns `None` on any non-CpuInt8 tensor.
    #[test]
    fn cpu_int8_view_only_on_cpu_int8_storage() {
        let cpu = Tensor::new_cpu(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(as_cpu_int8_view(&cpu).is_none());

        let bf16 = Tensor::new_cpu_bf16(vec![2, 2], vec![0u16; 4]);
        assert!(as_cpu_int8_view(&bf16).is_none());

        let int8 = quantize_int8_w8a16(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let view = as_cpu_int8_view(&int8);
        assert!(view.is_some());
        let (q, scales, shape) = view.unwrap();
        assert_eq!(q.len(), 4);
        assert_eq!(scales.len(), 2);
        assert_eq!(shape, &[2, 2]);
    }

    /// `Tensor::new_cpu_int8` rejects bad shapes via assertion. We
    /// catch via `std::panic::catch_unwind` so the test still
    /// exercises the contract without bringing the suite down.
    #[test]
    fn constructor_rejects_mismatched_shapes() {
        // q.len() vs product(shape).
        let r = std::panic::catch_unwind(|| {
            Tensor::new_cpu_int8(vec![3, 2], vec![0_i8; 5], vec![0.0_f32; 2])
        });
        assert!(r.is_err(), "expected panic on q.len() mismatch");

        // scales.len() vs shape.last().
        let r = std::panic::catch_unwind(|| {
            Tensor::new_cpu_int8(vec![3, 2], vec![0_i8; 6], vec![0.0_f32; 3])
        });
        assert!(r.is_err(), "expected panic on scales.len() mismatch");
    }

    /// The CpuInt8 storage variant is what new_cpu_int8 produces;
    /// confirm the variant tag and the inner shape is mirrored.
    #[test]
    fn cpu_int8_storage_carries_inner_shape() {
        let t = quantize_int8_w8a16(&vec![0.5_f32; 6], &[2, 3]);
        match &t.storage {
            TensorStorage::CpuInt8 { q, scales, shape } => {
                assert_eq!(q.len(), 6);
                assert_eq!(scales.len(), 3);
                assert_eq!(shape, &vec![2usize, 3]);
            }
            other => panic!("expected CpuInt8 storage, got {:?}", other),
        }
        assert_eq!(t.dtype, crate::tensor::DType::Int8);
    }
}
