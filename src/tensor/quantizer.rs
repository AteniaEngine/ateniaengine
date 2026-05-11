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
pub fn absmax_per_channel_symmetric(weights: &[f32], shape: &[usize]) -> (Vec<i8>, Vec<f32>) {
    assert!(
        !shape.is_empty(),
        "absmax_per_channel_symmetric: shape must be non-empty"
    );
    let total: usize = shape.iter().product();
    assert_eq!(
        weights.len(),
        total,
        "absmax_per_channel_symmetric: weights.len() = {} does not match product(shape) = {} (shape = {:?})",
        weights.len(),
        total,
        shape,
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
/// Used by tests; the M9.2/M9.4 loader path goes through
/// [`absmax_per_group_symmetric`] + the per-group CUDA upload.
pub fn quantize_int8_w8a16(weights: &[f32], shape: &[usize]) -> Tensor {
    let (q, scales) = absmax_per_channel_symmetric(weights, shape);
    Tensor::new_cpu_int8(shape.to_vec(), q, scales)
}

/// **M9.4** — per-group symmetric absmax INT8 quantiser.
///
/// Same row-major `[K, N]` semantics as
/// [`absmax_per_channel_symmetric`], but partitions the K axis
/// into groups of `group_size` consecutive elements. Each
/// `(group, column)` pair owns one F32 scale. For a weight `W`
/// and an element at `(k, n)`:
///
/// ```text
///     g = k / group_size
///     scale[g, n] = max(|W[g·G : (g+1)·G, n]|) / 127
///     q[k, n] = clamp(round(W[k, n] / scale[g, n]), -127, 127)
/// ```
///
/// The dequant round-trip is the standard
/// `(q[k, n] as f32) * scale[g, n]`.
///
/// # Why per-group beats per-channel for Llama
///
/// The M9.4 4-model F64 fixture under per-channel quantisation
/// produced drift 1.10 to 8.88 vs the F64 reference, all
/// breaching the ADR-004 `< 0.5` gate. Root cause: per-column
/// outliers dominate `max(|W[:, n]|)` and crush the resolution
/// for the rest of the column. Per-group localises the outlier's
/// influence to its own block (`group_size` elements), recovering
/// margin for the other `K - group_size` elements in the same
/// column.
///
/// # Returns
/// `(q, scales)` where:
///   - `q.len() == weights.len()`
///   - `scales.len() == ceil(K / group_size) * N`
///
/// `K` is `product(shape[..-1])`; `N` is `shape[-1]`.
///
/// # Panics
///   - `shape.is_empty()`
///   - `weights.len() != product(shape)`
///   - `group_size == 0`
pub fn absmax_per_group_symmetric(
    weights: &[f32],
    shape: &[usize],
    group_size: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert!(
        !shape.is_empty(),
        "absmax_per_group_symmetric: shape must be non-empty"
    );
    assert!(
        group_size > 0,
        "absmax_per_group_symmetric: group_size must be > 0"
    );
    let total: usize = shape.iter().product();
    assert_eq!(
        weights.len(),
        total,
        "absmax_per_group_symmetric: weights.len() = {} does not match product(shape) = {} (shape = {:?})",
        weights.len(),
        total,
        shape,
    );

    let n = *shape.last().unwrap();
    let k: usize = if shape.len() <= 1 {
        1
    } else {
        shape[..shape.len() - 1].iter().product()
    };
    let num_groups = if k == 0 {
        0
    } else {
        (k + group_size - 1) / group_size
    };

    // Per-(group, column) absmax sweep. The column-major loop
    // order is the natural "scan one block at a time" pattern;
    // for each (g, col) we touch exactly `group_size` elements
    // contiguous in the K direction (stride N in row-major
    // memory). Predictable strides; the optimiser autovectorises
    // the inner loop on AVX2.
    let mut scales: Vec<f32> = vec![0.0; num_groups * n];
    if k > 0 && n > 0 {
        for g in 0..num_groups {
            let row_lo = g * group_size;
            let row_hi = ((g + 1) * group_size).min(k);
            for col in 0..n {
                let mut max_abs = 0.0_f32;
                for row in row_lo..row_hi {
                    let v = weights[row * n + col].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                scales[g * n + col] = (max_abs / 127.0).max(1e-12);
            }
        }
    }

    let mut q: Vec<i8> = vec![0; total];
    if k > 0 && n > 0 {
        for row in 0..k {
            let g = row / group_size;
            for col in 0..n {
                let s = scales[g * n + col];
                let qf = (weights[row * n + col] / s).round();
                let qi = qf.clamp(-127.0, 127.0) as i32;
                q[row * n + col] = qi as i8;
            }
        }
    }

    (q, scales)
}

/// **M9.4** — convenience wrapper: build a CpuInt8 tensor from
/// an F32 source using per-group quantisation. The default
/// `group_size = 128` is the M9.4 production value (Q8_0).
pub fn quantize_int8_per_group(weights: &[f32], shape: &[usize], group_size: usize) -> Tensor {
    let (q, scales) = absmax_per_group_symmetric(weights, shape, group_size);
    Tensor::new_cpu_int8_per_group(shape.to_vec(), q, scales, group_size)
}

/// Borrow the `(q, scales, shape, group_size)` tuple out of a
/// [`TensorStorage::CpuInt8`] tensor without consuming it.
///
/// Returns `None` for any other storage variant. Used by the M9
/// CUDA upload path and tests; production matmul callers go
/// through `int8_per_group_to_bf16_in_vram` instead.
pub fn as_cpu_int8_view(tensor: &Tensor) -> Option<(&[i8], &[f32], &[usize], usize)> {
    match &tensor.storage {
        TensorStorage::CpuInt8 {
            q,
            scales,
            shape,
            group_size,
        } => Some((
            q.as_slice(),
            scales.as_slice(),
            shape.as_slice(),
            *group_size,
        )),
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
        assert!(
            (scales[0] - 1.0).abs() < 1e-9,
            "expected scale 1.0, got {}",
            scales[0]
        );
        assert_eq!(q, vec![2_i8, -4, 6, -127]);

        // Round-trip through TensorStorage::CpuInt8 → copy_to_cpu_vec.
        let t = Tensor::new_cpu_int8(vec![4, 1], q, scales);
        let dequant = t.copy_to_cpu_vec();
        assert_eq!(
            dequant, w,
            "bit-exact round-trip failed: {:?} vs {:?}",
            dequant, w
        );
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
                assert!(
                    err < 1.0 / 127.0 + 1e-6,
                    "round-trip error {err:e} at ({row}, {col}) exceeds 1/127"
                );
            }
        }
    }

    /// Scales vector length contract — must equal N (last axis).
    #[test]
    fn scales_length_equals_last_axis() {
        let (_q, scales) = absmax_per_channel_symmetric(&vec![0.5_f32; 5 * 7], &[5, 7]);
        assert_eq!(scales.len(), 7, "scales.len() must equal N = 7");

        let (_q, scales) = absmax_per_channel_symmetric(&vec![0.25_f32; 11 * 13], &[11, 13]);
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

        let (q_d, s_d, _, _) = as_cpu_int8_view(&direct).unwrap();
        let (q_w, s_w, _, _) = as_cpu_int8_view(&via_wrapper).unwrap();
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
        let (q, scales, shape, group_size) = view.unwrap();
        assert_eq!(q.len(), 4);
        assert_eq!(scales.len(), 2);
        assert_eq!(shape, &[2, 2]);
        // Per-channel construction sets group_size = K (one
        // group per column). For shape [2, 2], K = 2.
        assert_eq!(group_size, 2);
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
            TensorStorage::CpuInt8 {
                q,
                scales,
                shape,
                group_size,
            } => {
                assert_eq!(q.len(), 6);
                assert_eq!(scales.len(), 3);
                assert_eq!(shape, &vec![2usize, 3]);
                assert_eq!(*group_size, 2); // per-channel: one group per column
            }
            other => panic!("expected CpuInt8 storage, got {:?}", other),
        }
        assert_eq!(t.dtype, crate::tensor::DType::Int8);
    }

    // -----------------------------------------------------------------
    // M9.4 — per-group (Q8_0-style) quantisation tests.
    // -----------------------------------------------------------------

    /// **M9.4 contract** — per-group `scales.len()` is
    /// `ceil(K / group_size) * N`, not `N` like per-channel.
    #[test]
    fn m9_4_per_group_scales_length_contract() {
        // shape [256, 7], group_size = 128 → 2 groups × 7 = 14 scales.
        let (_q, scales) = absmax_per_group_symmetric(&vec![0.5_f32; 256 * 7], &[256, 7], 128);
        assert_eq!(
            scales.len(),
            2 * 7,
            "expected ceil(256/128) * 7 = 14 scales, got {}",
            scales.len()
        );

        // Non-divisible: shape [200, 7], g = 128 → ceil(200/128) = 2.
        let (_q, scales) = absmax_per_group_symmetric(&vec![0.5_f32; 200 * 7], &[200, 7], 128);
        assert_eq!(
            scales.len(),
            2 * 7,
            "expected ceil(200/128) * 7 = 14 scales, got {}",
            scales.len()
        );

        // group_size == K → one group per column → equivalent to
        // per-channel scales.
        let (_q, scales) = absmax_per_group_symmetric(&vec![0.5_f32; 5 * 7], &[5, 7], 5);
        assert_eq!(scales.len(), 7);
    }

    /// **M9.4 round-trip** — a single huge outlier in one group
    /// does NOT corrupt the dequant of elements in OTHER groups
    /// of the same column. This is the per-channel-vs-per-group
    /// contract that the M9.4 4-model F64 fixture is meant to
    /// surface as drift recovery.
    ///
    /// Setup: shape [256, 1], group_size = 128. Column 0 has
    /// 128 elements at ±0.001 in group 0 and a single 100.0
    /// outlier in group 1. Per-channel would scale to
    /// `100/127 ≈ 0.787`, crushing the 0.001 elements to zero
    /// (round-trip error ≈ 1.0). Per-group localises: group 0
    /// has its own small scale (`0.001/127`), group 1 has the
    /// outlier scale (`100/127`).
    #[test]
    fn m9_4_per_group_outlier_does_not_corrupt_other_groups() {
        let k = 256usize;
        let n = 1usize;
        let g = 128usize;

        let mut w = vec![0.0_f32; k * n];
        for row in 0..128 {
            // Group 0: small alternating ±0.001
            w[row * n + 0] = if row % 2 == 0 { 0.001 } else { -0.001 };
        }
        // Group 1: a single huge outlier at row 200, rest zero.
        w[200 * n + 0] = 100.0;

        // Per-group: outlier is contained in group 1.
        let (q_pg, scales_pg) = absmax_per_group_symmetric(&w, &[k, n], g);
        assert_eq!(scales_pg.len(), 2); // 2 groups × 1 col

        let t_pg = Tensor::new_cpu_int8_per_group(vec![k, n], q_pg, scales_pg, g);
        let dequant_pg = t_pg.copy_to_cpu_vec();

        let mut max_err_group_0 = 0.0_f32;
        for row in 0..128 {
            let err = (w[row * n] - dequant_pg[row * n]).abs();
            if err > max_err_group_0 {
                max_err_group_0 = err;
            }
        }
        assert!(
            max_err_group_0 < 1e-4,
            "per-group: group 0 max error {max_err_group_0:e} should stay close to \
             0.001/127 ≈ 7.87e-6 (the local scale's resolution); the outlier in \
             group 1 must NOT cross-contaminate"
        );

        // Per-channel reference: the same outlier crushes group 0.
        let (q_pc, scales_pc) = absmax_per_channel_symmetric(&w, &[k, n]);
        assert_eq!(scales_pc.len(), 1); // 1 column-wide scale
        let t_pc = Tensor::new_cpu_int8(vec![k, n], q_pc, scales_pc);
        let dequant_pc = t_pc.copy_to_cpu_vec();
        let mut max_err_pc_group_0 = 0.0_f32;
        for row in 0..128 {
            let err = (w[row * n] - dequant_pc[row * n]).abs();
            if err > max_err_pc_group_0 {
                max_err_pc_group_0 = err;
            }
        }
        // Per-channel scale is 100/127 ≈ 0.787; ±0.001 round to 0
        // → round-trip ≈ 0.001. That's ~127× worse than per-group.
        assert!(
            max_err_pc_group_0 > max_err_group_0 * 50.0,
            "per-channel error ({max_err_pc_group_0:e}) should be at least \
             50× per-group error ({max_err_group_0:e}) on this outlier-heavy \
             synthetic — got per_channel/per_group = {}",
            max_err_pc_group_0 / max_err_group_0.max(1e-12),
        );
    }

    /// **M9.4 round-trip** — when there are NO outliers
    /// (uniform-magnitude column) per-group and per-channel
    /// produce essentially the same drift envelope. Validates
    /// the per-group quantiser doesn't silently lose precision
    /// on benign inputs.
    #[test]
    fn m9_4_per_group_matches_per_channel_envelope_when_no_outliers() {
        let k = 256usize;
        let n = 3usize;
        let g = 128usize;

        // Linear ramp in [-1, 1] for every column.
        let mut w = vec![0.0_f32; k * n];
        for row in 0..k {
            for col in 0..n {
                w[row * n + col] = -1.0 + 2.0 * (row as f32) / ((k - 1) as f32);
            }
        }

        let t_pg = quantize_int8_per_group(&w, &[k, n], g);
        let dequant_pg = t_pg.copy_to_cpu_vec();

        // Element-wise round-trip envelope: |M|_max ≤ 1.0 in each
        // group → expected error ≤ 1/127.
        for row in 0..k {
            for col in 0..n {
                let err = (w[row * n + col] - dequant_pg[row * n + col]).abs();
                assert!(
                    err < 1.0 / 127.0 + 1e-6,
                    "per-group round-trip error {err:e} at ({row}, {col}) \
                     exceeds 1/127 on benign input"
                );
            }
        }
    }

    /// **M9.4 sanity** — `quantize_int8_per_group(shape, weights, K)`
    /// (one group per column) is bit-identical to
    /// `quantize_int8_w8a16(shape, weights)` (per-channel).
    /// Pins that the new code path is a strict generalisation.
    #[test]
    fn m9_4_per_group_with_full_column_group_matches_per_channel() {
        let k = 8usize;
        let n = 3usize;
        let mut w = vec![0.0_f32; k * n];
        for i in 0..(k * n) {
            w[i] = ((i as f32) * 0.1) - 0.4;
        }

        let pc = quantize_int8_w8a16(&w, &[k, n]);
        let pg = quantize_int8_per_group(&w, &[k, n], k);

        let (q_pc, s_pc, _, _) = as_cpu_int8_view(&pc).unwrap();
        let (q_pg, s_pg, _, _) = as_cpu_int8_view(&pg).unwrap();
        assert_eq!(q_pc, q_pg, "q bytes must match for group_size = K");
        assert_eq!(s_pc, s_pg, "scales must match for group_size = K");
    }

    /// **M9.4** — zero `group_size` is rejected (the constructor
    /// asserts).
    #[test]
    fn m9_4_per_group_rejects_zero_group_size() {
        let r = std::panic::catch_unwind(|| absmax_per_group_symmetric(&[0.0_f32; 4], &[2, 2], 0));
        assert!(r.is_err(), "expected panic on group_size = 0");
    }
}
