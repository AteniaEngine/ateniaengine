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

// ============================================================================
// M10β.1 — CPU-only outlier-decomposition spike.
//
// Experimental scaffolding for Track β of the M9 INT8 follow-up
// (`docs/HANDOFF_APX_V20_M9.md` §10). The M9 4-model F64 fixture
// showed that absmax INT8 (per-channel and per-group) cannot
// satisfy ADR-004 strict on Llama-family weights: a small number
// of outlier columns dominate the per-column `max(|W|)`, crushing
// the resolution of every other column to ~5 effective bits.
//
// This module's hypothesis: if we *remove* the top-k highest-
// magnitude columns from the INT8 base and preserve them exactly
// in a small dense F32 sidecar, the remaining absmax scales tighten
// and per-element drift drops sharply.
//
// **Scope contract (β.1).** CPU-only. No storage variant change.
// No loader / tier-planner / numcert / CUDA integration. The whole
// module is dead code at runtime — only consumed by the unit
// tests below and (future) the β.2 storage step.
//
// **The math** for a row-major `[K, N]` weight `W`, threshold k:
//
// ```text
//     c[n] = max_k(|W[k, n]|)                       (per-column absmax)
//     S    = top_k_indices(c)                       (set of outlier columns)
//     W'   = W with columns in S zeroed             (cleaned base)
//     (q, scales) = absmax_per_group_symmetric(W', shape, group_size)
//     O[k, j] = W[k, S[j]]   for j in [0, |S|)      (dense sidecar)
//
//     reconstruct(k, n) = dequant(q, scales)[k, n]   if n ∉ S
//                       = O[k, index_in_S(n)]        if n ∈ S
// ```
//
// `outlier_values` is **row-major `K × |S|`**: the j-th outlier
// column's K values land at offsets `[0..K] * |S| + j`. This is
// the layout that makes the reconstruction inner loop (over `j`)
// stride-1.

/// Errors returned by [`decompose_outliers_topk_by_absmax`]. The
/// spike uses an inline enum (rather than panicking like the
/// pre-existing absmax helpers) because the validation surface is
/// richer and the experimental path benefits from being driveable
/// from tests without `catch_unwind`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutlierDecompositionError {
    /// `shape` is not exactly 2 axes. The β.1 spike only handles
    /// `[K, N]`; higher-rank fallthrough is future work.
    NotTwoDimensional { rank: usize },
    /// `values.len() != K * N` for the given shape.
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    /// `group_size == 0`.
    InvalidGroupSize,
    /// `k > N` — cannot pick more outlier columns than the matrix
    /// has columns to begin with.
    OutlierKExceedsColumns { k: usize, n: usize },
}

impl std::fmt::Display for OutlierDecompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutlierDecompositionError::NotTwoDimensional { rank } => write!(
                f,
                "decompose_outliers_topk_by_absmax: shape must be 2D (got rank {rank})"
            ),
            OutlierDecompositionError::LengthMismatch { expected, actual } => write!(
                f,
                "decompose_outliers_topk_by_absmax: values.len() = {actual} does not match K * N = {expected}"
            ),
            OutlierDecompositionError::InvalidGroupSize => write!(
                f,
                "decompose_outliers_topk_by_absmax: group_size must be > 0"
            ),
            OutlierDecompositionError::OutlierKExceedsColumns { k, n } => write!(
                f,
                "decompose_outliers_topk_by_absmax: k = {k} exceeds N = {n}"
            ),
        }
    }
}

impl std::error::Error for OutlierDecompositionError {}

/// One outlier-decomposed weight. Carries the INT8 per-group base
/// + a dense F32 sidecar of preserved-exactly columns.
///
/// β.1 uses F32 in the sidecar so the spike can isolate the
/// numerical question from any subsequent BF16 packing question
/// (which belongs to β.2 storage work). Memory footprint is
/// therefore worse than what β.2 will end at; the spike is not
/// trying to win on bytes.
#[derive(Debug, Clone)]
pub struct OutlierDecomposition {
    /// INT8 base — same layout / semantics as the existing
    /// [`absmax_per_group_symmetric`] output. Length `K * N`.
    pub q: Vec<i8>,
    /// Per-(group, column) scales. Length
    /// `ceil(K / group_size) * N`. Outlier columns' entries are
    /// `1e-12` clamp values (their `q` rows are zeroed in
    /// [`decompose_outliers_topk_by_absmax`]).
    pub scales: Vec<f32>,
    /// Original shape `[K, N]`.
    pub shape: Vec<usize>,
    /// Group size along the K axis (matches the M9.4 contract).
    pub group_size: usize,
    /// Sorted-ascending list of column indices preserved in the
    /// sidecar. Length `M = min(k, N)`.
    pub outlier_cols: Vec<usize>,
    /// **Row-major `K × M`** dense F32 sidecar. Element
    /// `(row, j)` lives at offset `row * M + j` and equals the
    /// original `values[row * N + outlier_cols[j]]`.
    pub outlier_values: Vec<f32>,
}

impl OutlierDecomposition {
    /// `M`, the number of outlier columns actually preserved.
    pub fn outlier_count(&self) -> usize {
        self.outlier_cols.len()
    }
}

/// **M10β.1** — decompose a 2D weight into an INT8 per-group base
/// plus a top-k dense outlier sidecar.
///
/// `k = 0` is the boundary case that produces an empty sidecar
/// and a pure-INT8 base; `k = N` produces zero INT8 quality (all
/// columns are outliers) and an exact-reconstruction sidecar.
/// Anything in between is the experimental regime.
///
/// Ties on per-column absmax are broken by lower index first
/// (deterministic). NaN / inf inputs are not validated — same
/// contract as [`absmax_per_group_symmetric`].
pub fn decompose_outliers_topk_by_absmax(
    values: &[f32],
    shape: &[usize],
    group_size: usize,
    k: usize,
) -> Result<OutlierDecomposition, OutlierDecompositionError> {
    if shape.len() != 2 {
        return Err(OutlierDecompositionError::NotTwoDimensional { rank: shape.len() });
    }
    if group_size == 0 {
        return Err(OutlierDecompositionError::InvalidGroupSize);
    }
    let (k_rows, n_cols) = (shape[0], shape[1]);
    let expected = k_rows * n_cols;
    if values.len() != expected {
        return Err(OutlierDecompositionError::LengthMismatch {
            expected,
            actual: values.len(),
        });
    }
    if k > n_cols {
        return Err(OutlierDecompositionError::OutlierKExceedsColumns { k, n: n_cols });
    }

    // Per-column absmax sweep. Bypassed when the matrix is empty.
    let mut col_absmax: Vec<f32> = vec![0.0; n_cols];
    if k_rows > 0 && n_cols > 0 {
        for row in 0..k_rows {
            for col in 0..n_cols {
                let v = values[row * n_cols + col].abs();
                if v > col_absmax[col] {
                    col_absmax[col] = v;
                }
            }
        }
    }

    // Pick the top-k column indices by absmax. Tie-break by lower
    // column index first so the function is deterministic.
    let mut indexed: Vec<(usize, f32)> =
        col_absmax.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let actual_k = k.min(n_cols);
    let mut outlier_cols: Vec<usize> = indexed.into_iter().take(actual_k).map(|(i, _)| i).collect();
    outlier_cols.sort_unstable();

    // Build the dense sidecar `O[row, j] = values[row, outlier_cols[j]]`.
    let m = outlier_cols.len();
    let mut outlier_values: Vec<f32> = vec![0.0; k_rows * m];
    if m > 0 && k_rows > 0 {
        for row in 0..k_rows {
            for (j, &col) in outlier_cols.iter().enumerate() {
                outlier_values[row * m + j] = values[row * n_cols + col];
            }
        }
    }

    // Build the cleaned base: copy of `values` with outlier columns
    // zeroed. This is what gets handed to the existing per-group
    // absmax quantiser — the M9.4 algorithm is reused unchanged.
    let mut cleaned: Vec<f32> = values.to_vec();
    if m > 0 && k_rows > 0 {
        // `outlier_cols` is sorted ascending; binary_search is
        // O(log M) per (row, col). For the spike this is fine; β.2
        // can replace with a bitmap when the storage settles.
        for row in 0..k_rows {
            for col in 0..n_cols {
                if outlier_cols.binary_search(&col).is_ok() {
                    cleaned[row * n_cols + col] = 0.0;
                }
            }
        }
    }

    let (q, scales) = absmax_per_group_symmetric(&cleaned, shape, group_size);

    Ok(OutlierDecomposition {
        q,
        scales,
        shape: shape.to_vec(),
        group_size,
        outlier_cols,
        outlier_values,
    })
}

/// Reconstruct the full `[K, N]` matrix as a row-major `Vec<f32>`.
///
/// Dequantises the INT8 base into a fresh buffer, then overwrites
/// the outlier-column slots with the exact F32 sidecar values.
/// Equivalent (numerically) to evaluating
/// `dequant(q, scales) · (I − P_S) + W_O · P_S` where `P_S` is the
/// projection onto outlier columns and `W_O` is the sidecar.
pub fn reconstruct_outlier_decomposition(decomp: &OutlierDecomposition) -> Vec<f32> {
    let (k_rows, n_cols) = (decomp.shape[0], decomp.shape[1]);
    let mut out: Vec<f32> = vec![0.0; k_rows * n_cols];
    if k_rows == 0 || n_cols == 0 {
        return out;
    }

    // Dequant the INT8 base.
    for row in 0..k_rows {
        let g = row / decomp.group_size;
        for col in 0..n_cols {
            let s = decomp.scales[g * n_cols + col];
            out[row * n_cols + col] = (decomp.q[row * n_cols + col] as f32) * s;
        }
    }

    // Splice the outlier sidecar over the top.
    let m = decomp.outlier_cols.len();
    for (j, &col) in decomp.outlier_cols.iter().enumerate() {
        for row in 0..k_rows {
            out[row * n_cols + col] = decomp.outlier_values[row * m + j];
        }
    }

    out
}

// ============================================================================
// M10β-pivot.1 — AWQ-style per-row scaling spike (CPU only, no calibration).
//
// Track β.5 showed that removing per-column weight outliers does not satisfy
// ADR-004 end-to-end on TinyLlama; the cascade amplifies per-element error
// ~60× over the linear projection. The β-pivot.1 hypothesis is that the
// residual drift is dominated by *which* per-K-row channels matter, not by
// the magnitude of isolated weight outliers.
//
// AWQ (Lin et al., 2023) addresses this by applying a per-input-channel
// scale `s[k]` derived from activation statistics: large-activation rows
// receive a multiplicative boost before quantisation (so their absmax-
// driven INT8 scale is larger, giving them more precision), and the matmul
// math is preserved by dividing the dequantised weight by `s[k]` at
// reconstruction time:
//
// ```text
//     W'[k, n] = W[k, n] · s[k]                        (pre-scale)
//     (q, group_scales) = absmax_per_group(W', shape, g)
//     W_recon[k, n] = (q[k, n] · group_scales[g, n]) / s[k]
// ```
//
// At infinite precision `W_recon == W`; with INT8 precision the channels
// with large `s[k]` retain proportionally more quantisation headroom.
//
// **β-pivot.1 simplification: no-calibration weight-norm scales.** A proper
// AWQ run captures activation stats from a calibration pass. This spike
// substitutes a *weight-derived* scale (the per-row L2 norm raised to a
// power α and normalised to unit mean) so the experiment can run without
// a calibration harness. The numerical question this answers is binary:
//   - if per-row scaling improves drift markedly on TinyLlama, the
//     activation-aware hypothesis is supported and the next step is a real
//     calibration pass.
//   - if it does not improve drift, weight-only INT8 (with or without
//     activation-aware scaling) is hitting the model's intrinsic cascade
//     ceiling and the right pivot is GPTQ-class (Hessian-inverse).
//
// The helper is CPU-only, deterministic, dead code at runtime, and
// consumed exclusively by the β-pivot.1 forward harness in
// `tests/int8_outlier_f64_validation_test.rs`.

/// Errors returned by the AWQ helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwqError {
    NotTwoDimensional { rank: usize },
    LengthMismatch { expected: usize, actual: usize },
    InvalidGroupSize,
    InvalidAlpha,
    ScalesLengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for AwqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AwqError::NotTwoDimensional { rank } => {
                write!(f, "AWQ helpers require a 2D weight (got rank {rank})")
            }
            AwqError::LengthMismatch { expected, actual } => write!(
                f,
                "AWQ: values.len() = {actual} does not match K * N = {expected}"
            ),
            AwqError::InvalidGroupSize => write!(f, "AWQ: group_size must be > 0"),
            AwqError::InvalidAlpha => {
                write!(f, "AWQ: alpha must be finite and in [0, 1] for the spike")
            }
            AwqError::ScalesLengthMismatch { expected, actual } => write!(
                f,
                "AWQ: scales.len() = {actual} does not match K = {expected}"
            ),
        }
    }
}

impl std::error::Error for AwqError {}

/// **M10β-pivot.1** — per-K-row scales derived from the L2 norm of each
/// row, raised to the power `alpha` and normalised to unit mean.
///
/// This is the *no-calibration* fallback: no activation statistics are
/// required, so the spike can run without a calibration harness. The
/// normalisation keeps the scales from drifting the absolute weight
/// magnitude away from unit (which would interact badly with downstream
/// RMSNorm).
///
/// Recommended α defaults: `0.0` (everything = 1.0, equivalent to plain
/// INT8), `0.5` (the spike default), `1.0` (full norm-weighted scaling).
pub fn awq_per_row_scales_from_weight_norm(
    weights: &[f32],
    shape: &[usize],
    alpha: f32,
) -> Result<Vec<f32>, AwqError> {
    if shape.len() != 2 {
        return Err(AwqError::NotTwoDimensional { rank: shape.len() });
    }
    if !alpha.is_finite() || alpha < 0.0 || alpha > 1.0 {
        return Err(AwqError::InvalidAlpha);
    }
    let (k_rows, n_cols) = (shape[0], shape[1]);
    let expected = k_rows * n_cols;
    if weights.len() != expected {
        return Err(AwqError::LengthMismatch {
            expected,
            actual: weights.len(),
        });
    }

    if k_rows == 0 || n_cols == 0 {
        return Ok(Vec::new());
    }

    let mut row_norm: Vec<f32> = vec![0.0; k_rows];
    for row in 0..k_rows {
        let mut acc = 0.0_f64;
        for col in 0..n_cols {
            let v = weights[row * n_cols + col] as f64;
            acc += v * v;
        }
        // sqrt of the sum of squares, then raise to alpha. Clamp to a
        // small floor so a degenerate all-zero row does not produce a
        // zero scale (which would zero the dequant pre-multiplication).
        row_norm[row] = (acc.sqrt().powf(alpha as f64) as f32).max(1e-8);
    }

    // Normalise to unit mean — keeps the post-perturbation weight at the
    // same average magnitude as the input.
    let mean: f64 =
        row_norm.iter().map(|x| *x as f64).sum::<f64>() / (row_norm.len() as f64);
    let mean = mean.max(1e-8) as f32;
    for v in &mut row_norm {
        *v /= mean;
    }

    Ok(row_norm)
}

/// **M10β-pivot.1** — apply the AWQ-style quantisation perturbation to
/// the weight buffer in place, given per-K-row scales.
///
/// The output is the F32 reconstruction `(q * group_scales) / s[k]`,
/// which means the dispatcher consumes a plain F32 tensor and the
/// runtime does not need to know about AWQ at all. Drift attributable
/// to INT8 quantisation is encoded in the buffer's perturbation from
/// the input.
pub fn apply_awq_perturbation_inplace(
    weights: &mut [f32],
    shape: &[usize],
    group_size: usize,
    scales: &[f32],
) -> Result<(), AwqError> {
    if shape.len() != 2 {
        return Err(AwqError::NotTwoDimensional { rank: shape.len() });
    }
    if group_size == 0 {
        return Err(AwqError::InvalidGroupSize);
    }
    let (k_rows, n_cols) = (shape[0], shape[1]);
    let expected = k_rows * n_cols;
    if weights.len() != expected {
        return Err(AwqError::LengthMismatch {
            expected,
            actual: weights.len(),
        });
    }
    if scales.len() != k_rows {
        return Err(AwqError::ScalesLengthMismatch {
            expected: k_rows,
            actual: scales.len(),
        });
    }
    if k_rows == 0 || n_cols == 0 {
        return Ok(());
    }

    // (1) Pre-scale W' = W * s per row.
    for row in 0..k_rows {
        let s = scales[row];
        for col in 0..n_cols {
            weights[row * n_cols + col] *= s;
        }
    }

    // (2) Per-group absmax INT8 quantisation of the scaled buffer.
    let (q, group_scales) = absmax_per_group_symmetric(weights, shape, group_size);

    // (3) Dequantise into the same buffer.
    for idx in 0..expected {
        let row = idx / n_cols;
        let col = idx % n_cols;
        let g = row / group_size;
        weights[idx] = (q[idx] as f32) * group_scales[g * n_cols + col];
    }

    // (4) Inverse-scale the dequant. The matmul math is now
    //     X @ W_recon, where W_recon = (W * s through quant) / s
    //     ≈ W minus the quantisation noise weighted by 1/s.
    for row in 0..k_rows {
        let inv = 1.0 / scales[row].max(1e-8);
        for col in 0..n_cols {
            weights[row * n_cols + col] *= inv;
        }
    }

    Ok(())
}

/// Naive `[B, K] × [K, N]` row-major matmul against a
/// reconstructed [`OutlierDecomposition`]. O(BKN) — for the β.1
/// spike tests only. Not exposed to the runtime, not on the hot
/// path of anything; β.6 will introduce a CUDA mixed-precision
/// kernel that fuses the two products.
pub fn matmul_outlier_decomposition_cpu(
    lhs: &[f32],
    lhs_shape: &[usize],
    rhs: &OutlierDecomposition,
) -> Vec<f32> {
    assert_eq!(
        lhs_shape.len(),
        2,
        "matmul_outlier_decomposition_cpu: lhs must be 2D"
    );
    let (b, k_lhs) = (lhs_shape[0], lhs_shape[1]);
    let (k_rhs, n) = (rhs.shape[0], rhs.shape[1]);
    assert_eq!(
        k_lhs, k_rhs,
        "matmul_outlier_decomposition_cpu: lhs cols ({k_lhs}) != rhs rows ({k_rhs})"
    );
    let w = reconstruct_outlier_decomposition(rhs);
    let mut out = vec![0.0_f32; b * n];
    for i in 0..b {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k_lhs {
                acc += lhs[i * k_lhs + kk] * w[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
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

    // ====================================================================
    // M10β.1 — outlier-decomposition spike tests
    // ====================================================================

    /// Compute max(|a - b|) over two same-length slices. Helper for
    /// the error-reduction assertions below.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a 2D matrix where a handful of columns carry values
    /// orders of magnitude larger than the bulk — the regime that
    /// crushes plain absmax INT8 (HANDOFF_M9 §4 root-cause).
    fn synth_outlier_matrix(
        k: usize,
        n: usize,
        outlier_cols: &[usize],
        bulk_scale: f32,
        outlier_scale: f32,
    ) -> Vec<f32> {
        let mut w = vec![0.0_f32; k * n];
        for row in 0..k {
            for col in 0..n {
                // Pseudo-random-ish but reproducible: a small,
                // deterministic spread around zero.
                let idx = (row * n + col) as i32;
                let bit = ((idx.wrapping_mul(1103515245).wrapping_add(12345) >> 7) & 0xff) as f32;
                let base = (bit - 128.0) / 128.0; // ∈ [-1, ~1)
                let scale = if outlier_cols.contains(&col) {
                    outlier_scale
                } else {
                    bulk_scale
                };
                w[row * n + col] = base * scale;
            }
        }
        // Inject extreme spikes on a couple of rows in each outlier
        // column to guarantee the absmax detector picks them.
        for &col in outlier_cols {
            if k > 0 {
                w[col] = outlier_scale * 2.0; // row 0
            }
            if k > 1 {
                w[1 * n + col] = -outlier_scale * 2.0;
            }
        }
        w
    }

    #[test]
    fn decompose_outliers_topk_preserves_selected_columns() {
        let k = 8;
        let n = 6;
        let outliers = vec![1usize, 4];
        let w = synth_outlier_matrix(k, n, &outliers, 0.1, 10.0);

        let d = decompose_outliers_topk_by_absmax(&w, &[k, n], 4, 2).unwrap();

        // Top-2 by absmax with this synthesis must be columns 1 and 4.
        assert_eq!(d.outlier_cols, outliers);
        // Sidecar layout: [K, M=2] row-major, j-th column = original
        // values at outlier_cols[j].
        assert_eq!(d.outlier_values.len(), k * 2);
        for row in 0..k {
            for (j, &col) in d.outlier_cols.iter().enumerate() {
                assert_eq!(
                    d.outlier_values[row * 2 + j],
                    w[row * n + col],
                    "sidecar mismatch at (row={row}, j={j})"
                );
            }
        }
    }

    #[test]
    fn decompose_outliers_reconstructs_outlier_columns_exactly() {
        let k = 6;
        let n = 5;
        let outliers = vec![0usize, 3];
        let w = synth_outlier_matrix(k, n, &outliers, 0.05, 8.0);

        let d = decompose_outliers_topk_by_absmax(&w, &[k, n], 3, 2).unwrap();
        let recon = reconstruct_outlier_decomposition(&d);

        // Outlier-column slots must round-trip with zero error.
        for row in 0..k {
            for &col in &outliers {
                let lhs = recon[row * n + col];
                let rhs = w[row * n + col];
                assert_eq!(
                    lhs, rhs,
                    "outlier col {col} row {row} should reconstruct bit-exact (got {lhs}, want {rhs})"
                );
            }
        }
    }

    #[test]
    fn decompose_outliers_zero_k_matches_pure_int8_envelope() {
        let k = 8;
        let n = 6;
        let w = synth_outlier_matrix(k, n, &[], 0.5, 0.5);

        // k=0 → empty sidecar, base is the standard per-group quant.
        let d = decompose_outliers_topk_by_absmax(&w, &[k, n], 4, 0).unwrap();
        assert!(d.outlier_cols.is_empty());
        assert!(d.outlier_values.is_empty());

        let recon = reconstruct_outlier_decomposition(&d);
        let (q_ref, s_ref) = absmax_per_group_symmetric(&w, &[k, n], 4);
        // Manually dequant the reference and compare element-wise.
        let mut ref_recon = vec![0.0_f32; k * n];
        for row in 0..k {
            let g = row / 4;
            for col in 0..n {
                ref_recon[row * n + col] =
                    (q_ref[row * n + col] as f32) * s_ref[g * n + col];
            }
        }
        assert_eq!(
            recon, ref_recon,
            "k=0 path must be numerically identical to absmax_per_group_symmetric"
        );
    }

    #[test]
    fn decompose_outliers_k_equals_n_reconstructs_exactly() {
        let k = 5;
        let n = 4;
        let w = synth_outlier_matrix(k, n, &[1, 3], 0.4, 7.0);

        // k=N → every column is a sidecar column → reconstruction
        // is bit-exact regardless of the INT8 base.
        let d = decompose_outliers_topk_by_absmax(&w, &[k, n], 2, n).unwrap();
        assert_eq!(d.outlier_cols.len(), n);
        let recon = reconstruct_outlier_decomposition(&d);
        assert_eq!(
            recon, w,
            "k=N must round-trip the input exactly via the sidecar"
        );
    }

    #[test]
    fn decompose_outliers_reduces_error_vs_plain_int8_on_outlier_matrix() {
        // The headline assertion: on a matrix with two strong
        // outlier columns, splitting them out of the INT8 base
        // must reduce the reconstruction error by a clear margin
        // versus M9.4's `absmax_per_group_symmetric` baseline.
        let k = 32;
        let n = 16;
        let outliers = vec![5usize, 12];
        let w = synth_outlier_matrix(k, n, &outliers, 0.1, 50.0);

        // Plain INT8 per-group, group_size=8.
        let (q_plain, s_plain) = absmax_per_group_symmetric(&w, &[k, n], 8);
        let mut plain_recon = vec![0.0_f32; k * n];
        for row in 0..k {
            let g = row / 8;
            for col in 0..n {
                plain_recon[row * n + col] =
                    (q_plain[row * n + col] as f32) * s_plain[g * n + col];
            }
        }
        let plain_max_err = max_abs_diff(&plain_recon, &w);

        // Outlier-decomposed, k=2 (matches the two synthesised
        // outliers).
        let d = decompose_outliers_topk_by_absmax(&w, &[k, n], 8, 2).unwrap();
        let recon = reconstruct_outlier_decomposition(&d);
        let decomp_max_err = max_abs_diff(&recon, &w);

        // Outlier decomposition must clear the bar by a wide
        // margin: the synthesised outliers are 500× the bulk,
        // and isolating them should make the remaining base
        // resolution ~250× better. Assert at least 10× improvement
        // to leave headroom for the rounding noise floor.
        assert!(
            decomp_max_err * 10.0 < plain_max_err,
            "outlier decomposition should be ≥10× better; got \
             plain={plain_max_err} decomp={decomp_max_err}"
        );

        // Sanity: even when the matmul goes through the
        // reconstructed weight, the error stays bounded.
        let b = 3;
        let lhs: Vec<f32> = (0..(b * k))
            .map(|i| (((i as i32 * 11) % 17) as f32) / 5.0 - 1.6)
            .collect();
        let y_true = {
            let mut y = vec![0.0_f32; b * n];
            for i in 0..b {
                for j in 0..n {
                    let mut acc = 0.0_f32;
                    for kk in 0..k {
                        acc += lhs[i * k + kk] * w[kk * n + j];
                    }
                    y[i * n + j] = acc;
                }
            }
            y
        };
        let y_decomp = matmul_outlier_decomposition_cpu(&lhs, &[b, k], &d);
        let y_plain = {
            let mut y = vec![0.0_f32; b * n];
            for i in 0..b {
                for j in 0..n {
                    let mut acc = 0.0_f32;
                    for kk in 0..k {
                        acc += lhs[i * k + kk] * plain_recon[kk * n + j];
                    }
                    y[i * n + j] = acc;
                }
            }
            y
        };
        let mm_plain_err = max_abs_diff(&y_plain, &y_true);
        let mm_decomp_err = max_abs_diff(&y_decomp, &y_true);
        assert!(
            mm_decomp_err * 5.0 < mm_plain_err,
            "matmul error after decomp should be ≥5× better; got \
             plain_mm={mm_plain_err} decomp_mm={mm_decomp_err}"
        );
    }

    #[test]
    fn decompose_outliers_rejects_non_2d_shape() {
        let err = decompose_outliers_topk_by_absmax(&[0.0_f32; 6], &[2, 3, 1], 2, 1).unwrap_err();
        assert!(matches!(
            err,
            OutlierDecompositionError::NotTwoDimensional { rank: 3 }
        ));
        let err = decompose_outliers_topk_by_absmax(&[0.0_f32; 4], &[4], 2, 1).unwrap_err();
        assert!(matches!(
            err,
            OutlierDecompositionError::NotTwoDimensional { rank: 1 }
        ));
    }

    #[test]
    fn decompose_outliers_rejects_invalid_k() {
        // k > N is rejected.
        let err = decompose_outliers_topk_by_absmax(&[0.0_f32; 6], &[3, 2], 1, 3).unwrap_err();
        assert!(matches!(
            err,
            OutlierDecompositionError::OutlierKExceedsColumns { k: 3, n: 2 }
        ));
        // k == N is allowed (boundary, asserted above in its own test).
        assert!(decompose_outliers_topk_by_absmax(&[0.0_f32; 6], &[3, 2], 1, 2).is_ok());
    }

    #[test]
    fn decompose_outliers_rejects_invalid_group_size() {
        let err = decompose_outliers_topk_by_absmax(&[0.0_f32; 4], &[2, 2], 0, 1).unwrap_err();
        assert!(matches!(err, OutlierDecompositionError::InvalidGroupSize));
    }

    #[test]
    fn decompose_outliers_rejects_length_mismatch() {
        let err = decompose_outliers_topk_by_absmax(&[0.0_f32; 5], &[2, 3], 1, 1).unwrap_err();
        assert!(matches!(
            err,
            OutlierDecompositionError::LengthMismatch {
                expected: 6,
                actual: 5
            }
        ));
    }

    // ---- β-pivot.1 AWQ helpers ----

    #[test]
    fn awq_scales_are_finite_and_have_unit_mean() {
        let w: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let s = awq_per_row_scales_from_weight_norm(&w, &[8, 4], 0.5).unwrap();
        assert_eq!(s.len(), 8);
        assert!(s.iter().all(|v| v.is_finite()), "no NaN/inf");
        assert!(s.iter().all(|v| *v > 0.0), "no zero scales");
        let mean: f32 = s.iter().sum::<f32>() / (s.len() as f32);
        assert!(
            (mean - 1.0).abs() < 1e-4,
            "scales must normalise to unit mean (got {mean})"
        );
    }

    #[test]
    fn awq_scales_alpha_zero_collapses_to_uniform() {
        let w: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let s = awq_per_row_scales_from_weight_norm(&w, &[4, 4], 0.0).unwrap();
        // α = 0 means every row contributes the same `1.0` before
        // normalisation, so all-ones to within float rounding.
        for v in &s {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn awq_scales_alpha_one_emphasises_large_norm_rows() {
        let mut w = vec![0.0_f32; 8];
        for col in 0..4 {
            w[col] = 1.0; // row 0
            w[4 + col] = 0.1; // row 1
        }
        let s = awq_per_row_scales_from_weight_norm(&w, &[2, 4], 1.0).unwrap();
        assert!(
            s[0] > 5.0 * s[1],
            "α = 1 must amplify the large-norm row (got s[0]={}, s[1]={})",
            s[0],
            s[1]
        );
    }

    #[test]
    fn awq_scales_reject_non_2d_shape() {
        let w = vec![0.5_f32; 8];
        let r = awq_per_row_scales_from_weight_norm(&w, &[2, 2, 2], 0.5);
        assert!(matches!(r, Err(AwqError::NotTwoDimensional { rank: 3 })));
    }

    #[test]
    fn awq_scales_reject_invalid_alpha() {
        let w = vec![0.5_f32; 8];
        assert!(matches!(
            awq_per_row_scales_from_weight_norm(&w, &[2, 4], -0.5),
            Err(AwqError::InvalidAlpha)
        ));
        assert!(matches!(
            awq_per_row_scales_from_weight_norm(&w, &[2, 4], 1.5),
            Err(AwqError::InvalidAlpha)
        ));
    }

    #[test]
    fn awq_perturbation_is_deterministic() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32 * 0.07 - 1.0).sin()).collect();
        let scales = awq_per_row_scales_from_weight_norm(&w, &[16, 16], 0.5).unwrap();
        let mut a = w.clone();
        let mut b = w.clone();
        apply_awq_perturbation_inplace(&mut a, &[16, 16], 8, &scales).unwrap();
        apply_awq_perturbation_inplace(&mut b, &[16, 16], 8, &scales).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn awq_perturbation_with_alpha_zero_matches_plain_int8() {
        // α = 0 → s = [1, 1, ...] → AWQ degenerates to plain
        // per-group absmax INT8 round-trip.
        let w: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03 - 0.7).cos()).collect();
        let scales = vec![1.0_f32; 16];
        let mut awq_path = w.clone();
        apply_awq_perturbation_inplace(&mut awq_path, &[16, 8], 8, &scales).unwrap();
        let (q, sc) = absmax_per_group_symmetric(&w, &[16, 8], 8);
        let mut plain_path = vec![0.0_f32; w.len()];
        for idx in 0..w.len() {
            let row = idx / 8;
            let col = idx % 8;
            let g = row / 8;
            plain_path[idx] = (q[idx] as f32) * sc[g * 8 + col];
        }
        let max_diff = awq_path
            .iter()
            .zip(&plain_path)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "AWQ α=0 must collapse to plain INT8 (max diff = {max_diff})"
        );
    }

    #[test]
    fn awq_perturbation_preserves_shape() {
        let w: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let mut buf = w.clone();
        let scales = awq_per_row_scales_from_weight_norm(&w, &[16, 16], 0.5).unwrap();
        apply_awq_perturbation_inplace(&mut buf, &[16, 16], 16, &scales).unwrap();
        assert_eq!(buf.len(), w.len());
    }

    #[test]
    fn awq_perturbation_rejects_scales_length_mismatch() {
        let mut buf = vec![0.5_f32; 16];
        let err = apply_awq_perturbation_inplace(&mut buf, &[4, 4], 4, &[1.0; 3])
            .expect_err("must reject K-mismatched scales");
        assert!(matches!(
            err,
            AwqError::ScalesLengthMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }
}
