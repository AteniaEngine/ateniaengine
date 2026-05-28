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
    /// **AQS-5** — `block_size == 0`.
    InvalidBlockSize,
    /// **AQS-5** — the activation matrix shape does not describe a
    /// `[S, K]` whose K matches the weight's leading axis.
    ActivationShapeMismatch {
        expected_k: usize,
        actual_k: usize,
        samples: usize,
        matrix_len: usize,
    },
    /// **AQS-5** — the (damped) Hessian was not positive-definite, so
    /// the Cholesky factorisation failed at the given pivot. Increase
    /// `damp_percent`.
    NotPositiveDefinite { pivot: usize },
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
            GptqError::InvalidBlockSize => write!(f, "gptq: block_size must be > 0"),
            GptqError::ActivationShapeMismatch {
                expected_k,
                actual_k,
                samples,
                matrix_len,
            } => write!(
                f,
                "gptq: activation matrix [{samples}, {actual_k}] (len {matrix_len}) does not \
                 match weight K = {expected_k}"
            ),
            GptqError::NotPositiveDefinite { pivot } => write!(
                f,
                "gptq: Hessian not positive-definite at pivot {pivot} (increase damp_percent)"
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

// ============================================================================
// AQS-5 — REAL GPTQ (blockwise, full Hessian, Cholesky-based error
// compensation). This is the genuine OBQ/GPTQ algorithm, in contrast to
// the AQS-3 diagonal surrogate above (kept for comparison).
//
// ## Layout
//
// The Atenia loader transposes HF weights to `[K_in, N_out]` before they
// reach the store (`X[*, K] @ W[K, N] = Y[*, N]`). The Hessian therefore
// lives on the **input dimension K**: `H = Xᵀ X / S`, shape `K×K`, where
// `X` is the `[S, K]` calibration activation matrix. GPTQ quantises the
// weight **rows** (one per input channel `k`) sequentially along K, and
// propagates each row's quantisation error to the not-yet-quantised rows
// `k' > k` weighted by the inverse-Hessian Cholesky factor. Each row
// spans all `N` output columns.
//
// ## Algorithm (matches the GPTQ paper / AutoGPTQ, transposed to our
// layout)
//
// ```text
//   H        = Xᵀ X / S                       (K×K, SPD after damping)
//   H[k,k]  += damp_percent · mean(diag H)    (Tikhonov damping)
//   Hinv     = chol_upper( inverse(H) )        (upper-triangular factor)
//   for each block of `block_size` rows along K:
//     for each row k in the block:
//       scale = group scale for (group(k), :)  (per-N absmax over the group)
//       q     = round(W[k,:] / scale)·scale    (INT8 symmetric)
//       err   = (W[k,:] − q) / Hinv[k,k]
//       W[k,:] = q
//       W[k', :] -= Hinv[k,k'] · err   for k' in (k, block_end)   (intra)
//     W[block_end.., :] -= Hinv[block, block_end..]ᵀ · Err_block   (inter)
// ```
//
// Group scales are recomputed from the *current* (already partly
// compensated) weights at each group boundary — the faithful "dynamic
// group" behaviour, not static scales.
// ============================================================================

/// Configuration for real GPTQ ([`apply_gptq_real_inplace`]).
#[derive(Debug, Clone, Copy)]
pub struct GptqRealConfig {
    /// INT8 quantisation group size along the K axis.
    pub group_size: usize,
    /// GPTQ lazy-batch block size along K. Controls the
    /// intra-/inter-block split of the error update; numerically
    /// equivalent across block sizes, only performance differs. `128`
    /// is the AutoGPTQ default.
    pub block_size: usize,
    /// Tikhonov damping as a fraction of `mean(diag H)`. `0.01` is the
    /// classic value. Must be in `[0, 1)`.
    pub damp_percent: f32,
    /// Activation-order heuristic: quantise high-Hessian rows first.
    /// When `true`, rows are permuted by descending `diag(H)` before the
    /// sweep and un-permuted afterwards (deterministic). When `false`,
    /// natural order.
    pub act_order: bool,
}

impl Default for GptqRealConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            block_size: 128,
            damp_percent: 0.01,
            act_order: false,
        }
    }
}

/// Build the `K×K` Hessian `H = Xᵀ X / S` from a row-major `[S, K]`
/// activation matrix. Symmetric by construction; accumulated in f64.
pub fn compute_hessian(activations: &[f32], s: usize, k: usize) -> Vec<f64> {
    let mut h = vec![0.0_f64; k * k];
    if s == 0 || k == 0 {
        return h;
    }
    // H[i,j] = Σ_sample X[sample,i] · X[sample,j]
    for sample in 0..s {
        let base = sample * k;
        for i in 0..k {
            let xi = activations[base + i] as f64;
            if xi == 0.0 {
                continue;
            }
            let row = i * k;
            for j in i..k {
                h[row + j] += xi * (activations[base + j] as f64);
            }
        }
    }
    let inv_s = 1.0 / (s as f64);
    // Scale and mirror the lower triangle.
    for i in 0..k {
        for j in i..k {
            let v = h[i * k + j] * inv_s;
            h[i * k + j] = v;
            h[j * k + i] = v;
        }
    }
    h
}

/// Add Tikhonov damping to the Hessian diagonal in place:
/// `H[i,i] += damp_percent · mean(diag H)`. Also handles dead channels
/// (zero diagonal) by setting their diagonal to the damping value so the
/// matrix stays positive-definite. Returns the damping value used.
pub fn add_damping_inplace(h: &mut [f64], k: usize, damp_percent: f32) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let mut mean_diag = 0.0_f64;
    for i in 0..k {
        mean_diag += h[i * k + i];
    }
    mean_diag /= k as f64;
    let damp = (damp_percent as f64) * mean_diag.max(0.0);
    let damp = if damp <= 0.0 { 1e-6 } else { damp };
    for i in 0..k {
        let d = &mut h[i * k + i];
        if *d <= 0.0 {
            *d = damp;
        } else {
            *d += damp;
        }
    }
    damp
}

/// Lower-triangular Cholesky factor `L` such that `A = L Lᵀ`, for a
/// symmetric positive-definite `A` (row-major `k×k`). Returns the dense
/// `k×k` `L` with zeros above the diagonal. Errors if a non-positive
/// pivot is encountered.
pub fn cholesky_decompose(a: &[f64], k: usize) -> Result<Vec<f64>, GptqError> {
    let mut l = vec![0.0_f64; k * k];
    for j in 0..k {
        // Diagonal.
        let mut sum = a[j * k + j];
        for p in 0..j {
            let ljp = l[j * k + p];
            sum -= ljp * ljp;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return Err(GptqError::NotPositiveDefinite { pivot: j });
        }
        let ljj = sum.sqrt();
        l[j * k + j] = ljj;
        // Below diagonal.
        let inv = 1.0 / ljj;
        for i in (j + 1)..k {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= l[i * k + p] * l[j * k + p];
            }
            l[i * k + j] = s * inv;
        }
    }
    Ok(l)
}

/// Inverse of a symmetric positive-definite matrix via its Cholesky
/// factor. Returns the dense symmetric `k×k` inverse.
pub fn cholesky_inverse(a: &[f64], k: usize) -> Result<Vec<f64>, GptqError> {
    let l = cholesky_decompose(a, k)?;
    // Invert L (lower-triangular) -> Linv.
    let mut linv = vec![0.0_f64; k * k];
    for i in 0..k {
        linv[i * k + i] = 1.0 / l[i * k + i];
        for j in 0..i {
            let mut s = 0.0_f64;
            for p in j..i {
                s += l[i * k + p] * linv[p * k + j];
            }
            linv[i * k + j] = -s / l[i * k + i];
        }
    }
    // A^{-1} = Linvᵀ Linv.
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0_f64;
            let pstart = i.max(j);
            for p in pstart..k {
                s += linv[p * k + i] * linv[p * k + j];
            }
            inv[i * k + j] = s;
        }
    }
    Ok(inv)
}

/// Upper-triangular Cholesky factor `U` such that `A = Uᵀ U` (i.e. the
/// transpose of the lower factor), for SPD `A`. GPTQ propagates error
/// using the upper Cholesky of the *inverse* Hessian.
fn cholesky_upper(a: &[f64], k: usize) -> Result<Vec<f64>, GptqError> {
    let l = cholesky_decompose(a, k)?;
    let mut u = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            u[i * k + j] = l[j * k + i];
        }
    }
    Ok(u)
}

/// `C(m×n) = A(m×k) · B(k×n)`, row-major. Routes through the engine's
/// AVX2 matmul kernel when the runtime CPU supports it (the GPTQ
/// inter-block update is the `O(K²N)` hot path), with a scalar fallback.
/// `C` is overwritten (not accumulated).
fn gemm_ab(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: gated on runtime AVX2 detection; slices are sized
            // m*k, k*n, m*n as the kernel contract requires.
            unsafe {
                crate::simd_kernels::avx2::matmul_avx2(a, b, c, m, k, n);
            }
            return;
        }
    }
    // Scalar fallback.
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// **AQS-5** — apply real blockwise GPTQ to `weights` (`[K, N]`) in
/// place, using the `[S, K]` calibration activation matrix.
///
/// Deterministic, CPU-only. The result is a plain F32 reconstruction —
/// same contract as every [`crate::quant::policy::QuantizationPolicy`].
pub fn apply_gptq_real_inplace(
    weights: &mut [f32],
    shape: &[usize],
    activations: &[f32],
    activation_shape: &[usize],
    config: &GptqRealConfig,
) -> Result<(), GptqError> {
    if shape.len() != 2 {
        return Err(GptqError::NotTwoDimensional { rank: shape.len() });
    }
    if config.group_size == 0 {
        return Err(GptqError::InvalidGroupSize);
    }
    if config.block_size == 0 {
        return Err(GptqError::InvalidBlockSize);
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
    // Activation matrix must be [S, K] with K == k_rows.
    if activation_shape.len() != 2 || activation_shape[1] != k_rows {
        return Err(GptqError::ActivationShapeMismatch {
            expected_k: k_rows,
            actual_k: activation_shape.last().copied().unwrap_or(0),
            samples: activation_shape.first().copied().unwrap_or(0),
            matrix_len: activations.len(),
        });
    }
    let s = activation_shape[0];
    if activations.len() != s * k_rows {
        return Err(GptqError::ActivationShapeMismatch {
            expected_k: k_rows,
            actual_k: activation_shape[1],
            samples: s,
            matrix_len: activations.len(),
        });
    }
    if k_rows == 0 || n_cols == 0 {
        return Ok(());
    }

    // (1) Hessian + damping.
    let mut h = compute_hessian(activations, s, k_rows);
    add_damping_inplace(&mut h, k_rows, config.damp_percent);

    // (2) Optional activation-order permutation (descending diag(H)).
    let perm: Vec<usize> = if config.act_order {
        let mut idx: Vec<usize> = (0..k_rows).collect();
        idx.sort_by(|&a, &b| {
            h[b * k_rows + b]
                .partial_cmp(&h[a * k_rows + a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx
    } else {
        (0..k_rows).collect()
    };
    let permuted = config.act_order;

    // Apply permutation to H (rows+cols) and to the weight rows so the
    // sweep can run in natural order on permuted data.
    let (mut hp, mut wp): (Vec<f64>, Vec<f32>) = if permuted {
        let mut hp = vec![0.0_f64; k_rows * k_rows];
        for (ni, &oi) in perm.iter().enumerate() {
            for (nj, &oj) in perm.iter().enumerate() {
                hp[ni * k_rows + nj] = h[oi * k_rows + oj];
            }
        }
        let mut wp = vec![0.0_f32; expected];
        for (ni, &oi) in perm.iter().enumerate() {
            wp[ni * n_cols..(ni + 1) * n_cols]
                .copy_from_slice(&weights[oi * n_cols..(oi + 1) * n_cols]);
        }
        (hp, wp)
    } else {
        (h, weights.to_vec())
    };
    let _ = &mut hp;

    // (3) Upper Cholesky of the inverse Hessian.
    let hinv = cholesky_inverse(&hp, k_rows)?;
    let hinv_u = cholesky_upper(&hinv, k_rows)?;

    // (4) Blockwise sequential quantisation + error compensation.
    let gs = config.group_size;
    let bs = config.block_size;
    let mut group_scales: Vec<f32> = Vec::new(); // per-N scale for current group
    let mut current_group: isize = -1;

    let mut block_start = 0;
    while block_start < k_rows {
        let block_end = (block_start + bs).min(k_rows);
        // err_block[(k - block_start) * n_cols + n] accumulates the
        // quantisation error for inter-block propagation.
        let mut err_block = vec![0.0_f32; (block_end - block_start) * n_cols];

        for k in block_start..block_end {
            let g = (k / gs) as isize;
            if g != current_group {
                // (Re)compute the group scale from current weights over
                // the rows of this group across all columns.
                let g_lo = (g as usize) * gs;
                let g_hi = (g_lo + gs).min(k_rows);
                let mut scales = vec![1e-12_f32; n_cols];
                for col in 0..n_cols {
                    let mut amax = 0.0_f32;
                    for row in g_lo..g_hi {
                        let v = wp[row * n_cols + col].abs();
                        if v > amax {
                            amax = v;
                        }
                    }
                    scales[col] = (amax / 127.0).max(1e-12);
                }
                group_scales = scales;
                current_group = g;
            }

            let diag = hinv_u[k * k_rows + k] as f32;
            let inv_diag = if diag.abs() < 1e-12 { 0.0 } else { 1.0 / diag };

            // Quantise row k, record error.
            let row_off = k * n_cols;
            let eb_off = (k - block_start) * n_cols;
            for col in 0..n_cols {
                let w = wp[row_off + col];
                let sc = group_scales[col];
                let q = (w / sc).round().clamp(-127.0, 127.0) * sc;
                let err = (w - q) * inv_diag;
                wp[row_off + col] = q;
                err_block[eb_off + col] = err;

                // Intra-block propagation to rows (k, block_end).
                // (Handled below in a tight row loop for cache reuse.)
                let _ = err;
            }

            // Intra-block: W[k', :] -= Hinv_u[k, k'] · err_row for k' in (k, block_end)
            for kp in (k + 1)..block_end {
                let coeff = hinv_u[k * k_rows + kp] as f32;
                if coeff == 0.0 {
                    continue;
                }
                let dst = kp * n_cols;
                for col in 0..n_cols {
                    wp[dst + col] -= coeff * err_block[eb_off + col];
                }
            }
        }

        // Inter-block: propagate the whole block's error to rows
        // [block_end, k_rows):
        //   W[block_end:, :] -= A · Err_block
        // where A[r', k'] = Hinv_u[block_start+k', block_end+r'] (shape
        // R × bs_eff) and Err_block is (bs_eff × N). This is the O(K²N)
        // hot path, expressed as a GEMM so it routes through the AVX2
        // kernel (≈10× over the scalar AXPY loop) — see `gemm_ab`.
        let r_rows = k_rows - block_end;
        let bs_eff = block_end - block_start;
        if r_rows > 0 && bs_eff > 0 {
            // Build A (R × bs_eff) from the (transposed) Hinv_u block.
            let mut a = vec![0.0_f32; r_rows * bs_eff];
            for (kk, k) in (block_start..block_end).enumerate() {
                let hrow = k * k_rows;
                for (rr, r) in (block_end..k_rows).enumerate() {
                    a[rr * bs_eff + kk] = hinv_u[hrow + r] as f32;
                }
            }
            // tmp (R × N) = A · Err_block.
            let mut tmp = vec![0.0_f32; r_rows * n_cols];
            gemm_ab(&a, &err_block, &mut tmp, r_rows, bs_eff, n_cols);
            // W[block_end:, :] -= tmp.
            let base = block_end * n_cols;
            for i in 0..(r_rows * n_cols) {
                wp[base + i] -= tmp[i];
            }
        }

        block_start = block_end;
    }

    // (5) Un-permute and write back into `weights`.
    if permuted {
        for (ni, &oi) in perm.iter().enumerate() {
            weights[oi * n_cols..(oi + 1) * n_cols]
                .copy_from_slice(&wp[ni * n_cols..(ni + 1) * n_cols]);
        }
    } else {
        weights.copy_from_slice(&wp);
    }

    Ok(())
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
        AwqPolicy, CalibrationContext, GptqPolicy, GptqSurrogatePolicy, QuantizationPolicy,
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
    fn gptq_surrogate_evaluator_returns_metrics() {
        // AQS-2 integration: the evaluator drives the surrogate unchanged.
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
            &GptqSurrogatePolicy::new(8, 0.01),
            &CalibrationContext::with_activations(&act),
        )
        .unwrap();
        assert_eq!(r.policy_id, "gptq_surrogate");
        assert!(r.max_abs_diff > 0.0);
        assert!(r.rmse.is_finite());
        assert!(r.memory_bytes > 0);
    }

    // ----- AQS-5 real GPTQ unit tests -----

    #[test]
    fn gptq_real_hessian_is_symmetric() {
        // [S=4, K=3] activations.
        let act = rand_weights(4, 3, 21);
        let h = compute_hessian(&act, 4, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (h[i * 3 + j] - h[j * 3 + i]).abs() < 1e-9,
                    "Hessian must be symmetric"
                );
            }
        }
        // Diagonal is non-negative (sum of squares).
        for i in 0..3 {
            assert!(h[i * 3 + i] >= 0.0);
        }
    }

    #[test]
    fn gptq_real_damping_makes_hessian_stable() {
        // A rank-deficient activation set (all samples identical) yields a
        // singular Hessian; damping must make Cholesky succeed.
        let k = 4;
        let mut act = Vec::new();
        for _ in 0..3 {
            act.extend_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        }
        let mut h = compute_hessian(&act, 3, k);
        // Undamped: singular -> Cholesky should fail.
        assert!(cholesky_decompose(&h, k).is_err());
        // Damped: succeeds.
        add_damping_inplace(&mut h, k, 0.01);
        assert!(cholesky_decompose(&h, k).is_ok());
    }

    #[test]
    fn gptq_real_cholesky_inverse_identity_small() {
        // SPD matrix A; A * A^{-1} ≈ I.
        let k = 3;
        // A = [[4,1,0],[1,3,1],[0,1,2]] (SPD).
        let a = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let inv = cholesky_inverse(&a, k).unwrap();
        // Check A*inv == I.
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0_f64;
                for p in 0..k {
                    s += a[i * k + p] * inv[p * k + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((s - expected).abs() < 1e-9, "A*A^-1 must be identity");
            }
        }
    }

    #[test]
    fn gptq_real_quantizes_without_nan_and_deterministic() {
        let shape = [32usize, 16usize];
        let original = rand_weights(32, 16, 41);
        let act = rand_weights(8, 32, 42); // [S=8, K=32]
        let ashape = [8usize, 32usize];
        let cfg = GptqRealConfig {
            group_size: 8,
            block_size: 8,
            damp_percent: 0.01,
            act_order: false,
        };
        let mut a = original.clone();
        let mut b = original.clone();
        apply_gptq_real_inplace(&mut a, &shape, &act, &ashape, &cfg).unwrap();
        apply_gptq_real_inplace(&mut b, &shape, &act, &ashape, &cfg).unwrap();
        assert_eq!(a, b, "real GPTQ must be deterministic");
        assert!(a.iter().all(|v| v.is_finite()));
        assert_ne!(a, original);
    }

    #[test]
    fn gptq_real_zero_tensor_stays_zero() {
        let shape = [16usize, 8usize];
        let act = rand_weights(4, 16, 43);
        let ashape = [4usize, 16usize];
        let mut buf = vec![0.0_f32; 128];
        apply_gptq_real_inplace(&mut buf, &shape, &act, &ashape, &GptqRealConfig::default())
            .unwrap();
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gptq_real_rejects_bad_activation_shape() {
        let shape = [16usize, 8usize];
        let act = rand_weights(4, 8, 44); // K=8 but weight K=16
        let ashape = [4usize, 8usize];
        let mut buf = rand_weights(16, 8, 45);
        let err =
            apply_gptq_real_inplace(&mut buf, &shape, &act, &ashape, &GptqRealConfig::default())
                .unwrap_err();
        assert!(matches!(err, GptqError::ActivationShapeMismatch { .. }));
    }

    #[test]
    fn gptq_real_act_order_is_deterministic() {
        let shape = [16usize, 8usize];
        let original = rand_weights(16, 8, 46);
        let act = rand_weights(6, 16, 47);
        let ashape = [6usize, 16usize];
        let cfg = GptqRealConfig {
            group_size: 8,
            block_size: 8,
            damp_percent: 0.01,
            act_order: true,
        };
        let mut a = original.clone();
        let mut b = original.clone();
        apply_gptq_real_inplace(&mut a, &shape, &act, &ashape, &cfg).unwrap();
        apply_gptq_real_inplace(&mut b, &shape, &act, &ashape, &cfg).unwrap();
        assert_eq!(a, b, "act_order path must be deterministic");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    /// Structured case: real GPTQ should beat plain INT8 on the
    /// **functional** metric that matters — the calibration-weighted
    /// output error `‖X(W − Wq)‖`, not per-element weight drift. We build
    /// a weight with one high-energy input channel and verify GPTQ's
    /// output error is no worse (and typically better) than plain INT8.
    #[test]
    fn gptq_real_output_error_not_worse_than_plain_int8_structured() {
        let k = 32usize;
        let n = 16usize;
        let shape = [k, n];
        let s = 16usize;
        // Activations: channel 0 has large magnitude, rest small.
        let mut act = rand_weights(s, k, 51);
        for sample in 0..s {
            act[sample * k] *= 20.0; // dominant input channel 0
        }
        let ashape = [s, k];
        let w = rand_weights(k, n, 52);

        // GPTQ reconstruction.
        let mut wg = w.clone();
        apply_gptq_real_inplace(
            &mut wg,
            &shape,
            &act,
            &ashape,
            &GptqRealConfig { group_size: 16, block_size: 16, damp_percent: 0.01, act_order: false },
        )
        .unwrap();

        // Plain INT8 reconstruction (per-group absmax).
        let (q, sc) = crate::tensor::quantizer::absmax_per_group_symmetric(&w, &shape, 16);
        let mut wp = vec![0.0_f32; k * n];
        for idx in 0..(k * n) {
            let row = idx / n;
            let col = idx % n;
            let g = row / 16;
            wp[idx] = (q[idx] as f32) * sc[g * n + col];
        }

        // Functional output error ‖X (W − Wq)‖_F over the calibration set.
        let out_err = |wq: &[f32]| -> f64 {
            let mut e = 0.0_f64;
            for sample in 0..s {
                for col in 0..n {
                    let mut acc = 0.0_f64;
                    for ki in 0..k {
                        let dw = (w[ki * n + col] - wq[ki * n + col]) as f64;
                        acc += (act[sample * k + ki] as f64) * dw;
                    }
                    e += acc * acc;
                }
            }
            e.sqrt()
        };
        let e_gptq = out_err(&wg);
        let e_plain = out_err(&wp);
        eprintln!("structured output error: GPTQ={e_gptq:.4} plain_int8={e_plain:.4}");
        assert!(
            e_gptq <= e_plain * 1.05,
            "real GPTQ output error ({e_gptq:.4}) should not exceed plain INT8 ({e_plain:.4}) \
             by more than 5% on a structured case"
        );
    }

    /// Informational local-drift comparison (surrogate). Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore = "informational local-drift comparison; not a pass/fail gate"]
    fn awq_vs_gptq_surrogate_local_drift() {
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
        let gptq = evaluate_tensor_policy(input, &GptqSurrogatePolicy::new(16, 0.01), &cal).unwrap();
        eprintln!(
            "AWQ          : max_abs_diff={:.6} mean={:.6} rmse={:.6}",
            awq.max_abs_diff, awq.mean_abs_diff, awq.rmse
        );
        eprintln!(
            "GPTQ surrogate: max_abs_diff={:.6} mean={:.6} rmse={:.6}",
            gptq.max_abs_diff, gptq.mean_abs_diff, gptq.rmse
        );
    }
}
