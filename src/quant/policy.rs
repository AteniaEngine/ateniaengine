//! **AQS-1** — [`QuantizationPolicy`] trait + initial wrappers.
//!
//! The trait collapses the four existing experimental perturbation
//! helpers ([`crate::tensor::quantizer::absmax_per_group_symmetric`],
//! [`crate::tensor::quantizer::apply_awq_perturbation_inplace`],
//! [`crate::tensor::quantizer::apply_hybrid_awq_outlier_perturbation_inplace`])
//! and the trivial BF16 no-op into a single uniform interface that
//! AQS-2 (the policy evaluator) can drive without growing yet another
//! ad-hoc helper per axis.
//!
//! ## Surface
//!
//! ```text
//!   trait QuantizationPolicy {
//!       fn id(&self) -> &'static str;
//!       fn validate(&self, shape: &[usize]) -> Result<(), PolicyError>;
//!       fn apply_inplace(&self, weights: &mut [f32], shape: &[usize],
//!                        cal: &CalibrationContext<'_>) -> Result<(), PolicyError>;
//!       fn memory_bytes(&self, shape: &[usize]) -> u64;
//!   }
//! ```
//!
//! `apply_inplace` is the canonical AQS-1 surface: it perturbs the
//! F32 weight buffer in place so the runtime sees a plain F32 tensor
//! (drift attributable to quantisation is encoded in the buffer
//! delta). This matches the β-pivot.1 / β.5 / β-pivot.5 contracts and
//! avoids growing a second storage taxonomy purely for search.
//!
//! `memory_bytes` returns the **target** memory cost — i.e. what the
//! policy *would* consume if AQS-2 were to materialise it under its
//! real storage. AQS-1 does not realise these stores yet; the numbers
//! are an approximation used by the search cost model only.
//!
//! ## Non-goals
//!
//! * No GPTQ implementation — see [`GptqPolicy`] placeholder.
//! * No tier-planner integration.
//! * No mutation of the productive INT8 path; M9 callers continue to
//!   reach `quantize_int8_per_group` directly.
//! * No global registry / discovery. AQS-2 will own that surface.

use crate::tensor::quantizer::{
    absmax_per_group_symmetric, apply_awq_perturbation_inplace,
    apply_hybrid_awq_outlier_perturbation_inplace, AwqError,
};

// ============================================================================
// Calibration context
// ============================================================================

/// Read-only calibration metadata threaded through
/// [`QuantizationPolicy::apply_inplace`].
///
/// Policies that need activation statistics (currently [`AwqPolicy`]
/// and [`HybridPolicy`]) consume `activation_absmax`; the rest ignore
/// it. The struct is borrow-only so AQS-2 can reuse one capture across
/// many candidate policies without cloning per call.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationContext<'a> {
    /// Per-K-row activation absmax, length `shape[0]` (the input-
    /// channel dimension after the loader's HF transpose). `None`
    /// when the policy does not need calibration. Used by AWQ /
    /// Hybrid / the AQS-3 GPTQ surrogate.
    pub activation_absmax: Option<&'a [f32]>,
    /// **AQS-5** — full calibration activation matrix, row-major
    /// `[S, K]` (S samples × K input channels). Required by the real
    /// GPTQ policy ([`GptqPolicy`]) to form the `K×K` Hessian
    /// `H = XᵀX / S`. `None` for every other policy.
    pub activation_matrix: Option<&'a [f32]>,
    /// Shape `[S, K]` describing [`Self::activation_matrix`]. Must be
    /// present iff `activation_matrix` is.
    pub activation_shape: Option<&'a [usize]>,
    /// Optional 32-byte digest of the calibration corpus. Reserved
    /// for AQS manifest provenance.
    pub corpus_hash: Option<[u8; 32]>,
}

impl<'a> CalibrationContext<'a> {
    /// An empty context — for policies that take no calibration.
    pub fn empty() -> Self {
        Self {
            activation_absmax: None,
            activation_matrix: None,
            activation_shape: None,
            corpus_hash: None,
        }
    }

    /// Build a context carrying only per-K activation absmax.
    pub fn with_activations(act_absmax: &'a [f32]) -> Self {
        Self {
            activation_absmax: Some(act_absmax),
            activation_matrix: None,
            activation_shape: None,
            corpus_hash: None,
        }
    }

    /// **AQS-5** — build a context carrying a full `[S, K]` activation
    /// matrix (for real GPTQ).
    pub fn with_activation_matrix(matrix: &'a [f32], shape: &'a [usize]) -> Self {
        Self {
            activation_absmax: None,
            activation_matrix: Some(matrix),
            activation_shape: Some(shape),
            corpus_hash: None,
        }
    }
}

// ============================================================================
// Error model
// ============================================================================

/// Errors raised by [`QuantizationPolicy::validate`] and
/// [`QuantizationPolicy::apply_inplace`].
///
/// Mirrors the surface of [`AwqError`] /
/// [`crate::tensor::quantizer::OutlierDecompositionError`] but at the
/// policy abstraction level — callers should not depend on the inner
/// helpers' error types because AQS-2 may swap them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// Shape rank is not 2D. All AQS-1 policies work on
    /// `[K_in, N_out]` linear weights only.
    NotTwoDimensional { rank: usize },
    /// `weights.len()` does not match `shape[0] * shape[1]`.
    LengthMismatch { expected: usize, actual: usize },
    /// `group_size == 0`.
    InvalidGroupSize,
    /// `outlier_k > shape[1]` — cannot carve out more columns than
    /// the matrix has.
    OutlierKExceedsColumns { k: usize, n: usize },
    /// `alpha` outside `[0, 1]` or non-finite.
    InvalidAlpha,
    /// `damp_percent` outside `[0, 1)` or non-finite (GPTQ only).
    InvalidDampPercent,
    /// The policy needs `CalibrationContext::activation_absmax` but
    /// the caller passed `None`.
    MissingActivationStats,
    /// `activation_absmax.len()` does not match `shape[0]`.
    ActivationStatsLengthMismatch { expected: usize, actual: usize },
    /// The underlying helper failed for a reason that the policy
    /// abstraction does not map to a more specific variant.
    InnerError(String),
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyError::NotTwoDimensional { rank } => {
                write!(f, "policy: shape must be 2D (got rank {rank})")
            }
            PolicyError::LengthMismatch { expected, actual } => write!(
                f,
                "policy: weights.len() = {actual} does not match K * N = {expected}"
            ),
            PolicyError::InvalidGroupSize => write!(f, "policy: group_size must be > 0"),
            PolicyError::OutlierKExceedsColumns { k, n } => {
                write!(f, "policy: outlier_k = {k} exceeds N = {n}")
            }
            PolicyError::InvalidAlpha => {
                write!(f, "policy: alpha must be finite in [0, 1]")
            }
            PolicyError::InvalidDampPercent => {
                write!(f, "policy: damp_percent must be finite in [0, 1)")
            }
            PolicyError::MissingActivationStats => write!(
                f,
                "policy: this policy requires CalibrationContext::activation_absmax"
            ),
            PolicyError::ActivationStatsLengthMismatch { expected, actual } => write!(
                f,
                "policy: activation_absmax length = {actual} does not match K = {expected}"
            ),
            PolicyError::InnerError(msg) => write!(f, "policy: inner helper error: {msg}"),
        }
    }
}

impl std::error::Error for PolicyError {}

impl From<AwqError> for PolicyError {
    fn from(value: AwqError) -> Self {
        // Map by debug formatting to avoid hard-coupling to AwqError's
        // private variant layout; AQS-1 only needs to surface the message.
        PolicyError::InnerError(format!("{value:?}"))
    }
}

impl From<crate::quant::gptq::GptqError> for PolicyError {
    fn from(value: crate::quant::gptq::GptqError) -> Self {
        use crate::quant::gptq::GptqError as G;
        match value {
            G::NotTwoDimensional { rank } => PolicyError::NotTwoDimensional { rank },
            G::LengthMismatch { expected, actual } => {
                PolicyError::LengthMismatch { expected, actual }
            }
            G::InvalidGroupSize => PolicyError::InvalidGroupSize,
            G::InvalidDampPercent => PolicyError::InvalidDampPercent,
            G::HessianLengthMismatch { expected, actual } => {
                // Hessian length derives directly from the activation
                // stats length, so surface it as the activation mismatch.
                PolicyError::ActivationStatsLengthMismatch { expected, actual }
            }
            G::InvalidBlockSize => PolicyError::InvalidGroupSize,
            G::ActivationShapeMismatch {
                expected_k, actual_k, ..
            } => PolicyError::ActivationStatsLengthMismatch {
                expected: expected_k,
                actual: actual_k,
            },
            G::NotPositiveDefinite { pivot } => {
                PolicyError::InnerError(format!("GPTQ Hessian not SPD at pivot {pivot}"))
            }
        }
    }
}

// ============================================================================
// Shared validation helpers
// ============================================================================

/// Validate that `shape` is exactly 2D and that `weights_len`
/// matches `K * N`. Returns `(K, N)` on success.
fn check_shape_2d(
    shape: &[usize],
    weights_len: usize,
) -> Result<(usize, usize), PolicyError> {
    if shape.len() != 2 {
        return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
    }
    let (k, n) = (shape[0], shape[1]);
    let expected = k * n;
    if weights_len != expected {
        return Err(PolicyError::LengthMismatch {
            expected,
            actual: weights_len,
        });
    }
    Ok((k, n))
}

/// Validate that `cal.activation_absmax` is present and matches `K`.
fn check_activation_stats<'a>(
    cal: &CalibrationContext<'a>,
    k: usize,
) -> Result<&'a [f32], PolicyError> {
    let stats = cal
        .activation_absmax
        .ok_or(PolicyError::MissingActivationStats)?;
    if stats.len() != k {
        return Err(PolicyError::ActivationStatsLengthMismatch {
            expected: k,
            actual: stats.len(),
        });
    }
    Ok(stats)
}

/// Number of K-axis groups for a given shape + group_size.
fn num_groups(k: usize, group_size: usize) -> usize {
    if k == 0 || group_size == 0 {
        0
    } else {
        (k + group_size - 1) / group_size
    }
}

// ============================================================================
// QuantizationPolicy trait
// ============================================================================

/// Uniform interface over the AQS-1 experimental quantisation
/// strategies. See module docs for scope and the audit document for
/// design rationale.
pub trait QuantizationPolicy {
    /// Stable identifier used by AQS-2 manifests (`"bf16"`,
    /// `"plain_int8_g128"`, `"awq_a0.30_g128"`, …). MUST be stable
    /// across runs for the same configuration.
    fn id(&self) -> &'static str;

    /// Cheap structural validation — does not touch weights.
    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError>;

    /// Perturb `weights` in place according to the policy. The
    /// resulting buffer is still F32; the runtime is unchanged.
    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError>;

    /// Approximate target memory cost in bytes if this policy were
    /// materialised under its native storage. AQS-2 search cost model
    /// consumes this — AQS-1 does not store the result anywhere.
    fn memory_bytes(&self, shape: &[usize]) -> u64;
}

// ============================================================================
// Policy 1: Bf16Fallback
// ============================================================================

/// Identity policy used as the "no quantisation" baseline. The
/// in-place apply is a literal no-op; the only thing the wrapper
/// adds is the memory-cost estimate (BF16 = 2 bytes per element)
/// and the stable id.
///
/// This exists so AQS-2 can score a "do nothing" policy under the
/// same interface as the lossy ones.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bf16Fallback;

impl QuantizationPolicy for Bf16Fallback {
    fn id(&self) -> &'static str {
        "bf16"
    }

    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError> {
        if shape.len() != 2 {
            return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
        }
        Ok(())
    }

    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        _cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError> {
        let _ = check_shape_2d(shape, weights.len())?;
        Ok(())
    }

    fn memory_bytes(&self, shape: &[usize]) -> u64 {
        let numel: usize = shape.iter().product();
        (numel as u64) * 2
    }
}

// ============================================================================
// Policy 2: PlainInt8
// ============================================================================

/// M9 per-group absmax INT8, exposed through the policy interface.
/// Quantises with [`absmax_per_group_symmetric`] and dequantises
/// back into the F32 buffer — identical numerics to the productive
/// M9 path, only without materialising a `CpuInt8` storage.
#[derive(Debug, Clone, Copy)]
pub struct PlainInt8 {
    pub group_size: usize,
}

impl PlainInt8 {
    pub fn new(group_size: usize) -> Self {
        Self { group_size }
    }
}

impl QuantizationPolicy for PlainInt8 {
    fn id(&self) -> &'static str {
        "plain_int8"
    }

    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError> {
        if self.group_size == 0 {
            return Err(PolicyError::InvalidGroupSize);
        }
        if shape.len() != 2 {
            return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
        }
        Ok(())
    }

    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        _cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError> {
        if self.group_size == 0 {
            return Err(PolicyError::InvalidGroupSize);
        }
        let (k, n) = check_shape_2d(shape, weights.len())?;
        if k == 0 || n == 0 {
            return Ok(());
        }
        let (q, scales) = absmax_per_group_symmetric(weights, shape, self.group_size);
        for idx in 0..weights.len() {
            let row = idx / n;
            let col = idx % n;
            let g = row / self.group_size;
            weights[idx] = (q[idx] as f32) * scales[g * n + col];
        }
        Ok(())
    }

    fn memory_bytes(&self, shape: &[usize]) -> u64 {
        let numel: usize = shape.iter().product();
        let n = *shape.last().unwrap_or(&0);
        let k = if shape.len() >= 2 { shape[0] } else { 0 };
        let g = num_groups(k, self.group_size);
        (numel as u64) + (g as u64) * (n as u64) * 4
    }
}

// ============================================================================
// Policy 3: AwqPolicy
// ============================================================================

/// β-pivot.2 calibrated AWQ: per-K-row activation-aware scaling
/// applied around a plain per-group absmax INT8 quantise/dequantise.
/// Requires [`CalibrationContext::activation_absmax`].
#[derive(Debug, Clone, Copy)]
pub struct AwqPolicy {
    pub group_size: usize,
    pub alpha: f32,
}

impl AwqPolicy {
    pub fn new(group_size: usize, alpha: f32) -> Self {
        Self { group_size, alpha }
    }
}

impl QuantizationPolicy for AwqPolicy {
    fn id(&self) -> &'static str {
        "awq"
    }

    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError> {
        if self.group_size == 0 {
            return Err(PolicyError::InvalidGroupSize);
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(PolicyError::InvalidAlpha);
        }
        if shape.len() != 2 {
            return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
        }
        Ok(())
    }

    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError> {
        self.validate(shape)?;
        let (k, _n) = check_shape_2d(shape, weights.len())?;
        let stats = check_activation_stats(cal, k)?;
        let scales =
            crate::tensor::quantizer::awq_per_row_scales_from_activations(stats, self.alpha)?;
        apply_awq_perturbation_inplace(weights, shape, self.group_size, &scales)?;
        Ok(())
    }

    fn memory_bytes(&self, shape: &[usize]) -> u64 {
        // Same as PlainInt8 for now — the AWQ scales themselves are
        // fold-in (the buffer ends up plain F32 after apply) so they
        // do not consume runtime memory. AQS-2 may revisit this once
        // AWQ acquires a dedicated storage with externalised scales.
        PlainInt8::new(self.group_size).memory_bytes(shape)
    }
}

// ============================================================================
// Policy 4: HybridPolicy
// ============================================================================

/// β-pivot.5 hybrid: AWQ pre-scale + top-k outlier carve-out around
/// the per-group absmax INT8. Requires
/// [`CalibrationContext::activation_absmax`].
///
/// The audit (`docs/AQS_ARCHITECTURE_AUDIT.md`) documents that this
/// policy was empirically *worse* than [`AwqPolicy`] alone on
/// TinyLlama; it is wrapped for completeness and so AQS-2 can confirm
/// that result across other models without re-implementing the math.
#[derive(Debug, Clone, Copy)]
pub struct HybridPolicy {
    pub group_size: usize,
    pub alpha: f32,
    pub outlier_k: usize,
}

impl HybridPolicy {
    pub fn new(group_size: usize, alpha: f32, outlier_k: usize) -> Self {
        Self {
            group_size,
            alpha,
            outlier_k,
        }
    }
}

impl QuantizationPolicy for HybridPolicy {
    fn id(&self) -> &'static str {
        "hybrid_awq_outlier"
    }

    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError> {
        if self.group_size == 0 {
            return Err(PolicyError::InvalidGroupSize);
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(PolicyError::InvalidAlpha);
        }
        if shape.len() != 2 {
            return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
        }
        let n = shape[1];
        if self.outlier_k > n {
            return Err(PolicyError::OutlierKExceedsColumns {
                k: self.outlier_k,
                n,
            });
        }
        Ok(())
    }

    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError> {
        self.validate(shape)?;
        let (k, _n) = check_shape_2d(shape, weights.len())?;
        let stats = check_activation_stats(cal, k)?;
        let scales =
            crate::tensor::quantizer::awq_per_row_scales_from_activations(stats, self.alpha)?;
        apply_hybrid_awq_outlier_perturbation_inplace(
            weights,
            shape,
            self.group_size,
            &scales,
            self.outlier_k,
        )?;
        Ok(())
    }

    fn memory_bytes(&self, shape: &[usize]) -> u64 {
        // PlainInt8 base + F32 sidecar (K * outlier_k * 4 bytes).
        let base = PlainInt8::new(self.group_size).memory_bytes(shape);
        let k = if shape.len() >= 2 { shape[0] } else { 0 };
        base + (k as u64) * (self.outlier_k as u64) * 4
    }
}

// ============================================================================
// Policy 5a: GptqSurrogatePolicy (AQS-3 diagonal surrogate — kept for
// comparison; AQS-4 showed it fails end-to-end, see HANDOFF_AQS_4.md)
// ============================================================================

/// **AQS-3** — simplified, CPU-only GPTQ *surrogate*: diagonal Hessian +
/// Hessian-regularised scale search + sequential error diffusion.
///
/// Retained under its own name after AQS-5 because AQS-4 measured it at
/// 12.5 end-to-end drift on TinyLlama (worse than plain INT8) — it is NOT
/// representative of real GPTQ and exists only as a comparison baseline.
/// Requires [`CalibrationContext::activation_absmax`] (length `K`).
#[derive(Debug, Clone, Copy)]
pub struct GptqSurrogatePolicy {
    pub group_size: usize,
    pub damp_percent: f32,
}

impl Default for GptqSurrogatePolicy {
    fn default() -> Self {
        Self {
            group_size: 128,
            damp_percent: 0.01,
        }
    }
}

impl GptqSurrogatePolicy {
    pub fn new(group_size: usize, damp_percent: f32) -> Self {
        Self {
            group_size,
            damp_percent,
        }
    }
}

impl QuantizationPolicy for GptqSurrogatePolicy {
    fn id(&self) -> &'static str {
        "gptq_surrogate"
    }

    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError> {
        if self.group_size == 0 {
            return Err(PolicyError::InvalidGroupSize);
        }
        if !self.damp_percent.is_finite()
            || self.damp_percent < 0.0
            || self.damp_percent >= 1.0
        {
            return Err(PolicyError::InvalidDampPercent);
        }
        if shape.len() != 2 {
            return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
        }
        Ok(())
    }

    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError> {
        self.validate(shape)?;
        let (k, _n) = check_shape_2d(shape, weights.len())?;
        let stats = check_activation_stats(cal, k)?;
        let hessian = crate::quant::gptq::approximate_hessian_diag(stats, &[1, k]);
        let cfg = crate::quant::gptq::GptqConfig {
            group_size: self.group_size,
            damp_percent: self.damp_percent,
        };
        crate::quant::gptq::apply_gptq_reconstruction_inplace(weights, shape, &hessian, &cfg)?;
        Ok(())
    }

    fn memory_bytes(&self, shape: &[usize]) -> u64 {
        crate::quant::gptq::gptq_memory_bytes(shape, self.group_size.max(1))
    }
}

// ============================================================================
// Policy 5: GptqPolicy (AQS-5 — REAL blockwise GPTQ)
// ============================================================================

/// **AQS-5** — real, CPU-only blockwise GPTQ: full `K×K` Hessian from a
/// calibration activation matrix, Tikhonov damping, Cholesky-based
/// inverse-Hessian error compensation, dynamic per-group INT8 scales.
///
/// Requires [`CalibrationContext::activation_matrix`] + `activation_shape`
/// (`[S, K]`). This is the genuine OBQ/GPTQ algorithm — see the
/// `crate::quant::gptq` module docs for the math and layout. The runtime
/// sees a plain F32 reconstruction afterwards.
#[derive(Debug, Clone, Copy)]
pub struct GptqPolicy {
    pub group_size: usize,
    pub block_size: usize,
    pub damp_percent: f32,
    pub act_order: bool,
}

impl Default for GptqPolicy {
    fn default() -> Self {
        Self {
            group_size: 128,
            block_size: 128,
            damp_percent: 0.01,
            act_order: false,
        }
    }
}

impl GptqPolicy {
    /// Construct with the two most common knobs; `block_size = 128`,
    /// `act_order = false` (the AutoGPTQ defaults).
    pub fn new(group_size: usize, damp_percent: f32) -> Self {
        Self {
            group_size,
            block_size: 128,
            damp_percent,
            act_order: false,
        }
    }

    /// Full constructor.
    pub fn with_config(
        group_size: usize,
        block_size: usize,
        damp_percent: f32,
        act_order: bool,
    ) -> Self {
        Self {
            group_size,
            block_size,
            damp_percent,
            act_order,
        }
    }
}

impl QuantizationPolicy for GptqPolicy {
    fn id(&self) -> &'static str {
        "gptq"
    }

    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError> {
        if self.group_size == 0 || self.block_size == 0 {
            return Err(PolicyError::InvalidGroupSize);
        }
        if !self.damp_percent.is_finite()
            || self.damp_percent < 0.0
            || self.damp_percent >= 1.0
        {
            return Err(PolicyError::InvalidDampPercent);
        }
        if shape.len() != 2 {
            return Err(PolicyError::NotTwoDimensional { rank: shape.len() });
        }
        Ok(())
    }

    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError> {
        self.validate(shape)?;
        let (_k, _n) = check_shape_2d(shape, weights.len())?;
        // Real GPTQ needs the full [S, K] activation matrix.
        let (matrix, mshape) = match (cal.activation_matrix, cal.activation_shape) {
            (Some(m), Some(s)) => (m, s),
            _ => return Err(PolicyError::MissingActivationStats),
        };
        let cfg = crate::quant::gptq::GptqRealConfig {
            group_size: self.group_size,
            block_size: self.block_size,
            damp_percent: self.damp_percent,
            act_order: self.act_order,
        };
        crate::quant::gptq::apply_gptq_real_inplace(weights, shape, matrix, mshape, &cfg)?;
        Ok(())
    }

    fn memory_bytes(&self, shape: &[usize]) -> u64 {
        crate::quant::gptq::gptq_memory_bytes(shape, self.group_size.max(1))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_weights(k: usize, n: usize, seed: u64) -> Vec<f32> {
        // Deterministic xorshift — keeps tests dep-free.
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

    fn assert_finite(buf: &[f32]) {
        for &v in buf {
            assert!(v.is_finite(), "non-finite value in buffer");
        }
    }

    #[test]
    fn bf16_policy_is_noop() {
        let policy = Bf16Fallback;
        let shape = [16, 32];
        let original = rand_weights(16, 32, 1);
        let mut buf = original.clone();
        policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::empty())
            .unwrap();
        assert_eq!(buf, original);
        assert_eq!(policy.id(), "bf16");
        assert_eq!(policy.memory_bytes(&shape), (16 * 32) as u64 * 2);
    }

    #[test]
    fn plain_int8_policy_changes_weights_deterministically() {
        let policy = PlainInt8::new(8);
        let shape = [32, 16];
        let original = rand_weights(32, 16, 2);
        let mut buf_a = original.clone();
        let mut buf_b = original.clone();
        policy
            .apply_inplace(&mut buf_a, &shape, &CalibrationContext::empty())
            .unwrap();
        policy
            .apply_inplace(&mut buf_b, &shape, &CalibrationContext::empty())
            .unwrap();
        assert_eq!(buf_a, buf_b, "PlainInt8 must be deterministic");
        assert_ne!(buf_a, original, "PlainInt8 must perturb the buffer");
        assert_finite(&buf_a);
    }

    #[test]
    fn awq_policy_requires_activation_stats() {
        let policy = AwqPolicy::new(8, 0.3);
        let shape = [16, 16];
        let mut buf = rand_weights(16, 16, 3);
        let err = policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::empty())
            .unwrap_err();
        assert_eq!(err, PolicyError::MissingActivationStats);
    }

    #[test]
    fn awq_policy_applies_with_valid_stats() {
        let policy = AwqPolicy::new(8, 0.3);
        let shape = [16, 16];
        let original = rand_weights(16, 16, 4);
        let act = vec![0.5_f32; 16];
        let mut buf = original.clone();
        policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::with_activations(&act))
            .unwrap();
        assert_finite(&buf);
        // Buffer should differ from original (quantisation noise is real).
        let diff: f32 = original
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "AWQ apply should perturb the buffer");
    }

    #[test]
    fn awq_policy_rejects_wrong_activation_length() {
        let policy = AwqPolicy::new(8, 0.3);
        let shape = [16, 16];
        let act = vec![0.5_f32; 8]; // wrong length
        let mut buf = rand_weights(16, 16, 5);
        let err = policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::with_activations(&act))
            .unwrap_err();
        assert!(matches!(
            err,
            PolicyError::ActivationStatsLengthMismatch { .. }
        ));
    }

    #[test]
    fn hybrid_policy_requires_activation_stats() {
        let policy = HybridPolicy::new(8, 0.3, 2);
        let shape = [16, 16];
        let mut buf = rand_weights(16, 16, 6);
        let err = policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::empty())
            .unwrap_err();
        assert_eq!(err, PolicyError::MissingActivationStats);
    }

    #[test]
    fn hybrid_policy_applies_with_valid_stats() {
        let policy = HybridPolicy::new(8, 0.3, 2);
        let shape = [16, 16];
        let original = rand_weights(16, 16, 7);
        let act = vec![0.5_f32; 16];
        let mut buf = original.clone();
        policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::with_activations(&act))
            .unwrap();
        assert_finite(&buf);
    }

    #[test]
    fn policy_rejects_non_2d_shape() {
        let policy = PlainInt8::new(8);
        let shape = [4, 4, 4];
        let mut buf = vec![0.0_f32; 64];
        let err = policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::empty())
            .unwrap_err();
        assert!(matches!(err, PolicyError::NotTwoDimensional { rank: 3 }));

        assert!(matches!(
            Bf16Fallback.validate(&[1, 2, 3]).unwrap_err(),
            PolicyError::NotTwoDimensional { .. }
        ));
        assert!(matches!(
            AwqPolicy::new(8, 0.3).validate(&[1, 2, 3]).unwrap_err(),
            PolicyError::NotTwoDimensional { .. }
        ));
        assert!(matches!(
            HybridPolicy::new(8, 0.3, 1).validate(&[1, 2, 3]).unwrap_err(),
            PolicyError::NotTwoDimensional { .. }
        ));
    }

    #[test]
    fn policy_rejects_bad_group_size() {
        let shape = [16, 16];
        assert_eq!(
            PlainInt8::new(0).validate(&shape).unwrap_err(),
            PolicyError::InvalidGroupSize
        );
        assert_eq!(
            AwqPolicy::new(0, 0.3).validate(&shape).unwrap_err(),
            PolicyError::InvalidGroupSize
        );
        assert_eq!(
            HybridPolicy::new(0, 0.3, 1).validate(&shape).unwrap_err(),
            PolicyError::InvalidGroupSize
        );
    }

    #[test]
    fn policy_rejects_bad_alpha() {
        let shape = [16, 16];
        assert_eq!(
            AwqPolicy::new(8, -0.1).validate(&shape).unwrap_err(),
            PolicyError::InvalidAlpha
        );
        assert_eq!(
            AwqPolicy::new(8, 1.5).validate(&shape).unwrap_err(),
            PolicyError::InvalidAlpha
        );
        assert_eq!(
            AwqPolicy::new(8, f32::NAN).validate(&shape).unwrap_err(),
            PolicyError::InvalidAlpha
        );
    }

    #[test]
    fn hybrid_policy_rejects_excessive_outlier_k() {
        let shape = [16, 8];
        let err = HybridPolicy::new(8, 0.3, 16).validate(&shape).unwrap_err();
        assert!(matches!(
            err,
            PolicyError::OutlierKExceedsColumns { k: 16, n: 8 }
        ));
    }

    #[test]
    fn policy_memory_bytes_are_monotonic_reasonable() {
        let shape = [1024, 1024];
        let bf16 = Bf16Fallback.memory_bytes(&shape);
        let int8 = PlainInt8::new(128).memory_bytes(&shape);
        let awq = AwqPolicy::new(128, 0.3).memory_bytes(&shape);
        let hybrid = HybridPolicy::new(128, 0.3, 4).memory_bytes(&shape);

        // BF16 = 2 bytes/elem; INT8 should be lighter.
        assert!(int8 < bf16, "INT8 ({int8}) must be smaller than BF16 ({bf16})");
        // AWQ identical to PlainInt8 at this milestone.
        assert_eq!(awq, int8);
        // Hybrid adds outlier sidecar — strictly heavier than plain INT8.
        assert!(hybrid > int8);
        // But still under BF16 for a small outlier_k.
        assert!(hybrid < bf16);
    }

    #[test]
    fn policy_ids_are_stable() {
        assert_eq!(Bf16Fallback.id(), "bf16");
        assert_eq!(PlainInt8::new(128).id(), "plain_int8");
        assert_eq!(AwqPolicy::new(128, 0.3).id(), "awq");
        assert_eq!(HybridPolicy::new(128, 0.3, 4).id(), "hybrid_awq_outlier");
        assert_eq!(GptqPolicy::default().id(), "gptq");
        assert_eq!(GptqSurrogatePolicy::default().id(), "gptq_surrogate");
    }

    #[test]
    fn gptq_policy_requires_activation_matrix() {
        // Real GPTQ needs the full [S, K] matrix; absmax-only or empty
        // both surface as MissingActivationStats.
        let policy = GptqPolicy::new(8, 0.01);
        let shape = [16, 16];
        let mut buf = rand_weights(16, 16, 30);
        let err = policy
            .apply_inplace(&mut buf, &shape, &CalibrationContext::empty())
            .unwrap_err();
        assert_eq!(err, PolicyError::MissingActivationStats);

        let absmax = vec![0.5_f32; 16];
        let err2 = policy
            .apply_inplace(
                &mut buf,
                &shape,
                &CalibrationContext::with_activations(&absmax),
            )
            .unwrap_err();
        assert_eq!(err2, PolicyError::MissingActivationStats);
    }

    #[test]
    fn gptq_policy_uses_real_gptq_with_matrix() {
        // K=16, S=8 activation matrix → real GPTQ runs and perturbs.
        let policy = GptqPolicy::new(8, 0.01);
        let shape = [16, 16];
        let original = rand_weights(16, 16, 31);
        let act = rand_weights(8, 16, 71); // [S=8, K=16]
        let ashape = [8usize, 16usize];
        let mut a = original.clone();
        let mut b = original.clone();
        policy
            .apply_inplace(
                &mut a,
                &shape,
                &CalibrationContext::with_activation_matrix(&act, &ashape),
            )
            .unwrap();
        policy
            .apply_inplace(
                &mut b,
                &shape,
                &CalibrationContext::with_activation_matrix(&act, &ashape),
            )
            .unwrap();
        assert_eq!(a, b, "real GPTQ policy must be deterministic");
        assert_ne!(a, original, "real GPTQ must perturb the buffer");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gptq_surrogate_policy_applies_deterministically() {
        let policy = GptqSurrogatePolicy::new(8, 0.01);
        let shape = [32, 16];
        let original = rand_weights(32, 16, 31);
        let act = vec![0.5_f32; 32];
        let mut a = original.clone();
        let mut b = original.clone();
        policy
            .apply_inplace(&mut a, &shape, &CalibrationContext::with_activations(&act))
            .unwrap();
        policy
            .apply_inplace(&mut b, &shape, &CalibrationContext::with_activations(&act))
            .unwrap();
        assert_eq!(a, b, "GPTQ surrogate must be deterministic");
        assert_ne!(a, original);
    }

    #[test]
    fn gptq_policy_reports_memory_bytes() {
        let policy = GptqPolicy::new(16, 0.01);
        let shape = [64, 64];
        let bytes = policy.memory_bytes(&shape);
        // INT8 payload + scales + hessian metadata; lighter than F32.
        assert!(bytes > 0);
        assert!(bytes < (64 * 64) as u64 * 4);
    }

    #[test]
    fn gptq_policy_rejects_bad_group_size() {
        assert_eq!(
            GptqPolicy::new(0, 0.01).validate(&[16, 16]).unwrap_err(),
            PolicyError::InvalidGroupSize
        );
    }

    #[test]
    fn gptq_policy_rejects_bad_damp_percent() {
        assert_eq!(
            GptqPolicy::new(8, 1.0).validate(&[16, 16]).unwrap_err(),
            PolicyError::InvalidDampPercent
        );
        assert_eq!(
            GptqPolicy::new(8, -0.1).validate(&[16, 16]).unwrap_err(),
            PolicyError::InvalidDampPercent
        );
    }

    #[test]
    fn gptq_policy_reduces_to_identity_on_zero_tensor() {
        // Real GPTQ on a zero weight tensor must stay zero.
        let policy = GptqPolicy::new(8, 0.01);
        let shape = [16, 16];
        let act = rand_weights(8, 16, 73);
        let ashape = [8usize, 16usize];
        let mut buf = vec![0.0_f32; 256];
        policy
            .apply_inplace(
                &mut buf,
                &shape,
                &CalibrationContext::with_activation_matrix(&act, &ashape),
            )
            .unwrap();
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dyn_dispatch_works() {
        // Smoke test: the trait is object-safe and can be stored in
        // a `Vec<Box<dyn QuantizationPolicy>>`. AQS-2 will rely on
        // this property to drive the search.
        let shape = [16, 16];
        let policies: Vec<Box<dyn QuantizationPolicy>> = vec![
            Box::new(Bf16Fallback),
            Box::new(PlainInt8::new(8)),
            Box::new(AwqPolicy::new(8, 0.3)),
            Box::new(HybridPolicy::new(8, 0.3, 2)),
        ];
        for p in &policies {
            assert!(p.validate(&shape).is_ok(), "{} validate failed", p.id());
            let _ = p.memory_bytes(&shape);
        }
    }
}
