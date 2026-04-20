//! Simple arithmetic estimator for static memory footprint of tensor operations.
//!
//! As of APX v18, the forecaster also exposes an optional real VRAM signal
//! read via [`crate::amm::vram_probe`]. That signal is independent from the
//! existing byte counter: it reports what the hardware currently sees, not
//! what Atenia has been told it allocated. Both coexist; neither replaces
//! the other.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::amm::vram_probe::{self, VramProbeError};
use crate::tensor::tensor::Tensor;

/// Tracks current and predicted memory usage for upcoming tensor operations.
#[derive(Default)]
pub struct MemoryForecaster {
    pub current_bytes: usize,
    pub predicted_next_bytes: usize,
    /// Set to `true` the first time a VRAM probe call on this instance fails.
    /// Used to emit the "probe unavailable" warning at most once per instance.
    vram_probe_failed: AtomicBool,
}

impl MemoryForecaster {
    /// Creates a new memory forecaster with zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tensor into the forecaster, contributing to current usage.
    pub fn register_tensor(&mut self, tensor: &Tensor) {
        self.current_bytes += tensor.estimated_bytes();
    }

    /// Estimates the memory footprint of a binary tensor operation by summing operand sizes.
    ///
    /// Assumes the result tensor shares `a`'s storage footprint. Not a predictive model:
    /// this is static arithmetic over declared tensor sizes, with no runtime signals.
    pub fn predict_add_operation(&mut self, a: &Tensor, b: &Tensor) {
        self.predicted_next_bytes =
            a.estimated_bytes() + b.estimated_bytes() + a.estimated_bytes();
    }

    /// Returns `true` if the predicted memory allocation would exceed the provided limit.
    pub fn is_over_limit(&self, limit_bytes: usize) -> bool {
        self.predicted_next_bytes > limit_bytes
    }

    /// Returns free VRAM on an NVIDIA GPU, if available.
    ///
    /// On success: `Some(bytes)` reported by `nvidia-smi`. On failure (no
    /// NVIDIA driver, no GPU, probe error): `None`. The first failure on
    /// this instance emits a one-time warning on stderr; subsequent failures
    /// are silent. Use [`vram_probe_failed_once`](Self::vram_probe_failed_once)
    /// to query failure state programmatically.
    pub fn available_vram_bytes(&self) -> Option<u64> {
        match vram_probe::read_nvidia_vram_free_bytes() {
            Ok(b) => Some(b),
            Err(e) => {
                self.note_probe_failure(&e);
                None
            }
        }
    }

    /// Returns an estimate of VRAM currently held by processes other than Atenia.
    ///
    /// Computed as `driver_reported_used_vram − current_bytes` (saturating at
    /// zero), where `current_bytes` is what this forecaster has been told
    /// Atenia registered. A positive result means external processes are
    /// consuming VRAM beyond what Atenia accounts for.
    ///
    /// Returns `None` if the probe is unavailable. Same warning behavior as
    /// [`available_vram_bytes`](Self::available_vram_bytes).
    ///
    /// Note: `current_bytes` is an estimator of Atenia-side allocations in
    /// bytes, not necessarily VRAM-resident bytes. Treat the result as a
    /// coarse diagnostic, not a precise accounting.
    pub fn external_memory_pressure_bytes(&self) -> Option<u64> {
        match vram_probe::read_nvidia_vram_snapshot() {
            Ok(snap) => {
                let own = self.current_bytes as u64;
                Some(snap.used_bytes.saturating_sub(own))
            }
            Err(e) => {
                self.note_probe_failure(&e);
                None
            }
        }
    }

    /// Returns `true` if any VRAM probe call on this instance has ever failed.
    ///
    /// Useful for diagnostics and for downstream modules that want to know
    /// whether the forecaster is currently backed by a live VRAM signal or
    /// only the static estimator.
    pub fn vram_probe_failed_once(&self) -> bool {
        self.vram_probe_failed.load(Ordering::Relaxed)
    }

    /// Records a probe failure; emits a warning to stderr the first time only.
    fn note_probe_failure(&self, err: &VramProbeError) {
        let was_failed = self.vram_probe_failed.swap(true, Ordering::Relaxed);
        if !was_failed {
            eprintln!(
                "[AMM forecaster] warning: VRAM probe unavailable ({:?}); \
                 continuing with the static byte estimator only",
                err
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_is_false_initially() {
        let f = MemoryForecaster::new();
        assert!(!f.vram_probe_failed_once());
    }

    #[test]
    fn note_probe_failure_sets_flag() {
        let f = MemoryForecaster::new();
        f.note_probe_failure(&VramProbeError::NvidiaSmiNotFound);
        assert!(f.vram_probe_failed_once());
    }

    #[test]
    fn note_probe_failure_is_idempotent() {
        let f = MemoryForecaster::new();
        // First call emits the warning.
        f.note_probe_failure(&VramProbeError::NvidiaSmiNotFound);
        // Second call must not panic and must leave the flag set.
        f.note_probe_failure(&VramProbeError::ParseError("x".into()));
        f.note_probe_failure(&VramProbeError::CommandFailed("y".into()));
        assert!(f.vram_probe_failed_once());
    }
}
