//! Simple arithmetic estimator for static memory footprint of tensor operations.

use crate::tensor::tensor::Tensor;

/// Tracks current and predicted memory usage for upcoming tensor operations.
#[derive(Default)]
pub struct MemoryForecaster {
    pub current_bytes: usize,
    pub predicted_next_bytes: usize,
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
}
