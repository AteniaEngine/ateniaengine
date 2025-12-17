//! Predictive models for upcoming memory requirements.

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

    /// Predicts the memory that will be required if an addition between `a` and `b` occurs.
    ///
    /// This simplistic heuristic assumes the result tensor shares `a`'s storage footprint.
    pub fn predict_add_operation(&mut self, a: &Tensor, b: &Tensor) {
        self.predicted_next_bytes =
            a.estimated_bytes() + b.estimated_bytes() + a.estimated_bytes();
    }

    /// Returns `true` if the predicted memory allocation would exceed the provided limit.
    pub fn is_over_limit(&self, limit_bytes: usize) -> bool {
        self.predicted_next_bytes > limit_bytes
    }
}
