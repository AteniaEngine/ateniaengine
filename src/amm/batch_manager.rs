//! Batch orchestration utilities for efficient memory usage.

use crate::amm::forecaster::MemoryForecaster;
use crate::tensor::tensor::Tensor;

/// Computes safe batch sizes for training/inference passes given a memory budget.
pub struct BatchManager {
    pub memory_limit_bytes: usize,
    pub safety_margin_bytes: usize,
}

impl BatchManager {
    pub fn new(memory_limit_bytes: usize, safety_margin_bytes: usize) -> Self {
        Self {
            memory_limit_bytes,
            safety_margin_bytes,
        }
    }

    /// Given a prototype tensor that describes the per-sample footprint,
    /// returns the maximum batch size that fits within the memory budget once the
    /// safety margin has been subtracted.
    pub fn estimate_max_batch_size(&self, sample_tensor: &Tensor) -> usize {
        let mut forecaster = MemoryForecaster::new();

        let per_sample_bytes = sample_tensor.estimated_bytes();
        if per_sample_bytes == 0 {
            return 0;
        }

        let available = self
            .memory_limit_bytes
            .saturating_sub(self.safety_margin_bytes);
        if available == 0 {
            return 0;
        }

        let mut batch_size = available / per_sample_bytes;
        if batch_size == 0 {
            return 0;
        }

        loop {
            forecaster.current_bytes = 0;
            for _ in 0..batch_size {
                forecaster.register_tensor(sample_tensor);
            }

            let predicted = forecaster.current_bytes;
            if predicted <= available {
                return batch_size;
            }

            batch_size -= 1;
            if batch_size == 0 {
                return 0;
            }
        }
    }
}
