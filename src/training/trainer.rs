//! Core training loop orchestration.

use crate::amm::batch_manager::BatchManager;
use crate::tensor::tensor::{Device, Tensor};

pub struct TrainerConfig {
    pub memory_limit_bytes: usize,
    pub safety_margin_bytes: usize,
}

pub struct Trainer {
    pub config: TrainerConfig,
    pub batch_manager: BatchManager,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let batch_manager = BatchManager::new(
            config.memory_limit_bytes,
            config.safety_margin_bytes,
        );

        Self {
            config,
            batch_manager,
        }
    }

    /// Simulates a forward pass by summing tensor data for now.
    pub fn forward(&self, batch: &Tensor) -> f32 {
        batch.as_cpu_slice().iter().sum()
    }

    /// Placeholder update method (no-op for now).
    pub fn update(&self, _loss: f32) {}

    /// Simulated training loop using adaptive batch sizes.
    pub fn train(&self, dataset: &Vec<Tensor>, sample_template: &Tensor) -> Vec<f32> {
        let batch_size = self
            .batch_manager
            .estimate_max_batch_size(sample_template);

        let mut losses = Vec::new();

        if batch_size == 0 {
            return losses;
        }

        for chunk in dataset.chunks(batch_size) {
            let mut combined = Vec::new();
            for tensor in chunk {
                combined.extend_from_slice(tensor.as_cpu_slice());
            }

            let mut batch_tensor = Tensor::new_cpu_with_layout(
                vec![chunk.len(), sample_template.numel()],
                combined,
                Device::CPU,
                sample_template.dtype,
                sample_template.layout,
            );
            batch_tensor.strides = sample_template.strides.clone();

            let loss = self.forward(&batch_tensor);
            self.update(loss);
            losses.push(loss);
        }

        losses
    }
}
