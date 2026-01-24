use crate::v17::compute::tensor::Tensor;

/// Synthetic, MNIST-like CNN model used for APX 17.10.4.
///
/// The goal is determinism and auditability, not accuracy.
#[derive(Debug, Clone)]
pub struct MnistCNNModel {
    pub conv_weights: Tensor, // [out_channels, in_channels, k_h, k_w]
    pub conv_bias: Tensor,   // [out_channels]
    pub dense_weights: Tensor, // [10, flat_dim]
    pub dense_bias: Tensor,    // [10]
    pub target_digit: usize,
}

impl MnistCNNModel {
    /// Builds a small, deterministic CNN where a single target digit is
    /// guaranteed to have the highest logit for the fixed synthetic input.
    pub fn synthetic() -> Self {
        // Conv layer: 1 input channel, 1 output channel, 3x3 kernel.
        let conv_weights = Tensor::new(vec![1, 1, 3, 3], vec![
            0.0, 0.1, 0.0,
            0.1, 0.6, 0.1,
            0.0, 0.1, 0.0,
        ])
        .expect("conv_weights shape mismatch");
        let conv_bias = Tensor::new(vec![1], vec![0.0]).expect("conv_bias shape mismatch");

        // After Conv + ReLU + 2x2 MaxPool with stride 2, the spatial size is 14x14
        // with 1 channel, so flat_dim = 1 * 14 * 14.
        let flat_dim = 1 * 14 * 14;

        // Dense layer: 10 classes. Weights are zero for all classes except
        // the target digit, which sums all features. This guarantees that the
        // target digit has the largest logit for non-negative inputs.
        let target_digit = 3usize;
        let mut dense_w_data = vec![0.0_f32; 10 * flat_dim];
        for i in 0..flat_dim {
            dense_w_data[target_digit * flat_dim + i] = 1.0;
        }
        let dense_weights = Tensor::new(vec![10, flat_dim], dense_w_data)
            .expect("dense_weights shape mismatch");

        let dense_bias = Tensor::new(vec![10], vec![0.0; 10])
            .expect("dense_bias shape mismatch");

        Self {
            conv_weights,
            conv_bias,
            dense_weights,
            dense_bias,
            target_digit,
        }
    }
}
