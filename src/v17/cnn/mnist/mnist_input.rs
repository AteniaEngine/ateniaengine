use crate::v17::compute::tensor::Tensor;

/// Deterministic synthetic MNIST-like input: 1x1x28x28.
pub fn mnist_synthetic_input() -> Tensor {
    // Simple pattern: ramp values from 0.0 to 1.0 across the 28x28 grid.
    let n = 1usize;
    let c = 1usize;
    let h = 28usize;
    let w = 28usize;
    let total = n * c * h * w;
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        data.push(i as f32 / total as f32);
    }
    Tensor::new(vec![n, c, h, w], data).expect("mnist_synthetic_input shape mismatch")
}
