use atenia_engine::nn::normalization::rmsnorm_backward_parallel;
use atenia_engine::nn::softmax::softmax_backward_parallel;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn tensor_from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
    assert_eq!(data.len(), shape.iter().product());
    let mut t = Tensor::with_layout(shape.to_vec(), 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    t.data = data;
    t
}

#[test]
fn test_parallel_softmax_backward_shapes() {
    let prob = tensor_from_vec(
        vec![0.2, 0.3, 0.5, 0.1, 0.1, 0.8],
        &[2, 3],
    );

    let grad = tensor_from_vec(
        vec![0.1, -0.2, 0.3, 0.5, -0.3, 0.1],
        &[2, 3],
    );

    let out = softmax_backward_parallel(&prob, &grad);

    assert_eq!(out.shape, vec![2, 3]);
}

#[test]
fn test_parallel_rmsnorm_backward_shapes() {
    let x = Tensor::randn(&[4, 8], Device::CPU);
    let g = Tensor::randn(&[4, 8], Device::CPU);

    let out = rmsnorm_backward_parallel(&x, &g);

    assert_eq!(out.shape, vec![4, 8]);
}

#[test]
fn test_parallel_softmax_backward_does_not_panic() {
    let prob = Tensor::randn(&[10, 32], Device::CPU);
    let grad = Tensor::randn(&[10, 32], Device::CPU);

    let _ = softmax_backward_parallel(&prob, &grad);
}

#[test]
fn test_parallel_rmsnorm_backward_does_not_panic() {
    let x = Tensor::randn(&[10, 32], Device::CPU);
    let g = Tensor::randn(&[10, 32], Device::CPU);

    let _ = rmsnorm_backward_parallel(&x, &g);
}
