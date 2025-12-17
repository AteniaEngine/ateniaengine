use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::nn::linear::linear;

#[test]
fn test_linear_backward_gpu_ir() {
    let x = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let w = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let b = Tensor::new(vec![2], 0.0f32, Device::CPU, DType::F32);

    let y = linear(&x, &w, Some(&b));
    let grad_output = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    let op = y.op().expect("y should have an op");
    let spec = op.inner.backward_gpu(&[x.clone()], &grad_output);

    assert_eq!(spec.name, "linear_backward");
    assert!(spec.code.contains("linear_backward"));
}
