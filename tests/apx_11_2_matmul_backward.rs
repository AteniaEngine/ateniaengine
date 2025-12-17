use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::nn::linear::matmul;

#[test]
fn test_matmul_backward_gpu_ir() {
    let a = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let b = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    let y = matmul(&a, &b);

    let grad_output = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    let op = y.op().expect("output of matmul must contain an op");
    let spec = op.inner.backward_gpu(&[a.clone(), b.clone()], &grad_output);

    assert_eq!(spec.name, "matmul_backward");
    assert!(spec.code.contains("matmul_backward"));
    assert_eq!(spec.inputs.len(), 2);
    assert_eq!(spec.grads.len(), 1);
}
