use atenia_engine::gpu_autodiff::ir_backward::BackwardKernelSpec;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn test_backward_ir_structural() {
    // Fake tensors — no autograd yet
    let t_in = Tensor::new(vec![1], 1.0f32, Device::CPU, DType::F32);
    let t_grad = Tensor::new(vec![1], 1.0f32, Device::CPU, DType::F32);

    let spec = BackwardKernelSpec::new(
        "linear_backward",
        "extern \"C\" __global__ void linear_backward() {}",
        vec![t_in.clone()],
        vec![t_grad.clone()],
        vec![]
    );

    assert_eq!(spec.name, "linear_backward");
    assert!(spec.code.contains("linear_backward"));
    assert_eq!(spec.inputs.len(), 1);
    assert_eq!(spec.grads.len(), 1);
}
