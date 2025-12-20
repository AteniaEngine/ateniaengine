// APX 11.3 - AttentionBackward GPU (IR Generator)
// Minimal test validating that the attention operator generates a well-formed GPU backward IR.

use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::ops::attention::AttentionOp;

#[test]
fn apx_11_3_attention_backward_ir() {
    // Fake CPU tensors (backward is still IR-only at this stage).
    let q = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let k = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let v = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let p = Tensor::new(vec![2, 2], 0.5f32, Device::CPU, DType::F32);
    let out = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    let grad_output = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    // Use the operator directly (APX 11.3 is still IR-only, with no real GPU execution).
    let op = AttentionOp::new();
    let spec = op.inner.backward_gpu(&[q, k, v, p, out], &grad_output);

    assert_eq!(spec.name, "attention_backward");
    assert!(spec.code.contains("attention_backward"));
    assert_eq!(spec.inputs.len(), 5);
    assert_eq!(spec.grads.len(), 1);
}
