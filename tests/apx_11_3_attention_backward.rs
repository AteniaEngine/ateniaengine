// APX 11.3 - AttentionBackward GPU (IR Generator)
// Test mínimo que valida que el operador de atención genera un IR de backward GPU bien formado.

use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::ops::attention::AttentionOp;

#[test]
fn apx_11_3_attention_backward_ir() {
    // Fake tensors en CPU (el backward sigue siendo IR-only en esta fase).
    let q = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let k = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let v = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);
    let p = Tensor::new(vec![2, 2], 0.5f32, Device::CPU, DType::F32);
    let out = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    let grad_output = Tensor::new(vec![2, 2], 1.0f32, Device::CPU, DType::F32);

    // Usar el operador directamente (APX 11.3 sigue siendo sólo IR, sin ejecución GPU real).
    let op = AttentionOp::new();
    let spec = op.inner.backward_gpu(&[q, k, v, p, out], &grad_output);

    assert_eq!(spec.name, "attention_backward");
    assert!(spec.code.contains("attention_backward"));
    assert_eq!(spec.inputs.len(), 5);
    assert_eq!(spec.grads.len(), 1);
}
