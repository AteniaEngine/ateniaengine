use std::sync::Arc;

use crate::tensor::Tensor;
use crate::ops::op_ref::{BackwardOp, OpRef};
use crate::gpu_autodiff::ir_backward::BackwardKernelSpec;

pub struct AttentionOp;

impl AttentionOp {
    pub fn new() -> OpRef {
        Arc::new(Self {}).into()
    }
}

impl From<Arc<AttentionOp>> for OpRef {
    fn from(op: Arc<AttentionOp>) -> Self {
        OpRef::new(op)
    }
}

impl BackwardOp for AttentionOp {
    fn backward_gpu(&self, inputs: &[Tensor], grad_output: &Tensor) -> BackwardKernelSpec {
        // Expected inputs: [Q, K, V, P, output]
        // Expected grad_output: dY

        let code = r#"
extern \"C\" __global__
void attention_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* P,
    const float* dY,
    float* dQ,
    float* dK,
    float* dV,
    float* dP,
    float* dS,
    int B, int H, int M, int D
) {
    // APX 11.3 — IR ONLY (no numeric implementation yet)
    // Actual fused GPU kernels come in APX 11.10
}
"#;

        BackwardKernelSpec::new(
            "attention_backward",
            code,
            inputs.to_vec(),
            vec![grad_output.clone()],
            vec![],
        )
    }
}
