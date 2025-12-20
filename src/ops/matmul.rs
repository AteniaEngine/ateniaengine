use std::sync::Arc;

use crate::tensor::Tensor;
use crate::ops::op_ref::{BackwardOp, OpRef};
use crate::gpu_autodiff::ir_backward::BackwardKernelSpec;

pub struct MatMulOp;

impl MatMulOp {
    pub fn new() -> OpRef {
        Arc::new(Self {}).into()
    }
}

impl From<Arc<MatMulOp>> for OpRef {
    fn from(op: Arc<MatMulOp>) -> Self {
        OpRef::new(op)
    }
}

impl BackwardOp for MatMulOp {
    fn backward_gpu(&self, inputs: &[Tensor], grad_output: &Tensor) -> BackwardKernelSpec {
        let a = &inputs[0];
        let b = &inputs[1];

        let code = r#"
extern \"C\" __global__
void matmul_backward(
    const float* A,
    const float* B,
    const float* dY,
    float* dA,
    float* dB,
    int M, int K, int N
) {
    // APX 11.2 — GPU IR (not implemented yet)
    // Real kernels arrive in APX 11.3
}
"#;

        BackwardKernelSpec::new(
            "matmul_backward",
            code,
            vec![a.clone(), b.clone()],
            vec![grad_output.clone()],
            vec![],
        )
    }
}
