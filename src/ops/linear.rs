use std::sync::Arc;

use crate::tensor::Tensor;
use crate::ops::op_ref::{BackwardOp, OpRef};
use crate::gpu_autodiff::ir_backward::BackwardKernelSpec;

pub struct LinearOp {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearOp {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> OpRef {
        Arc::new(Self { weight, bias }).into()
    }
}

impl From<Arc<LinearOp>> for OpRef {
    fn from(op: Arc<LinearOp>) -> Self {
        OpRef::new(op)
    }
}

impl BackwardOp for LinearOp {
    fn backward_gpu(&self, inputs: &[Tensor], grad_output: &Tensor) -> BackwardKernelSpec {
        let input = &inputs[0];

        let code = r#"
extern "C" __global__
void linear_backward(
    const float* X,
    const float* dY,
    const float* W,
    float* dX,
    float* dW,
    float* dB,
    int M, int K, int N
) {
    // APX 11.1 — real GPU IR (non-numeric in this version)
}
"#;

        BackwardKernelSpec::new(
            "linear_backward",
            code,
            vec![input.clone()],
            vec![grad_output.clone()],
            vec![],
        )
    }
}
