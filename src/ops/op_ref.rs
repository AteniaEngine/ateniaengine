use std::sync::Arc;

use crate::tensor::Tensor;
use crate::gpu_autodiff::ir_backward::BackwardKernelSpec;

pub trait BackwardOp {
    fn backward_gpu(&self, inputs: &[Tensor], grad_output: &Tensor) -> BackwardKernelSpec;
}

#[derive(Clone)]
pub struct OpRef {
    pub inner: Arc<dyn BackwardOp + Send + Sync>,
}

impl OpRef {
    pub fn new(op: Arc<dyn BackwardOp + Send + Sync>) -> Self {
        Self { inner: op }
    }
}
