// APX 11.0 — Backward IR Specification

use crate::tensor::TensorRef;

#[derive(Clone, Debug)]
pub struct BackwardKernelSpec {
    pub name: String,
    pub code: String,
    pub inputs: Vec<TensorRef>,
    pub grads: Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
}

impl BackwardKernelSpec {
    pub fn new(
        name: impl Into<String>,
        code: impl Into<String>,
        inputs: Vec<TensorRef>,
        grads: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
    ) -> Self {
        Self {
            name: name.into(),
            code: code.into(),
            inputs,
            grads,
            outputs,
        }
    }
}
