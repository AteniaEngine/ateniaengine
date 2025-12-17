//! Graph node definitions for Atenia Model Graph.

use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    CPU,
    GPU,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActType {
    ReLU,
    SiLU,
    GELU,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Parameter,
    Add,
    Sub,
    Mul,
    MatMul,
    Transpose2D,
    IndexSelect,
    Reshape { target: Vec<isize> },
    TransposeLastTwo,
    BatchMatMul,
    BroadcastAdd,
    LogSoftmax,
    Gather,
    CrossEntropyLoss,
    Linear,
    RmsNorm,
    SiLU,
    Softmax,
    /// Generic activation wrapper for APX 4.8+ (keeps SiLU variant for compat).
    Activation(ActType),
    /// Fused Linear + Activation node.
    FusedLinearActivation(ActType),
    /// Fused multi-node chain: Linear -> [Activations...] -> Linear.
    FusedLinearActivationChain(Vec<ActType>),
    /// No-op placeholder used for structurally removed nodes.
    NoOp,
    Output,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub node_type: NodeType,
    pub inputs: Vec<usize>,
    pub output: Option<Tensor>,
    pub target: ExecutionTarget,
}

impl Node {
    pub fn new(id: usize, node_type: NodeType, inputs: Vec<usize>) -> Self {
        Self {
            id,
            node_type,
            inputs,
            output: None,
            target: ExecutionTarget::CPU,
        }
    }

    pub fn set_output(&mut self, tensor: Tensor) {
        self.output = Some(tensor);
    }
}
