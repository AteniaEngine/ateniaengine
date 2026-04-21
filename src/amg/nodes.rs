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

/// Configuration for a Conv2D node (APX v20 M1).
///
/// Only the hyperparameters that cannot be derived from tensor shapes
/// live here. Kernel spatial size is inferred at execution time from
/// the weight tensor shape (OIHW layout: `kernel_h = weight.shape[2]`,
/// `kernel_w = weight.shape[3]`). Weights and bias enter the node via
/// the graph's input edges, not via this config.
///
/// Convention: tuples are `(h, w)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2DConfig {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2DConfig {
    pub const fn new(stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self { stride, padding }
    }
}

impl Default for Conv2DConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
        }
    }
}

/// Configuration for a MaxPool2D node (APX v20 M1).
///
/// MaxPool2D has no learnable weights; kernel spatial size lives in
/// the config because there is no weight tensor to derive it from.
///
/// Convention: tuples are `(h, w)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool2DConfig {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2DConfig {
    pub const fn new(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self { kernel, stride, padding }
    }

    /// Convenience constructor for the common non-overlapping case:
    /// `stride == kernel` and `padding == (0, 0)`.
    pub const fn non_overlapping(kernel: (usize, usize)) -> Self {
        Self {
            kernel,
            stride: kernel,
            padding: (0, 0),
        }
    }
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
    /// 2D convolution (APX v20 M1). Inputs: `[input, weight]` or
    /// `[input, weight, bias]`. Weight layout: OIHW. Input layout: NCHW.
    Conv2D(Conv2DConfig),
    /// 2D max pooling (APX v20 M1). Inputs: `[input]`. Input layout: NCHW.
    MaxPool2D(MaxPool2DConfig),
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
