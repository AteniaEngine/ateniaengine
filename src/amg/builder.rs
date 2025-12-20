//! Graph builder utilities for assembling Atenia Model Graphs.

use super::nodes::{Node, NodeType, ActType};
use super::scheduler::build_execution_plan;
use crate::tensor::Tensor;
use crate::apx4_8::pattern::detect_and_fuse_linear_activation;
use crate::apx4_9::patterns::fuse_linear_activation_linear;
use crate::apx4_7::{PersistentPlan, FusionPlan};

pub struct GraphBuilder {
    pub nodes: Vec<Node>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn add_node(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::new(id, node_type, inputs));
        id
    }

    pub fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        self.add_node(node_type, inputs)
    }

    pub fn input(&mut self) -> usize {
        self.add_node(NodeType::Input, vec![])
    }

    pub fn parameter(&mut self, value: Tensor) -> usize {
        let id = self.add_node(NodeType::Parameter, vec![]);
        self.nodes[id].output = Some(value);
        id
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::Add, vec![a, b])
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::Sub, vec![a, b])
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::Mul, vec![a, b])
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::MatMul, vec![a, b])
    }

    /// Gather rows from a parameter tensor using index tensor.
    pub fn index_select(&mut self, table_id: usize, indices_id: usize) -> usize {
        self.add_node(NodeType::IndexSelect, vec![table_id, indices_id])
    }

    pub fn reshape(&mut self, src: usize, target: Vec<isize>) -> usize {
        self.add_node(NodeType::Reshape { target }, vec![src])
    }

    pub fn transpose_last_two(&mut self, src: usize) -> usize {
        self.add_node(NodeType::TransposeLastTwo, vec![src])
    }

    pub fn batch_matmul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::BatchMatMul, vec![a, b])
    }

    pub fn broadcast_add(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::BroadcastAdd, vec![a, b])
    }

    pub fn log_softmax(&mut self, src: usize) -> usize {
        self.add_node(NodeType::LogSoftmax, vec![src])
    }

    pub fn gather(&mut self, data: usize, indices: usize) -> usize {
        self.add_node(NodeType::Gather, vec![data, indices])
    }

    pub fn cross_entropy_loss(&mut self, log_probs: usize, targets: usize) -> usize {
        self.add_node(NodeType::CrossEntropyLoss, vec![log_probs, targets])
    }

    pub fn output(&mut self, src: usize) -> usize {
        self.add_node(NodeType::Output, vec![src])
    }

    /// Create a Linear node with required inputs [x, w] and optional bias.
    pub fn linear(&mut self, x_id: usize, w_id: usize, b_id: Option<usize>) -> usize {
        let mut inputs = vec![x_id, w_id];
        if let Some(b) = b_id {
            inputs.push(b);
        }
        self.add_node(NodeType::Linear, inputs)
    }

    /// RMSNorm over last dimension.
    pub fn rms_norm(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::RmsNorm, vec![x_id])
    }

    /// ReLU activation.
    pub fn relu(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Activation(ActType::ReLU), vec![x_id])
    }

    /// SiLU activation.
    pub fn silu(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Activation(ActType::SiLU), vec![x_id])
    }

    /// GELU activation.
    pub fn gelu(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Activation(ActType::GELU), vec![x_id])
    }

    /// Softmax along last dimension.
    pub fn softmax(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Softmax, vec![x_id])
    }

    pub fn build(self) -> super::graph::Graph {
        let mut graph = super::graph::Graph::new(self.nodes);

        // Apply APX 4.8 and 4.9 structural fusions on the already-built graph.
        let _ = detect_and_fuse_linear_activation(&mut graph);
        let _ = fuse_linear_activation_linear(&mut graph);

        // Rebuild execution plan so it matches the fused graph structure.
        let (plan, fused_ops) = build_execution_plan(&graph.nodes);
        graph.plan = plan;
        graph.fused_ops = fused_ops;

        // Recompute APX 4.7 persistent and fusion plans so that any Linear→Linear
        // fusions are consistent with the transformed graph (and do not intercept
        // newly fused APX 4.9 nodes incorrectly).
        graph.persistent_plan = Some(PersistentPlan::analyze(&graph));
        graph.fusion_plan = Some(FusionPlan::analyze(&graph));

        graph
    }
}
