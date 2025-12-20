// APX 8.1 — DualGraph Builder
// Does not execute GPU. Does not touch backward.
// Only builds the duplicated CPU+GPU graph for later use.

use crate::amg::graph::Graph;
use crate::amg::nodes::Node;
use crate::tensor::Device;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePlacement {
    CPU,
    GPU,
}

#[derive(Clone, Debug)]
pub struct DualNode {
    pub id_cpu: usize,
    pub id_gpu: usize,
    pub op: String,
    pub inputs: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct DualGraph {
    pub cpu_nodes: Vec<Node>,
    pub gpu_nodes: Vec<Node>,
    pub mapping_cpu_to_gpu: Vec<usize>,
    pub mapping_gpu_to_cpu: Vec<usize>,
}

impl DualGraph {
    pub fn new() -> Self {
        DualGraph {
            cpu_nodes: Vec::new(),
            gpu_nodes: Vec::new(),
            mapping_cpu_to_gpu: Vec::new(),
            mapping_gpu_to_cpu: Vec::new(),
        }
    }
}

pub struct DualGraphBuilder;

impl DualGraphBuilder {
    pub fn build(graph: &Graph) -> DualGraph {
        let mut dg = DualGraph::new();

        for (i, n) in graph.nodes.iter().enumerate() {
            // CPU copy
            dg.cpu_nodes.push(n.clone());

            // GPU mirror: clone the node and, if it has an output tensor,
            // clone that tensor and mark it as GPU for future use.
            let mut gpu_clone = n.clone();
            if let Some(ref out) = n.output {
                let mut t = out.clone();
                t.device = Device::GPU;
                gpu_clone.output = Some(t);
            }
            dg.gpu_nodes.push(gpu_clone);

            dg.mapping_cpu_to_gpu.push(i);
            dg.mapping_gpu_to_cpu.push(i);
        }

        dg
    }
}
