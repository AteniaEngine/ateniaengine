use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;

#[derive(Debug, Clone)]
pub struct FusionPlan {
    pub fused_pairs: Vec<(usize, usize)>,
}

/// Detect Linear->Linear pairs (A -> B) based on the data flow:
/// A's output feeds as the first input of B.
pub fn find_fusable_pairs(g: &Graph) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();

    for (a_idx, a_node) in g.nodes.iter().enumerate() {
        if !matches!(a_node.node_type, NodeType::Linear) {
            continue;
        }

        // Look for B nodes that consume A as first input and are also Linear.
        for (b_idx, b_node) in g.nodes.iter().enumerate() {
            if !matches!(b_node.node_type, NodeType::Linear) {
                continue;
            }

            if let Some(&first_input) = b_node.inputs.get(0) {
                if first_input == a_idx {
                    pairs.push((a_idx, b_idx));
                }
            }
        }
    }

    pairs
}

impl FusionPlan {
    pub fn analyze(g: &Graph) -> Self {
        let fused = find_fusable_pairs(g);
        FusionPlan { fused_pairs: fused }
    }
}
