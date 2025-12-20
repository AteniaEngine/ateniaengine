use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;

/// Detect and fuse Linear -> Activation pairs in the graph.
/// The fusion converts the Activation node into FusedLinearActivation and marks
/// the Linear as NoOp, keeping downstream dependencies.
pub fn detect_and_fuse_linear_activation(graph: &mut Graph) -> usize {
    let mut count = 0;

    loop {
        let mut fused_one = false;

        for act_idx in 0..graph.nodes.len() {
            let act_node_type = graph.nodes[act_idx].node_type.clone();
            let act_kind = match act_node_type {
                NodeType::Activation(a) => a,
                _ => continue,
            };

            if graph.nodes[act_idx].inputs.is_empty() {
                continue;
            }
            let lin_idx = graph.nodes[act_idx].inputs[0];

            if !matches!(graph.nodes[lin_idx].node_type, NodeType::Linear) {
                continue;
            }

            // Fusion: the activation node becomes FusedLinearActivation
            // and takes the same inputs as the Linear.
            let lin_inputs = graph.nodes[lin_idx].inputs.clone();
            graph.nodes[act_idx].node_type = NodeType::FusedLinearActivation(act_kind);
            graph.nodes[act_idx].inputs = lin_inputs;

            // The original Linear becomes NoOp.
            graph.nodes[lin_idx].node_type = NodeType::NoOp;

            // Ensure the NoOp keeps only its data source as input
            // (first input of the original Linear).
            let old_inputs = graph.nodes[lin_idx].inputs.clone();
            if let Some(source) = old_inputs.get(0).cloned() {
                graph.nodes[lin_idx].inputs.clear();
                graph.nodes[lin_idx].inputs.push(source);
            }

            // Redirect downstream dependencies: any node that depended on the
            // original Linear must now depend on the fused node.
            let fused_id = act_idx;
            let old_output = lin_idx;
            for node in graph.nodes.iter_mut() {
                for inp in node.inputs.iter_mut() {
                    if *inp == old_output {
                        *inp = fused_id;
                    }
                }
            }

            count += 1;
            fused_one = true;
            break;
        }

        if !fused_one {
            break;
        }
    }

    count
}
