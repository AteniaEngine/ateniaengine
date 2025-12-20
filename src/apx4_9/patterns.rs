use crate::amg::graph::Graph;
use crate::amg::nodes::{ActType, NodeType};

pub fn fuse_linear_activation_linear(graph: &mut Graph) -> usize {
    let mut fused = 0;
    let node_count = graph.nodes.len();

    for lin2_idx in 0..node_count {
        if !matches!(graph.nodes[lin2_idx].node_type, NodeType::Linear) {
            continue;
        }

        let inputs2 = graph.nodes[lin2_idx].inputs.clone();
        if inputs2.len() < 2 {
            continue;
        }

        let mut cur_id = inputs2[0];
        let mut chain: Vec<(usize, ActType)> = Vec::new();

        loop {
            match graph.nodes[cur_id].node_type.clone() {
                NodeType::Activation(act) => {
                    if graph.nodes[cur_id].inputs.len() != 1 {
                        break;
                    }
                    chain.push((cur_id, act));
                    cur_id = graph.nodes[cur_id].inputs[0];
                }
                NodeType::Linear => {
                    if chain.is_empty() {
                        break;
                    }

                    let lin1_idx = cur_id;
                    let l1_inputs = graph.nodes[lin1_idx].inputs.clone();
                    if l1_inputs.len() < 2 {
                        break;
                    }

                    let w1_id = l1_inputs[1];
                    let w2_id = inputs2[1];

                    let (w1_shape, w2_shape) = match (
                        graph.nodes[w1_id].output.as_ref(),
                        graph.nodes[w2_id].output.as_ref(),
                    ) {
                        (Some(w1_t), Some(w2_t)) => (&w1_t.shape, &w2_t.shape),
                        _ => break,
                    };

                    if w1_shape.len() != 2 || w2_shape.len() != 2 {
                        break;
                    }

                    let out_features1 = w1_shape[1];
                    let in_features2 = w2_shape[0];
                    if out_features1 != in_features2 {
                        break;
                    }

                    let x_id = l1_inputs[0];
                    let b1_id = l1_inputs.get(2).cloned();
                    let b2_id = inputs2.get(2).cloned();

                    let acts: Vec<ActType> = chain
                        .iter()
                        .rev()
                        .map(|(_, a)| a.clone())
                        .collect();

                    if acts.is_empty() {
                        break;
                    }

                    let mut fused_inputs = Vec::new();
                    fused_inputs.push(x_id);
                    fused_inputs.push(w1_id);
                    if let Some(b1) = b1_id {
                        fused_inputs.push(b1);
                    }
                    fused_inputs.push(w2_id);
                    if let Some(b2) = b2_id {
                        fused_inputs.push(b2);
                    }

                    graph.nodes[lin2_idx].node_type =
                        NodeType::FusedLinearActivationChain(acts);
                    graph.nodes[lin2_idx].inputs = fused_inputs;

                    graph.nodes[lin1_idx].node_type = NodeType::NoOp;
                    // Linear1 becomes NoOp: keep only its first input as source.
                    let old_inputs = graph.nodes[lin1_idx].inputs.clone();
                    if let Some(source) = old_inputs.get(0).cloned() {
                        graph.nodes[lin1_idx].inputs.clear();
                        graph.nodes[lin1_idx].inputs.push(source);
                    }
                    for (aid, _) in chain {
                        graph.nodes[aid].node_type = NodeType::NoOp;
                    }

                    fused += 1;
                    break;
                }
                NodeType::FusedLinearActivation(act1) => {
                    // APX 4.8 case already applied: FusedLinearActivation -> [Activation*] -> Linear.
                    // Treat it as initial Linear+Activation plus the remaining activation chain.
                    let fused_idx = cur_id;
                    let f_inputs = graph.nodes[fused_idx].inputs.clone();
                    if f_inputs.len() < 2 {
                        break;
                    }

                    let w1_id = f_inputs[1];
                    let w2_id = inputs2[1];

                    let (w1_shape, w2_shape) = match (
                        graph.nodes[w1_id].output.as_ref(),
                        graph.nodes[w2_id].output.as_ref(),
                    ) {
                        (Some(w1_t), Some(w2_t)) => (&w1_t.shape, &w2_t.shape),
                        _ => break,
                    };

                    if w1_shape.len() != 2 || w2_shape.len() != 2 {
                        break;
                    }

                    let out_features1 = w1_shape[1];
                    let in_features2 = w2_shape[0];
                    if out_features1 != in_features2 {
                        break;
                    }

                    let x_id = f_inputs[0];
                    let b1_id = f_inputs.get(2).cloned();
                    let b2_id = inputs2.get(2).cloned();

                    let mut acts: Vec<ActType> = Vec::new();
                    acts.push(act1);
                    acts.extend(chain.iter().rev().map(|(_, a)| a.clone()));

                    if acts.is_empty() {
                        break;
                    }

                    let mut fused_inputs = Vec::new();
                    fused_inputs.push(x_id);
                    fused_inputs.push(w1_id);
                    if let Some(b1) = b1_id {
                        fused_inputs.push(b1);
                    }
                    fused_inputs.push(w2_id);
                    if let Some(b2) = b2_id {
                        fused_inputs.push(b2);
                    }

                    graph.nodes[lin2_idx].node_type =
                        NodeType::FusedLinearActivationChain(acts);
                    graph.nodes[lin2_idx].inputs = fused_inputs;

                    graph.nodes[fused_idx].node_type = NodeType::NoOp;
                    // The initial FusedLinearActivation becomes NoOp: keep only its
                    // first input as source.
                    let old_inputs = graph.nodes[fused_idx].inputs.clone();
                    if let Some(source) = old_inputs.get(0).cloned() {
                        graph.nodes[fused_idx].inputs.clear();
                        graph.nodes[fused_idx].inputs.push(source);
                    }
                    for (aid, _) in chain {
                        graph.nodes[aid].node_type = NodeType::NoOp;
                    }

                    fused += 1;
                    break;
                }
                _ => {
                    break;
                }
            }
        }
    }

    fused
}
