use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;

/// Detecta y fusiona pares Linear -> Activation en el grafo.
/// La fusión convierte el nodo Activation en FusedLinearActivation y marca
/// el Linear como NoOp, manteniendo las dependencias aguas abajo.
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

            // Fusión: el nodo de activación se convierte en FusedLinearActivation
            // y toma los mismos inputs que el Linear.
            let lin_inputs = graph.nodes[lin_idx].inputs.clone();
            graph.nodes[act_idx].node_type = NodeType::FusedLinearActivation(act_kind);
            graph.nodes[act_idx].inputs = lin_inputs;

            // El Linear original pasa a ser NoOp.
            graph.nodes[lin_idx].node_type = NodeType::NoOp;

            // Asegurar que el NoOp conserve solo su fuente de datos como input
            // (primer input del Linear original).
            let old_inputs = graph.nodes[lin_idx].inputs.clone();
            if let Some(source) = old_inputs.get(0).cloned() {
                graph.nodes[lin_idx].inputs.clear();
                graph.nodes[lin_idx].inputs.push(source);
            }

            // Redirigir dependencias aguas abajo: cualquier nodo que dependía del
            // Linear original ahora debe depender del nodo fusionado.
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
