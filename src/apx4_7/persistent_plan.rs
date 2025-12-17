use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;

#[derive(Debug, Clone)]
pub struct PersistentPlan {
    pub gpu_segments: Vec<(usize, usize)>,
}

impl PersistentPlan {
    pub fn analyze(g: &Graph) -> Self {
        let mut segments = Vec::new();
        let mut start = None;

        for (i, node) in g.nodes.iter().enumerate() {
            let gpu_ok = matches!(node.node_type, NodeType::Linear);

            match (gpu_ok, start) {
                (true, None) => start = Some(i),
                (true, Some(_)) => {}
                (false, Some(s)) => {
                    segments.push((s, i - 1));
                    start = None;
                }
                _ => {}
            }
        }

        if let Some(s) = start {
            segments.push((s, g.nodes.len() - 1));
        }

        PersistentPlan {
            gpu_segments: segments,
        }
    }
}
