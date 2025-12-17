use crate::amg::nodes::NodeType;

#[derive(Debug, Clone)]
pub struct GpuSegment {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct GpuPlan {
    pub segments: Vec<GpuSegment>,
}

impl GpuPlan {
    pub fn build(node_types: &[NodeType]) -> Self {
        let mut segments = vec![];
        let mut start: Option<usize> = None;

        for (i, nt) in node_types.iter().enumerate() {
            let gpu_ok = is_cuda_available_for(nt);

            match gpu_ok {
                true => {
                    if start.is_none() {
                        start = Some(i);
                    }
                }
                false => {
                    if let Some(s) = start {
                        segments.push(GpuSegment { start: s, end: i - 1 });
                        start = None;
                    }
                }
            }
        }

        if let Some(s) = start {
            segments.push(GpuSegment {
                start: s,
                end: node_types.len() - 1,
            });
        }

        Self { segments }
    }
}

pub fn is_cuda_available_for(node: &NodeType) -> bool {
    matches!(
        node,
        NodeType::MatMul
        // Linear desactivado temporalmente para que el backward CPU se registre correctamente
    )
}
