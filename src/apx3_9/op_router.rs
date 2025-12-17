use crate::amg::nodes::NodeType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecTarget {
    CPU,
    CpuOptimized,
    GPU,
    Auto,
}

pub fn route(node_type: &NodeType, shape: &[usize]) -> ExecTarget {
    match node_type {
        // Small ops → plain CPU
        NodeType::Add
        | NodeType::Mul
        | NodeType::Reshape { .. }
        | NodeType::BroadcastAdd
        | NodeType::IndexSelect => ExecTarget::CPU,

        // Norm & activations → prefer optimized path for larger tensors
        NodeType::RmsNorm | NodeType::SiLU | NodeType::Softmax => {
            if shape.iter().product::<usize>() >= 256 {
                ExecTarget::CpuOptimized
            } else {
                ExecTarget::CPU
            }
        }

        // Large kernels → future GPU (APX 4.0)
        NodeType::Linear | NodeType::BatchMatMul | NodeType::MatMul => ExecTarget::Auto,

        // Default
        _ => ExecTarget::CPU,
    }
}
