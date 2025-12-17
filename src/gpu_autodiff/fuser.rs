use crate::gpu_autodiff::fusion::FusedKernelSpec;
use crate::gpu_autodiff::ir_backward::BackwardKernelSpec;

pub struct FusionPlanner;

impl FusionPlanner {
    pub fn fuse(sequence: &[BackwardKernelSpec]) -> Option<FusedKernelSpec> {
        if sequence.len() < 2 {
            return None;
        }

        // APX 11.4 — simple rule: fuse everything sequentially
        Some(FusedKernelSpec::new("fused_backward", sequence.to_vec()))
    }
}
