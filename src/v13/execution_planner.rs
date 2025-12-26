use super::kernel_model::KernelProfile;
use super::memory_types::{MemorySnapshot, MemoryTier};
use super::execution_trace::ExecutionDecisionTrace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
    CpuFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionRule {
    GpuNotAvailable,
    KernelNotGpuFriendly,
    TensorOnSsd,
    HighVramPressure,
    GpuPreferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub target: ExecutionTarget,
    pub reason: String,
    pub trace: Option<ExecutionDecisionTrace>,
}

pub struct HybridExecutionPlanner;

impl HybridExecutionPlanner {
    pub fn plan(
        kernel: &KernelProfile,
        tensor_tiers: &[MemoryTier],
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> ExecutionPlan {
        let mut evaluated_rules = Vec::new();

        // Rule 1: GPU not available at all.
        evaluated_rules.push(DecisionRule::GpuNotAvailable);
        if !gpu_available {
            let target = ExecutionTarget::CpuFallback;
            let reason = "GPU not available".to_string();
            let winning_rule = DecisionRule::GpuNotAvailable;
            let trace = ExecutionDecisionTrace {
                kernel_name: kernel.name.clone(),
                evaluated_rules,
                winning_rule,
                target,
                reason: reason.clone(),
            };
            return ExecutionPlan {
                target,
                reason,
                trace: Some(trace),
            };
        }

        // Rule 2: Kernel not suitable for GPU execution.
        evaluated_rules.push(DecisionRule::KernelNotGpuFriendly);
        if !kernel.is_gpu_friendly() {
            let target = ExecutionTarget::Cpu;
            let reason = "Kernel not suitable for GPU execution".to_string();
            let winning_rule = DecisionRule::KernelNotGpuFriendly;
            let trace = ExecutionDecisionTrace {
                kernel_name: kernel.name.clone(),
                evaluated_rules,
                winning_rule,
                target,
                reason: reason.clone(),
            };
            return ExecutionPlan {
                target,
                reason,
                trace: Some(trace),
            };
        }

        // Rule 3: Any tensor on SSD forces CPU execution.
        evaluated_rules.push(DecisionRule::TensorOnSsd);
        if tensor_tiers.iter().any(|tier| matches!(tier, MemoryTier::Ssd)) {
            let target = ExecutionTarget::Cpu;
            let reason = "Tensor resides on SSD".to_string();
            let winning_rule = DecisionRule::TensorOnSsd;
            let trace = ExecutionDecisionTrace {
                kernel_name: kernel.name.clone(),
                evaluated_rules,
                winning_rule,
                target,
                reason: reason.clone(),
            };
            return ExecutionPlan {
                target,
                reason,
                trace: Some(trace),
            };
        }

        // Rule 4: High VRAM pressure triggers CPU fallback.
        evaluated_rules.push(DecisionRule::HighVramPressure);
        let vram_pressure = snapshot.vram.pressure.unwrap_or(0.0);
        if vram_pressure > 0.90 {
            let target = ExecutionTarget::CpuFallback;
            let reason = "VRAM pressure too high".to_string();
            let winning_rule = DecisionRule::HighVramPressure;
            let trace = ExecutionDecisionTrace {
                kernel_name: kernel.name.clone(),
                evaluated_rules,
                winning_rule,
                target,
                reason: reason.clone(),
            };
            return ExecutionPlan {
                target,
                reason,
                trace: Some(trace),
            };
        }

        // Rule 5: Otherwise, prefer GPU execution.
        evaluated_rules.push(DecisionRule::GpuPreferred);
        let target = ExecutionTarget::Gpu;
        let reason = "GPU execution preferred".to_string();
        let winning_rule = DecisionRule::GpuPreferred;
        let trace = ExecutionDecisionTrace {
            kernel_name: kernel.name.clone(),
            evaluated_rules,
            winning_rule,
            target,
            reason: reason.clone(),
        };
        ExecutionPlan {
            target,
            reason,
            trace: Some(trace),
        }
    }
}
