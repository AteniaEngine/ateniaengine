use super::async_executor::AsyncExecutor;
use super::execution_planner::{ExecutionTarget, HybridExecutionPlanner};
use super::hybrid_memory::HybridMemoryManager;
use super::kernel_model::KernelProfile;
use super::memory_types::{MemorySnapshot, MemoryTier};
use super::streams::{StreamKind, TaskKind};

pub struct StreamRouter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedBundle {
    pub plan_target: ExecutionTarget,
    pub submitted_task_ids: Vec<u64>,
    pub reason: String,
}

impl StreamRouter {
    pub fn route_kernel(
        exec: &mut AsyncExecutor,
        kernel: &KernelProfile,
        tensor_tiers: &[MemoryTier],
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> RoutedBundle {
        let plan = HybridExecutionPlanner::plan(kernel, tensor_tiers, snapshot, gpu_available);

        let mut submitted_task_ids = Vec::new();

        // Rule 2: inject SSD prefetch task if any tensor resides on SSD.
        if tensor_tiers.iter().any(|tier| matches!(tier, MemoryTier::Ssd)) {
            let prefetch_name = format!("prefetch:{}", kernel.name);
            let id = exec.submit(
                StreamKind::SsdPrefetch,
                TaskKind::Io { name: prefetch_name },
                1,
            );
            submitted_task_ids.push(id);
        }

        // Route compute task according to planner target.
        let stream = match plan.target {
            ExecutionTarget::Gpu => StreamKind::Gpu,
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => StreamKind::Cpu,
        };

        let compute_name = kernel.name.clone();
        let compute_id = exec.submit(
            stream,
            TaskKind::Compute { name: compute_name },
            1,
        );
        submitted_task_ids.push(compute_id);

        RoutedBundle {
            plan_target: plan.target,
            submitted_task_ids,
            reason: plan.reason,
        }
    }

    pub fn route_kernel_with_memory(
        exec: &mut AsyncExecutor,
        mem: &mut HybridMemoryManager,
        kernel: &KernelProfile,
        tensor_ids: &[&str],
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> RoutedBundle {
        // Derive tensor tiers from the hybrid memory manager.
        let mut id_and_tier: Vec<(String, MemoryTier)> = Vec::new();
        for id in tensor_ids {
            let tier = match mem.get_tier(id) {
                Some(t) => t,
                None => MemoryTier::Ram,
            };
            id_and_tier.push((id.to_string(), tier));
        }

        let tensor_tiers: Vec<MemoryTier> = id_and_tier.iter().map(|(_, t)| *t).collect();

        let base_plan = HybridExecutionPlanner::plan(kernel, &tensor_tiers, snapshot, gpu_available);

        let mut submitted_task_ids = Vec::new();
        let mut degraded_to_cpu = false;
        let mut reason = base_plan.reason.clone();

        // Determine the required tier for compute based on the planner target.
        let required_tier_for_compute = match base_plan.target {
            ExecutionTarget::Gpu => MemoryTier::Vram,
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => MemoryTier::Ram,
        };

        // First pass: enqueue SSD prefetch tasks when needed.
        for (id, current_tier) in &id_and_tier {
            if *current_tier == MemoryTier::Ssd && required_tier_for_compute != MemoryTier::Ssd {
                let prefetch_name = format!("prefetch:{}", id);
                let task_id = exec.submit(
                    StreamKind::SsdPrefetch,
                    TaskKind::Io { name: prefetch_name },
                    1,
                );
                submitted_task_ids.push(task_id);
            }
        }

        // Second pass: plan and apply memory moves, enqueue transfer tasks.
        for (id, current_tier) in &id_and_tier {
            // Determine whether this tensor needs a move.
            if *current_tier == MemoryTier::Ssd && required_tier_for_compute != MemoryTier::Ssd {
                // SSD -> RAM/VRAM.
                let target = required_tier_for_compute;
                match mem.plan_move(id, target, snapshot) {
                    Ok(move_plan) => {
                        if move_plan.to != target {
                            // Could not reach the required tier (e.g. VRAM unavailable).
                            degraded_to_cpu = true;
                            reason = format!(
                                "{}; degraded to CPU because required memory tier is unavailable",
                                base_plan.reason
                            );
                            continue;
                        }

                        // Apply move; on error, degrade to CPU but do not panic.
                        if mem.apply_move(id, &move_plan).is_err() {
                            degraded_to_cpu = true;
                            reason = format!(
                                "{}; degraded to CPU because memory move failed",
                                base_plan.reason
                            );
                            continue;
                        }

                        let (stream, name) = match target {
                            MemoryTier::Ram => (
                                StreamKind::Cpu,
                                format!("move:ssd->ram:{}", id),
                            ),
                            MemoryTier::Vram => (
                                StreamKind::Gpu,
                                format!("move:ssd->vram:{}", id),
                            ),
                            _ => (StreamKind::Cpu, format!("move:ssd->other:{}", id)),
                        };

                        let task_id = exec.submit(
                            stream,
                            TaskKind::Transfer { name },
                            1,
                        );
                        submitted_task_ids.push(task_id);
                    }
                    Err(_) => {
                        degraded_to_cpu = true;
                        reason = format!(
                            "{}; degraded to CPU because planning memory move failed",
                            base_plan.reason
                        );
                    }
                }
            } else if *current_tier == MemoryTier::Ram
                && required_tier_for_compute == MemoryTier::Vram
                && matches!(base_plan.target, ExecutionTarget::Gpu)
            {
                // RAM -> VRAM for GPU execution.
                match mem.plan_move(id, MemoryTier::Vram, snapshot) {
                    Ok(move_plan) => {
                        if move_plan.to != MemoryTier::Vram {
                            // VRAM unavailable, degrade to CPU.
                            degraded_to_cpu = true;
                            reason = format!(
                                "{}; degraded to CPU because VRAM is unavailable",
                                base_plan.reason
                            );
                            continue;
                        }

                        if mem.apply_move(id, &move_plan).is_err() {
                            degraded_to_cpu = true;
                            reason = format!(
                                "{}; degraded to CPU because memory move failed",
                                base_plan.reason
                            );
                            continue;
                        }

                        let name = format!("move:ram->vram:{}", id);
                        let task_id = exec.submit(
                            StreamKind::Gpu,
                            TaskKind::Transfer { name },
                            1,
                        );
                        submitted_task_ids.push(task_id);
                    }
                    Err(_) => {
                        degraded_to_cpu = true;
                        reason = format!(
                            "{}; degraded to CPU because planning memory move failed",
                            base_plan.reason
                        );
                    }
                }
            } else if *current_tier == MemoryTier::Vram
                && required_tier_for_compute == MemoryTier::Ram
                && !matches!(base_plan.target, ExecutionTarget::Gpu)
            {
                // VRAM -> RAM for CPU execution.
                match mem.plan_move(id, MemoryTier::Ram, snapshot) {
                    Ok(move_plan) => {
                        if move_plan.to != MemoryTier::Ram {
                            degraded_to_cpu = true;
                            reason = format!(
                                "{}; degraded to CPU because RAM is unavailable",
                                base_plan.reason
                            );
                            continue;
                        }

                        if mem.apply_move(id, &move_plan).is_err() {
                            degraded_to_cpu = true;
                            reason = format!(
                                "{}; degraded to CPU because memory move failed",
                                base_plan.reason
                            );
                            continue;
                        }

                        let name = format!("move:vram->ram:{}", id);
                        let task_id = exec.submit(
                            StreamKind::Cpu,
                            TaskKind::Transfer { name },
                            1,
                        );
                        submitted_task_ids.push(task_id);
                    }
                    Err(_) => {
                        degraded_to_cpu = true;
                        reason = format!(
                            "{}; degraded to CPU because planning memory move failed",
                            base_plan.reason
                        );
                    }
                }
            }
        }

        // Decide final execution target for compute.
        let final_target = if degraded_to_cpu {
            ExecutionTarget::CpuFallback
        } else {
            base_plan.target
        };

        let compute_stream = match final_target {
            ExecutionTarget::Gpu => StreamKind::Gpu,
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => StreamKind::Cpu,
        };

        let compute_id = exec.submit(
            compute_stream,
            TaskKind::Compute {
                name: kernel.name.clone(),
            },
            1,
        );
        submitted_task_ids.push(compute_id);

        RoutedBundle {
            plan_target: final_target,
            submitted_task_ids,
            reason,
        }
    }
}
