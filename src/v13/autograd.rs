use super::async_executor::AsyncExecutor;
use super::execution_planner::ExecutionTarget;
use super::hybrid_memory::HybridMemoryManager;
use super::memory_types::{MemorySnapshot, MemoryTier, MoveError, StorageBacking};
use super::persistent_cache::CacheError;
use super::reconfigurable_graph::NodeId;
use super::streams::{StreamKind, TaskKind};

pub type GradId = String;

#[derive(Debug, Clone)]
pub struct TensorGrad {
    pub id: GradId,
    pub tier: MemoryTier,
}

#[derive(Debug, Clone)]
pub enum GradMoveRecord {
    Planned {
        grad_id: String,
        from: MemoryTier,
        to: MemoryTier,
    },
    Applied {
        grad_id: String,
        from: MemoryTier,
        to: MemoryTier,
    },
    Skipped {
        grad_id: String,
        reason: String,
    },
    Failed {
        grad_id: String,
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct AutogradNodeTrace {
    pub node_id: NodeId,
    pub forward_target: ExecutionTarget,
    pub requested_backward_target: ExecutionTarget,
    pub final_backward_target: ExecutionTarget,
    pub reason: String,
    pub moves: Vec<GradMoveRecord>,
}

#[derive(Debug, Clone)]
pub struct AutogradTrace {
    pub nodes: Vec<AutogradNodeTrace>,
}

#[derive(Debug, Clone)]
pub struct AutogradNode {
    pub node_id: NodeId,
    pub forward_target: ExecutionTarget,
    pub backward_target: ExecutionTarget,
    pub input_grads: Vec<TensorGrad>,
    pub output_grads: Vec<TensorGrad>,
    pub reason: String,
}

pub struct AutogradGraph {
    nodes: Vec<AutogradNode>,
}

impl AutogradGraph {
    pub fn new() -> Self {
        AutogradGraph { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: AutogradNode) {
        self.nodes.push(node);
    }

    pub fn nodes(&self) -> &[AutogradNode] {
        &self.nodes
    }
}

fn decide_backward_target(
    forward_target: ExecutionTarget,
    grad_tiers: &[MemoryTier],
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) -> ExecutionTarget {
    let _ = snapshot;

    let all_in_vram = !grad_tiers.is_empty()
        && grad_tiers
            .iter()
            .all(|t| matches!(t, MemoryTier::Vram));

    if matches!(forward_target, ExecutionTarget::Gpu) && all_in_vram && gpu_available {
        ExecutionTarget::Gpu
    } else {
        ExecutionTarget::Cpu
    }
}

pub fn prepare_gradients(
    mem: &mut HybridMemoryManager,
    grads: &[TensorGrad],
    target: ExecutionTarget,
    snapshot: &MemorySnapshot,
) -> Result<(), MoveError> {
    let desired_tier = match target {
        ExecutionTarget::Gpu => MemoryTier::Vram,
        ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => MemoryTier::Ram,
    };

    for grad in grads {
        let current = match mem.get_tier(&grad.id) {
            Some(t) => t,
            None => continue,
        };

        if current == desired_tier {
            continue;
        }

        let plan = mem.plan_move(&grad.id, desired_tier, snapshot)?;
        mem.apply_move(&grad.id, &plan)?;
    }

    Ok(())
}

pub fn execute_backward(
    exec: &mut AsyncExecutor,
    mem: &mut HybridMemoryManager,
    graph: &AutogradGraph,
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) {
    for node in graph.nodes().iter().rev() {
        let mut grad_tiers: Vec<MemoryTier> = Vec::new();
        for g in &node.input_grads {
            grad_tiers.push(g.tier);
        }
        for g in &node.output_grads {
            grad_tiers.push(g.tier);
        }

        let mut target = decide_backward_target(
            node.forward_target,
            &grad_tiers,
            snapshot,
            gpu_available,
        );

        let all_grads_cpu = (&node.input_grads, &node.output_grads);

        if matches!(target, ExecutionTarget::Gpu) {
            let prep_res = {
                if let Err(e) = prepare_gradients(mem, all_grads_cpu.0, target, snapshot) {
                    Err(e)
                } else if let Err(e) = prepare_gradients(mem, all_grads_cpu.1, target, snapshot) {
                    Err(e)
                } else {
                    Ok(())
                }
            };

            if prep_res.is_err() {
                exec.timeline.push(format!(
                    "BACKWARD_FALLBACK node={} from=Gpu to=Cpu",
                    node.node_id
                ));
                target = ExecutionTarget::Cpu;
            }
        }

        if matches!(target, ExecutionTarget::Cpu | ExecutionTarget::CpuFallback) {
            let _ = prepare_gradients(mem, all_grads_cpu.0, ExecutionTarget::Cpu, snapshot);
            let _ = prepare_gradients(mem, all_grads_cpu.1, ExecutionTarget::Cpu, snapshot);
        }

        let stream = match target {
            ExecutionTarget::Gpu => StreamKind::Gpu,
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => StreamKind::Cpu,
        };

        let name = format!("backward:node{}", node.node_id);
        exec.submit(stream, TaskKind::Compute { name: name.clone() }, 1);

        exec.timeline.push(format!(
            "BACKWARD node={} target={:?} reason={}",
            node.node_id, target, node.reason
        ));
    }
}

pub fn execute_backward_with_trace(
    exec: &mut AsyncExecutor,
    mem: &mut HybridMemoryManager,
    graph: &AutogradGraph,
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) -> AutogradTrace {
    let mut traces: Vec<AutogradNodeTrace> = Vec::new();

    for node in graph.nodes().iter().rev() {
        let mut grad_tiers: Vec<MemoryTier> = Vec::new();
        for g in &node.input_grads {
            grad_tiers.push(g.tier);
        }
        for g in &node.output_grads {
            grad_tiers.push(g.tier);
        }

        let requested_target = decide_backward_target(
            node.forward_target,
            &grad_tiers,
            snapshot,
            gpu_available,
        );

        let mut final_target = requested_target;
        let mut moves: Vec<GradMoveRecord> = Vec::new();

        let mut all_grads: Vec<&TensorGrad> = Vec::new();
        for g in &node.input_grads {
            all_grads.push(g);
        }
        for g in &node.output_grads {
            all_grads.push(g);
        }

        fn process_moves_for_target(
            mem: &mut HybridMemoryManager,
            grads: &[&TensorGrad],
            desired_tier: MemoryTier,
            snapshot: &MemorySnapshot,
            records: &mut Vec<GradMoveRecord>,
        ) -> bool {
            let mut any_failed = false;

            for grad in grads {
                let current = match mem.get_tier(&grad.id) {
                    Some(t) => t,
                    None => {
                        records.push(GradMoveRecord::Skipped {
                            grad_id: grad.id.clone(),
                            reason: "tensor not registered in memory manager".to_string(),
                        });
                        continue;
                    }
                };

                if current == desired_tier {
                    records.push(GradMoveRecord::Skipped {
                        grad_id: grad.id.clone(),
                        reason: "already in target tier".to_string(),
                    });
                    continue;
                }

                records.push(GradMoveRecord::Planned {
                    grad_id: grad.id.clone(),
                    from: current,
                    to: desired_tier,
                });

                let plan = match mem.plan_move(&grad.id, desired_tier, snapshot) {
                    Ok(p) => p,
                    Err(err) => {
                        any_failed = true;
                        records.push(GradMoveRecord::Failed {
                            grad_id: grad.id.clone(),
                            reason: format!("plan_move failed: {:?}", err),
                        });
                        continue;
                    }
                };

                match mem.apply_move(&grad.id, &plan) {
                    Ok(()) => {
                        records.push(GradMoveRecord::Applied {
                            grad_id: grad.id.clone(),
                            from: plan.from,
                            to: plan.to,
                        });
                    }
                    Err(err) => {
                        any_failed = true;
                        records.push(GradMoveRecord::Failed {
                            grad_id: grad.id.clone(),
                            reason: format!("apply_move failed: {:?}", err),
                        });
                    }
                }
            }

            any_failed
        }

        let mut gpu_moves_failed = false;

        match requested_target {
            ExecutionTarget::Gpu => {
                gpu_moves_failed = process_moves_for_target(
                    mem,
                    &all_grads,
                    MemoryTier::Vram,
                    snapshot,
                    &mut moves,
                );
            }
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => {
                let _ = process_moves_for_target(
                    mem,
                    &all_grads,
                    MemoryTier::Ram,
                    snapshot,
                    &mut moves,
                );
            }
        }

        let mut reason = node.reason.clone();

        if matches!(requested_target, ExecutionTarget::Gpu) && gpu_moves_failed {
            final_target = ExecutionTarget::Cpu;
            if reason.is_empty() {
                reason = "GPU fallback due to move failure".to_string();
            } else {
                reason = format!("{}; GPU fallback due to move failure", reason);
            }

            let _ = process_moves_for_target(
                mem,
                &all_grads,
                MemoryTier::Ram,
                snapshot,
                &mut moves,
            );
        }

        let stream = match final_target {
            ExecutionTarget::Gpu => StreamKind::Gpu,
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => StreamKind::Cpu,
        };

        let name = format!("backward:node{}", node.node_id);
        exec.submit(stream, TaskKind::Compute { name: name.clone() }, 1);

        let target_label = match final_target {
            ExecutionTarget::Gpu => "Gpu",
            ExecutionTarget::Cpu | ExecutionTarget::CpuFallback => "Cpu",
        };

        let fallback_flag = if matches!(requested_target, ExecutionTarget::Gpu)
            && !matches!(final_target, ExecutionTarget::Gpu)
        {
            1
        } else {
            0
        };

        exec.timeline.push(format!(
            "AUTOGRAD node={} target={} moves={} fallback={}",
            node.node_id,
            target_label,
            moves.len(),
            fallback_flag,
        ));

        traces.push(AutogradNodeTrace {
            node_id: node.node_id,
            forward_target: node.forward_target,
            requested_backward_target: requested_target,
            final_backward_target: final_target,
            reason,
            moves,
        });
    }

    AutogradTrace { nodes: traces }
}

#[derive(Debug, Clone)]
pub struct GradPersistReport {
    pub saved: usize,
    pub skipped: usize,
}

pub fn persist_grads_after_backward(
    mem: &mut HybridMemoryManager,
    grad_ids: &[String],
    created_unix: u64,
) -> Result<GradPersistReport, CacheError> {
    let mut saved = 0usize;
    let mut skipped = 0usize;

    // Use a snapshot with no pressure information for planning moves.
    let empty_snapshot = MemorySnapshot {
        vram: crate::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ram: crate::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ssd: crate::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
    };

    for gid in grad_ids {
        let tier = match mem.get_tier(gid) {
            Some(t) => t,
            None => {
                skipped += 1;
                continue;
            }
        };

        match tier {
            MemoryTier::Ram => {
                // Persist RAM-backed gradient bytes.
                let backing = mem.backing_for_test(gid);
                let bytes = match backing {
                    Some(StorageBacking::Ram(buf)) => buf.clone(),
                    _ => {
                        skipped += 1;
                        continue;
                    }
                };

                mem.persist_gradient_to_ssd_cache(gid, &bytes, created_unix, true)?;
                saved += 1;
            }
            MemoryTier::Vram => {
                // Move to RAM first using existing move APIs.
                let plan = match mem.plan_move(gid, MemoryTier::Ram, &empty_snapshot) {
                    Ok(p) => p,
                    Err(_) => {
                        skipped += 1;
                        continue;
                    }
                };

                if mem.apply_move(gid, &plan).is_err() {
                    skipped += 1;
                    continue;
                }

                let backing = mem.backing_for_test(gid);
                let bytes = match backing {
                    Some(StorageBacking::Ram(buf)) => buf.clone(),
                    _ => {
                        skipped += 1;
                        continue;
                    }
                };

                mem.persist_gradient_to_ssd_cache(gid, &bytes, created_unix, true)?;
                saved += 1;
            }
            MemoryTier::Ssd => {
                // Assume already persisted via cache or legacy SSD; do not duplicate.
                skipped += 1;
            }
            MemoryTier::Cpu => {
                // No defined persistence policy for Cpu tier; skip.
                skipped += 1;
            }
        }
    }

    Ok(GradPersistReport { saved, skipped })
}

pub fn warm_grads_before_backward(
    mem: &mut HybridMemoryManager,
    grad_ids: &[String],
) -> usize {
    let mut restored = 0usize;

    for gid in grad_ids {
        if mem.get_tier(gid).is_some() {
            continue;
        }

        let restored_bytes = match mem.restore_gradient_from_ssd_cache(gid) {
            Ok(b) => b,
            Err(_) => continue,
        };

        if mem
            .register_tensor_with_data(gid, restored_bytes, MemoryTier::Ram)
            .is_ok()
        {
            restored += 1;
        }
    }

    restored
}
