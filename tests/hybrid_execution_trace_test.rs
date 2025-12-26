use atenia_engine::v13::execution_planner::{DecisionRule, ExecutionTarget, HybridExecutionPlanner};
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};

fn make_kernel(name: &str, kind: KernelKind) -> KernelProfile {
    KernelProfile {
        name: name.to_string(),
        kind,
        estimated_flops: 0,
        estimated_bytes: 0,
    }
}

fn make_snapshot(vram_pressure: Option<f32>) -> MemorySnapshot {
    let tier = TierStatus {
        total_bytes: None,
        free_bytes: None,
        pressure: vram_pressure,
    };
    MemorySnapshot {
        vram: tier,
        ram: tier,
        ssd: tier,
    }
}

#[test]
fn trace_records_rule_order() {
    let kernel = make_kernel("gpu_not_available", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.1));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, false);

    match plan.trace {
        Some(trace) => {
            assert_eq!(trace.evaluated_rules, vec![DecisionRule::GpuNotAvailable]);
            assert_eq!(trace.winning_rule, DecisionRule::GpuNotAvailable);
            assert_eq!(trace.target, ExecutionTarget::CpuFallback);
        }
        None => panic!("expected trace to be present"),
    }
}

#[test]
fn trace_records_multiple_rules_until_match() {
    let kernel = make_kernel("small_kernel", KernelKind::Small);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.1));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    match plan.trace {
        Some(trace) => {
            assert_eq!(
                trace.evaluated_rules,
                vec![DecisionRule::GpuNotAvailable, DecisionRule::KernelNotGpuFriendly],
            );
            assert_eq!(trace.winning_rule, DecisionRule::KernelNotGpuFriendly);
            assert_eq!(trace.target, ExecutionTarget::Cpu);
        }
        None => panic!("expected trace to be present"),
    }
}

#[test]
fn trace_gpu_preferred_case() {
    let kernel = make_kernel("good_gpu_kernel", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.2));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    match plan.trace {
        Some(trace) => {
            assert_eq!(
                trace.evaluated_rules,
                vec![
                    DecisionRule::GpuNotAvailable,
                    DecisionRule::KernelNotGpuFriendly,
                    DecisionRule::TensorOnSsd,
                    DecisionRule::HighVramPressure,
                    DecisionRule::GpuPreferred,
                ],
            );
            assert_eq!(trace.winning_rule, DecisionRule::GpuPreferred);
            assert_eq!(trace.target, ExecutionTarget::Gpu);
        }
        None => panic!("expected trace to be present"),
    }
}

#[test]
fn trace_reason_matches_plan() {
    let kernel = make_kernel("reason_match", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.2));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    match plan.trace {
        Some(trace) => {
            assert_eq!(trace.reason, plan.reason);
            assert_eq!(trace.target, plan.target);
        }
        None => panic!("expected trace to be present"),
    }
}
