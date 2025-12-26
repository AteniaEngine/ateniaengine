use atenia_engine::v13::execution_planner::ExecutionTarget;
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};
use atenia_engine::v13::reconfigurable_graph::{GraphPlacementPlan, ReconfigurableGraph};

fn make_snapshot(vram_pressure: f32, ram_pressure: f32) -> MemorySnapshot {
    MemorySnapshot {
        vram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(vram_pressure),
        },
        ram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(ram_pressure),
        },
        ssd: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(0.0),
        },
    }
}

fn make_kernel(name: &str, kind: KernelKind) -> KernelProfile {
    KernelProfile {
        name: name.to_string(),
        kind,
        estimated_bytes: 0,
        estimated_flops: 0,
    }
}

fn node_targets(plan: &GraphPlacementPlan) -> Vec<(u64, ExecutionTarget)> {
    plan
        .placements
        .iter()
        .map(|p| (p.node_id, p.target))
        .collect()
}

#[test]
fn graph_can_replan_with_different_snapshots() {
    let mut graph = ReconfigurableGraph::new();

    let kernel_heavy = make_kernel("heavy", KernelKind::ComputeHeavy);
    let kernel_small = make_kernel("small", KernelKind::Small);

    let tiers_ram = vec![MemoryTier::Ram];

    let id1 = graph.add_node(kernel_heavy, tiers_ram.clone());
    let id2 = graph.add_node(kernel_small, tiers_ram.clone());

    // Snapshot A: low VRAM pressure, GPU available.
    let snapshot_a = make_snapshot(0.10, 0.10);
    let plan_a = graph.plan_for_snapshot(&snapshot_a, true);
    let targets_a = node_targets(&plan_a);

    // Under low pressure and GPU available, heavy kernel should go to GPU,
    // while small kernel stays on CPU.
    assert_eq!(targets_a.len(), 2);
    assert_eq!(targets_a[0].0, id1);
    assert_eq!(targets_a[1].0, id2);
    assert_eq!(targets_a[0].1, ExecutionTarget::Gpu);
    assert_eq!(targets_a[1].1, ExecutionTarget::Cpu);

    // Snapshot B: high VRAM pressure, GPU still available.
    let snapshot_b = make_snapshot(0.96, 0.10);
    let plan_b = graph.plan_for_snapshot(&snapshot_b, true);
    let targets_b = node_targets(&plan_b);

    assert_eq!(targets_b.len(), 2);
    assert_eq!(targets_b[0].0, id1);
    assert_eq!(targets_b[1].0, id2);

    // High VRAM pressure forces CPU fallback for GPU-friendly kernels.
    assert!(
        targets_b[0].1 == ExecutionTarget::CpuFallback
            || targets_b[0].1 == ExecutionTarget::Cpu
    );
    assert_eq!(targets_b[1].1, ExecutionTarget::Cpu);

    // Plans should differ for the heavy node between snapshots.
    assert_ne!(targets_a[0].1, targets_b[0].1);

    // Reasons should mention pressure or GPU usage in some form.
    let heavy_reason_a = &plan_a.placements[0].reason;
    let heavy_reason_b = &plan_b.placements[0].reason;
    assert!(!heavy_reason_a.is_empty());
    assert!(!heavy_reason_b.is_empty());
}

#[test]
fn tensor_on_ssd_forces_cpu_in_plan() {
    let mut graph = ReconfigurableGraph::new();

    let kernel = make_kernel("ssd_kernel", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ssd];

    graph.add_node(kernel, tiers);

    let snapshot = make_snapshot(0.10, 0.10);
    let plan = graph.plan_for_snapshot(&snapshot, true);

    assert_eq!(plan.placements.len(), 1);
    let placement = &plan.placements[0];
    assert_eq!(placement.target, ExecutionTarget::Cpu);
    assert!(placement.reason.to_lowercase().contains("ssd"));
}

#[test]
fn snapshot_summary_contains_pressures() {
    let mut graph = ReconfigurableGraph::new();

    let kernel = make_kernel("k", KernelKind::Small);
    let tiers = vec![MemoryTier::Ram];

    graph.add_node(kernel, tiers);

    let snapshot = make_snapshot(0.42, 0.84);
    let plan = graph.plan_for_snapshot(&snapshot, true);

    let summary = &plan.snapshot_summary;
    assert!(summary.contains("vram="));
    assert!(summary.contains("ram="));
    assert!(summary.contains("0.42"));
    assert!(summary.contains("0.84"));
}
