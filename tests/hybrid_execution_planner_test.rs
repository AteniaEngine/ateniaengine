use atenia_engine::v13::execution_planner::{ExecutionTarget, HybridExecutionPlanner};
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};

fn make_kernel(name: &str, kind: KernelKind) -> KernelProfile {
    KernelProfile {
        name: name.to_string(),
        kind,
        estimated_flops: 1_000_000,
        estimated_bytes: 1_000_000,
    }
}

fn snapshot_with_vram_pressure(pressure: Option<f32>) -> MemorySnapshot {
    MemorySnapshot {
        vram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure,
        },
        ram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ssd: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
    }
}

#[test]
fn cpu_when_gpu_not_available() {
    let kernel = make_kernel("heavy_kernel", KernelKind::ComputeHeavy);
    let tiers = [MemoryTier::Ram];
    let snapshot = snapshot_with_vram_pressure(Some(0.1));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, false);

    assert_eq!(plan.target, ExecutionTarget::CpuFallback);
    assert!(plan.reason.contains("GPU not available"));
}

#[test]
fn cpu_when_kernel_small() {
    let kernel = make_kernel("small_kernel", KernelKind::Small);
    let tiers = [MemoryTier::Ram];
    let snapshot = snapshot_with_vram_pressure(Some(0.1));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    assert_eq!(plan.target, ExecutionTarget::Cpu);
    assert!(plan
        .reason
        .contains("Kernel not suitable for GPU execution"));
}

#[test]
fn cpu_when_tensor_on_ssd() {
    let kernel = make_kernel("gpu_friendly", KernelKind::ComputeHeavy);
    let tiers = [MemoryTier::Ram, MemoryTier::Ssd];
    let snapshot = snapshot_with_vram_pressure(Some(0.1));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    assert_eq!(plan.target, ExecutionTarget::Cpu);
    assert!(plan.reason.contains("Tensor resides on SSD"));
}

#[test]
fn cpu_fallback_when_vram_pressure_high() {
    let kernel = make_kernel("gpu_friendly", KernelKind::ComputeHeavy);
    let tiers = [MemoryTier::Ram];
    let snapshot = snapshot_with_vram_pressure(Some(0.95));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    assert_eq!(plan.target, ExecutionTarget::CpuFallback);
    assert!(plan.reason.contains("VRAM pressure too high"));
}

#[test]
fn gpu_when_all_conditions_good() {
    let kernel = make_kernel("gpu_friendly", KernelKind::ComputeHeavy);
    let tiers = [MemoryTier::Ram];
    let snapshot = snapshot_with_vram_pressure(Some(0.2));

    let plan = HybridExecutionPlanner::plan(&kernel, &tiers, &snapshot, true);

    assert_eq!(plan.target, ExecutionTarget::Gpu);
    assert!(plan.reason.contains("GPU execution preferred"));
}
