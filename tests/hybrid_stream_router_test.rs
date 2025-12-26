use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};
use atenia_engine::v13::stream_router::StreamRouter;
use atenia_engine::v13::streams::StreamConfig;

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

fn make_executor(advanced: bool) -> AsyncExecutor {
    let cfg = StreamConfig {
        advanced_streams_supported: advanced,
    };
    AsyncExecutor::new(cfg)
}

#[test]
fn routes_gpu_compute_to_gpu_stream() {
    let mut exec = make_executor(true);
    let kernel = make_kernel("gpu_kernel", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.1));

    let bundle = StreamRouter::route_kernel(&mut exec, &kernel, &tiers, &snapshot, true);

    assert_eq!(bundle.plan_target, atenia_engine::v13::execution_planner::ExecutionTarget::Gpu);

    exec.run_to_completion();

    let mut saw_enqueue_gpu = false;
    let mut saw_run_gpu = false;

    for entry in &exec.timeline {
        if entry.starts_with("ENQUEUE") && entry.contains("stream=Gpu") {
            saw_enqueue_gpu = true;
        }
        if entry.starts_with("RUN") && entry.contains("stream=Gpu") {
            saw_run_gpu = true;
        }
        // Ensure no SSD prefetch tasks.
        if entry.contains("SsdPrefetch") {
            panic!("did not expect any SsdPrefetch entries");
        }
    }

    assert!(saw_enqueue_gpu);
    assert!(saw_run_gpu);
}

#[test]
fn injects_prefetch_when_tensor_on_ssd() {
    let mut exec = make_executor(true);
    let kernel = make_kernel("ssd_kernel", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram, MemoryTier::Ssd];
    let snapshot = make_snapshot(Some(0.1));

    let bundle = StreamRouter::route_kernel(&mut exec, &kernel, &tiers, &snapshot, true);

    // We expect two tasks: prefetch then compute.
    assert_eq!(bundle.submitted_task_ids.len(), 2);

    // Check timeline order for ENQUEUE events before running.
    let enqueues: Vec<&String> = exec
        .timeline
        .iter()
        .filter(|l| l.starts_with("ENQUEUE"))
        .collect();

    assert_eq!(enqueues.len(), 2);
    assert!(enqueues[0].contains("stream=SsdPrefetch"));
    assert!(enqueues[1].contains("name=ssd_kernel"));

    exec.run_to_completion();

    // Ensure both prefetch and compute RUN entries exist.
    let runs: Vec<&String> = exec
        .timeline
        .iter()
        .filter(|l| l.starts_with("RUN"))
        .collect();

    let has_prefetch_run = runs
        .iter()
        .any(|l| l.contains("stream=SsdPrefetch"));
    let has_compute_run = runs
        .iter()
        .any(|l| l.contains("name=ssd_kernel"));

    if !has_prefetch_run || !has_compute_run {
        panic!("expected both prefetch and compute RUN entries");
    }
}

#[test]
fn routes_to_cpu_when_gpu_not_available() {
    let mut exec = make_executor(true);
    let kernel = make_kernel("fallback_kernel", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.1));

    let bundle = StreamRouter::route_kernel(&mut exec, &kernel, &tiers, &snapshot, false);

    assert_eq!(
        bundle.plan_target,
        atenia_engine::v13::execution_planner::ExecutionTarget::CpuFallback,
    );

    exec.run_to_completion();

    for entry in &exec.timeline {
        if entry.starts_with("RUN") {
            assert!(entry.contains("stream=Cpu"));
        }
    }
}

#[test]
fn fallback_mode_serializes_gpu_task() {
    let mut exec = make_executor(false);
    let kernel = make_kernel("gpu_like_kernel", KernelKind::ComputeHeavy);
    let tiers = vec![MemoryTier::Ram];
    let snapshot = make_snapshot(Some(0.1));

    let bundle = StreamRouter::route_kernel(&mut exec, &kernel, &tiers, &snapshot, true);

    assert_eq!(
        bundle.plan_target,
        atenia_engine::v13::execution_planner::ExecutionTarget::Gpu,
    );

    exec.run_to_completion();

    let mut saw_fallback_gpu = false;

    for entry in &exec.timeline {
        if entry.starts_with("FALLBACK") && entry.contains("stream=Gpu") {
            saw_fallback_gpu = true;
        }
        if entry.starts_with("RUN") {
            assert!(entry.contains("stream=Cpu"));
        }
    }

    assert!(saw_fallback_gpu);
}
