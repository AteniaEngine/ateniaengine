use std::collections::HashMap;
use std::sync::Mutex;

use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};
use atenia_engine::v13::stream_router::StreamRouter;
use atenia_engine::v13::streams::StreamConfig;
use atenia_engine::v13::vram_adapter::VramAdapter;
use atenia_engine::v13::memory_types::MoveError;

fn make_kernel(name: &str, kind: KernelKind) -> KernelProfile {
    KernelProfile {
        name: name.to_string(),
        kind,
        estimated_flops: 0,
        estimated_bytes: 0,
    }
}

fn neutral_snapshot() -> MemorySnapshot {
    let tier = TierStatus {
        total_bytes: None,
        free_bytes: None,
        pressure: Some(0.1),
    };
    MemorySnapshot {
        vram: tier,
        ram: tier,
        ssd: tier,
    }
}

fn make_executor() -> AsyncExecutor {
    let cfg = StreamConfig {
        advanced_streams_supported: true,
    };
    AsyncExecutor::new(cfg)
}

fn make_cache_dir(name: &str) -> String {
    let _ = std::fs::remove_dir_all(name);
    name.to_string()
}

#[test]
fn ssd_tensor_is_moved_to_ram_before_cpu_compute() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_router_mem_ssd");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    let data = vec![1u8, 2, 3, 4];
    let id = "t1";

    match mem.register_tensor_with_data(id, data, MemoryTier::Ssd) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let mut exec = make_executor();
    let kernel = make_kernel("cpu_kernel", KernelKind::Small);
    let snapshot = neutral_snapshot();

    let bundle = StreamRouter::route_kernel_with_memory(
        &mut exec,
        &mut mem,
        &kernel,
        &[id],
        &snapshot,
        false,
    );

    // Kernel is small or GPU unavailable, so we expect CPU execution.
    assert!(matches!(
        bundle.plan_target,
        atenia_engine::v13::execution_planner::ExecutionTarget::Cpu
            | atenia_engine::v13::execution_planner::ExecutionTarget::CpuFallback
    ));

    // Check ENQUEUE ordering: prefetch -> move -> compute.
    let enqueues: Vec<&String> = exec
        .timeline
        .iter()
        .filter(|l| l.starts_with("ENQUEUE"))
        .collect();

    assert_eq!(enqueues.len(), 3);
    assert!(enqueues[0].contains("stream=SsdPrefetch"));
    assert!(enqueues[0].contains("prefetch:t1"));
    assert!(enqueues[1].contains("move:ssd->ram:t1"));
    assert!(enqueues[2].contains("name=cpu_kernel"));

    exec.run_to_completion();

    // Tensor should now reside in RAM.
    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));

    // Ensure there is at least one RUN for prefetch, transfer, and compute.
    let runs: Vec<&String> = exec
        .timeline
        .iter()
        .filter(|l| l.starts_with("RUN"))
        .collect();

    let has_prefetch_run = runs
        .iter()
        .any(|l| l.contains("stream=SsdPrefetch") && l.contains("prefetch:t1"));
    let has_transfer_run = runs
        .iter()
        .any(|l| l.contains("move:ssd->ram:t1"));
    let has_compute_run = runs
        .iter()
        .any(|l| l.contains("name=cpu_kernel"));

    if !has_prefetch_run || !has_transfer_run || !has_compute_run {
        panic!("expected RUN entries for prefetch, transfer, and compute");
    }
}

struct FakeVramAdapter {
    storage: Mutex<HashMap<String, Vec<u8>>>,
}

impl FakeVramAdapter {
    fn new() -> Self {
        FakeVramAdapter {
            storage: Mutex::new(HashMap::new()),
        }
    }
}

impl VramAdapter for FakeVramAdapter {
    fn is_available(&self) -> bool {
        true
    }

    fn upload(&self, id: &str, data: &[u8]) -> Result<(), MoveError> {
        let mut guard = match self.storage.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(MoveError::BackendUnavailable(
                    "Failed to lock FakeVramAdapter storage".to_string(),
                ))
            }
        };
        guard.insert(id.to_string(), data.to_vec());
        Ok(())
    }

    fn download(&self, id: &str) -> Result<Vec<u8>, MoveError> {
        let guard = match self.storage.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(MoveError::BackendUnavailable(
                    "Failed to lock FakeVramAdapter storage".to_string(),
                ))
            }
        };
        match guard.get(id) {
            Some(bytes) => Ok(bytes.clone()),
            None => Err(MoveError::BackendUnavailable(
                "VRAM handle not found in FakeVramAdapter".to_string(),
            )),
        }
    }

    fn free(&self, id: &str) -> Result<(), MoveError> {
        let mut guard = match self.storage.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(MoveError::BackendUnavailable(
                    "Failed to lock FakeVramAdapter storage".to_string(),
                ))
            }
        };
        guard.remove(id);
        Ok(())
    }
}

#[test]
fn ram_tensor_is_moved_to_vram_before_gpu_compute_using_fake_vram() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_router_mem_vram");
    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(&cache_dir, vram);

    let data = vec![5u8, 6, 7, 8];
    let id = "t2";

    match mem.register_tensor_with_data(id, data, MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let mut exec = make_executor();
    let kernel = make_kernel("gpu_kernel", KernelKind::ComputeHeavy);
    let snapshot = neutral_snapshot();

    let bundle = StreamRouter::route_kernel_with_memory(
        &mut exec,
        &mut mem,
        &kernel,
        &[id],
        &snapshot,
        true,
    );

    assert_eq!(
        bundle.plan_target,
        atenia_engine::v13::execution_planner::ExecutionTarget::Gpu,
    );

    // ENQUEUE ordering: transfer then compute on GPU.
    let enqueues: Vec<&String> = exec
        .timeline
        .iter()
        .filter(|l| l.starts_with("ENQUEUE"))
        .collect();

    assert_eq!(enqueues.len(), 2);
    assert!(enqueues[0].contains("move:ram->vram:t2"));
    assert!(enqueues[0].contains("stream=Gpu"));
    assert!(enqueues[1].contains("name=gpu_kernel"));
    assert!(enqueues[1].contains("stream=Gpu"));

    exec.run_to_completion();

    // Tensor should now reside in VRAM.
    assert_eq!(mem.get_tier(id), Some(MemoryTier::Vram));

    // Ensure we see a GPU RUN for both transfer and compute.
    let runs: Vec<&String> = exec
        .timeline
        .iter()
        .filter(|l| l.starts_with("RUN"))
        .collect();

    let has_transfer_run = runs
        .iter()
        .any(|l| l.contains("move:ram->vram:t2") && l.contains("stream=Gpu"));
    let has_compute_run = runs
        .iter()
        .any(|l| l.contains("name=gpu_kernel") && l.contains("stream=Gpu"));

    if !has_transfer_run || !has_compute_run {
        panic!("expected GPU RUN entries for transfer and compute");
    }
}

#[test]
fn vram_unavailable_degrades_to_cpu() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_router_mem_vram_unavailable");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    let data = vec![9u8, 10, 11, 12];
    let id = "t3";

    match mem.register_tensor_with_data(id, data, MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let mut exec = make_executor();
    let kernel = make_kernel("gpu_like_kernel", KernelKind::ComputeHeavy);
    let snapshot = neutral_snapshot();

    let bundle = StreamRouter::route_kernel_with_memory(
        &mut exec,
        &mut mem,
        &kernel,
        &[id],
        &snapshot,
        true,
    );

    // Planner would normally choose GPU, but VRAM is unavailable so we degrade.
    assert_eq!(
        bundle.plan_target,
        atenia_engine::v13::execution_planner::ExecutionTarget::CpuFallback,
    );
    assert!(bundle.reason.contains("degraded") || bundle.reason.contains("fallback"));

    exec.run_to_completion();

    // Tensor should remain in RAM.
    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));

    // Compute must be enqueued and run on CPU.
    let mut saw_cpu_compute_enqueue = false;
    let mut saw_cpu_compute_run = false;

    for entry in &exec.timeline {
        if entry.starts_with("ENQUEUE") && entry.contains("name=gpu_like_kernel") {
            assert!(entry.contains("stream=Cpu"));
            saw_cpu_compute_enqueue = true;
        }
        if entry.starts_with("RUN") && entry.contains("name=gpu_like_kernel") {
            assert!(entry.contains("stream=Cpu"));
            saw_cpu_compute_run = true;
        }
    }

    assert!(saw_cpu_compute_enqueue);
    assert!(saw_cpu_compute_run);
}
