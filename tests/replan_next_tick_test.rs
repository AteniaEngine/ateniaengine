use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::batch_loop::{BatchLoopRunner, TickResult};
use atenia_engine::v13::execution_planner::ExecutionTarget;
use atenia_engine::v13::graph_executor::GraphExecutor;
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, MoveError, TierStatus};
use atenia_engine::v13::offload_engine::SmartOffloadEngine;
use atenia_engine::v13::reconfigurable_graph::ReconfigurableGraph;
use atenia_engine::v13::streams::StreamConfig;
use atenia_engine::v13::vram_adapter::VramAdapter;

fn make_cache_dir(name: &str) -> String {
    let _ = std::fs::remove_dir_all(name);
    name.to_string()
}

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

fn make_cfg() -> StreamConfig {
    StreamConfig {
        advanced_streams_supported: true,
    }
}

#[derive(Clone)]
struct FakeVramAdapter {
    store: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl FakeVramAdapter {
    fn new() -> Self {
        FakeVramAdapter {
            store: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl VramAdapter for FakeVramAdapter {
    fn is_available(&self) -> bool {
        true
    }

    fn upload(&self, key: &str, data: &[u8]) -> Result<(), MoveError> {
        match self.store.lock() {
            Ok(mut guard) => {
                guard.insert(key.to_string(), data.to_vec());
                Ok(())
            }
            Err(_) => Err(MoveError::BackendUnavailable(
                "VRAM adapter mutex poisoned".to_string(),
            )),
        }
    }

    fn download(&self, key: &str) -> Result<Vec<u8>, MoveError> {
        match self.store.lock() {
            Ok(guard) => match guard.get(key) {
                Some(bytes) => Ok(bytes.clone()),
                None => Err(MoveError::BackendUnavailable(
                    "Key not found in fake VRAM".to_string(),
                )),
            },
            Err(_) => Err(MoveError::BackendUnavailable(
                "VRAM adapter mutex poisoned".to_string(),
            )),
        }
    }

    fn free(&self, key: &str) -> Result<(), MoveError> {
        match self.store.lock() {
            Ok(mut guard) => {
                let _ = guard.remove(key);
                Ok(())
            }
            Err(_) => Err(MoveError::BackendUnavailable(
                "VRAM adapter mutex poisoned".to_string(),
            )),
        }
    }
}

#[test]
fn offload_changes_next_tick_placement() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_replan_next");
    let vram_adapter: Box<dyn VramAdapter + Send + Sync> = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(&cache_dir, vram_adapter);

    // Register t1 in RAM with enough bytes to be a good offload candidate.
    let data = vec![1u8; 128];
    let _ = mem.register_tensor_with_data("t1", data, MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel = make_kernel("heavy_node", KernelKind::ComputeHeavy);
    graph.add_node_with_tensors(
        kernel,
        vec!["t1".to_string()],
        vec![MemoryTier::Ram],
    );

    let snapshots = vec![
        make_snapshot(0.10, 0.10), // tick0: low pressure
        make_snapshot(0.99, 0.99), // tick1: high VRAM and RAM pressure for offload to SSD
        make_snapshot(0.10, 0.10), // tick2: low pressure again, but tiers changed by offload
    ];

    let mut offload = SmartOffloadEngine::default();
    // Ensure offload can act immediately and has at least one action available.
    offload.cooldown_ticks = 0;
    offload.max_actions_per_tick = 4;

    let cfg = make_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);
    let mut runner = BatchLoopRunner::new(offload, ge);

    let (_timeline, tick_results): (Vec<String>, Vec<TickResult>) =
        runner.run_ticks_with_plans(&graph, &mut exec, &mut mem, &snapshots, true);

    assert!(tick_results.len() >= 3);

    let plan0 = &tick_results[0].plan;
    let plan2 = &tick_results[2].plan;

    // At tick0, tensor is in RAM, GPU available, low pressure: prefer GPU.
    assert_eq!(plan0.placements.len(), 1);
    assert_eq!(plan0.placements[0].target, ExecutionTarget::Gpu);

    // At tick2, the planner should see the tensor on SSD and force CPU.
    assert_eq!(plan2.placements.len(), 1);
    let placement2 = &plan2.placements[0];
    assert!(
        placement2.target == ExecutionTarget::Cpu
            || placement2.target == ExecutionTarget::CpuFallback
    );
    assert!(placement2.reason.to_lowercase().contains("ssd"));

    // Also ensure the offload actually ran on tick1.
    let mut saw_offload_plan_tick1 = false;
    let mut saw_offload_apply_ok_tick1 = false;
    for line in &exec.timeline {
        if line.contains("OFFLOAD_PLAN tick=1") && line.contains("actions=1") {
            saw_offload_plan_tick1 = true;
        }
        if line.contains("OFFLOAD_APPLY tick=1 ok") {
            saw_offload_apply_ok_tick1 = true;
        }
    }

    assert!(saw_offload_plan_tick1);
    assert!(saw_offload_apply_ok_tick1);
}
