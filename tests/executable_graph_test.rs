use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::graph_executor::GraphExecutor;
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, MoveError, TierStatus};
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

fn make_cpu_cfg() -> StreamConfig {
    StreamConfig {
        advanced_streams_supported: true,
    }
}

#[test]
fn graph_execution_enqueues_tasks_in_node_order() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_exec_graph");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    // Register t1 on SSD and t2 in RAM.
    let t1_data = vec![1u8, 2, 3, 4];
    let _ = mem.register_tensor_with_data("t1", t1_data, MemoryTier::Ssd);
    mem.register_tensor("t2", 4, MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel1 = make_kernel("node1", KernelKind::ComputeHeavy);
    let kernel2 = make_kernel("node2", KernelKind::Small);

    graph.add_node_with_tensors(
        kernel1,
        vec!["t1".to_string()],
        vec![MemoryTier::Ssd],
    );
    graph.add_node_with_tensors(
        kernel2,
        vec!["t2".to_string()],
        vec![MemoryTier::Ram],
    );

    let snapshot = make_snapshot(0.1, 0.1);
    let cfg = make_cpu_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);

    let _bundles = ge.enqueue_graph(&graph, &mut exec, &mut mem, &snapshot, false);

    // All ENQUEUE entries for node1 (t1) must appear before the compute ENQUEUE for node2.
    let mut node2_compute_index: Option<usize> = None;
    for (idx, line) in exec.timeline.iter().enumerate() {
        if line.contains("ENQUEUE") && line.contains("name=node2") {
            node2_compute_index = Some(idx);
            break;
        }
    }

    let node2_idx = match node2_compute_index {
        Some(i) => i,
        None => panic!("no compute ENQUEUE for node2 found"),
    };

    for (idx, line) in exec.timeline.iter().enumerate() {
        if line.contains("ENQUEUE")
            && (line.contains("prefetch:t1")
                || line.contains("move:ssd->ram:t1")
                || line.contains("name=node1"))
        {
            assert!(idx < node2_idx);
        }
    }
}

#[test]
fn graph_execution_updates_memory_tiers() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_exec_graph_tiers");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    let t1_data = vec![10u8, 20, 30, 40];
    let _ = mem.register_tensor_with_data("t1", t1_data, MemoryTier::Ssd);
    mem.register_tensor("t2", 4, MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel1 = make_kernel("node1", KernelKind::ComputeHeavy);
    let kernel2 = make_kernel("node2", KernelKind::Small);

    graph.add_node_with_tensors(
        kernel1,
        vec!["t1".to_string()],
        vec![MemoryTier::Ssd],
    );
    graph.add_node_with_tensors(
        kernel2,
        vec!["t2".to_string()],
        vec![MemoryTier::Ram],
    );

    let snapshot = make_snapshot(0.1, 0.1);
    let cfg = make_cpu_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);

    let _bundles = ge.enqueue_graph(&graph, &mut exec, &mut mem, &snapshot, false);

    // After enqueue_graph, t1 should have been moved to RAM for CPU compute.
    assert_eq!(mem.get_tier("t1"), Some(MemoryTier::Ram));
    assert_eq!(mem.get_tier("t2"), Some(MemoryTier::Ram));

    exec.run_to_completion();

    // Ensure we have RUN entries for both nodes.
    let mut seen_node1 = false;
    let mut seen_node2 = false;
    for line in &exec.timeline {
        if line.contains("RUN") && line.contains("name=node1") {
            seen_node1 = true;
        }
        if line.contains("RUN") && line.contains("name=node2") {
            seen_node2 = true;
        }
    }
    assert!(seen_node1);
    assert!(seen_node2);
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
fn gpu_plan_enqueues_gpu_compute_for_compute_heavy_node() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_exec_graph_gpu");
    let vram_adapter: Box<dyn VramAdapter + Send + Sync> = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(&cache_dir, vram_adapter);

    let t_data = vec![1u8, 2, 3, 4];
    let _ = mem.register_tensor_with_data("t", t_data, MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel = make_kernel("gpu_node", KernelKind::ComputeHeavy);

    graph.add_node_with_tensors(
        kernel,
        vec!["t".to_string()],
        vec![MemoryTier::Ram],
    );

    let snapshot = make_snapshot(0.1, 0.1);
    let cfg = make_cpu_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);

    let _bundles = ge.enqueue_graph(&graph, &mut exec, &mut mem, &snapshot, true);

    // After enqueue_graph, tensor should have been moved to VRAM for GPU compute.
    assert_eq!(mem.get_tier("t"), Some(MemoryTier::Vram));

    let mut saw_transfer = false;
    let mut saw_gpu_compute = false;
    for line in &exec.timeline {
        if line.contains("ENQUEUE stream=Gpu") && line.contains("move:ram->vram:t") {
            saw_transfer = true;
        }
        if line.contains("ENQUEUE stream=Gpu") && line.contains("kind=Compute")
            && line.contains("name=gpu_node")
        {
            saw_gpu_compute = true;
        }
    }

    assert!(saw_transfer);
    assert!(saw_gpu_compute);
}
