use atenia_engine::v13::autograd::{
    execute_backward, AutogradGraph, AutogradNode, TensorGrad,
};
use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::execution_planner::ExecutionTarget;
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemoryTier, MemorySnapshot, MoveError, TierStatus};
use atenia_engine::v13::reconfigurable_graph::NodeId;
use atenia_engine::v13::streams::StreamConfig;
use atenia_engine::v13::vram_adapter::VramAdapter;

use std::collections::HashMap;
use std::sync::Mutex;

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

fn make_snapshot() -> MemorySnapshot {
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

fn make_autograd_node(
    node_id: NodeId,
    forward_target: ExecutionTarget,
    input_grads: Vec<TensorGrad>,
) -> AutogradNode {
    AutogradNode {
        node_id,
        forward_target,
        backward_target: ExecutionTarget::Cpu,
        input_grads,
        output_grads: Vec::new(),
        reason: "test node".to_string(),
    }
}

#[test]
fn backward_runs_on_gpu_when_grads_in_vram() {
    let cache_dir = "./.atenia_cache_test_autograd";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(cache_dir, vram);

    let data: Vec<u8> = (0u8..16u8).collect();

    for id in ["g1", "g2"] {
        if let Err(e) = mem.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
            panic!("register_tensor_with_data should succeed: {:?}", e);
        }
        let snapshot = make_snapshot();
        let plan = match mem.plan_move(id, MemoryTier::Vram, &snapshot) {
            Ok(p) => p,
            Err(e) => panic!("plan_move to VRAM should succeed: {:?}", e),
        };
        if let Err(e) = mem.apply_move(id, &plan) {
            panic!("apply_move to VRAM should succeed: {:?}", e);
        }
        assert_eq!(mem.get_tier(id), Some(MemoryTier::Vram));
    }

    let grads = vec![
        TensorGrad {
            id: "g1".to_string(),
            tier: MemoryTier::Vram,
        },
        TensorGrad {
            id: "g2".to_string(),
            tier: MemoryTier::Vram,
        },
    ];

    let mut graph = AutogradGraph::new();
    graph.add_node(make_autograd_node(1, ExecutionTarget::Gpu, grads));

    let snapshot = make_snapshot();
    let mut exec = make_executor();

    execute_backward(&mut exec, &mut mem, &graph, &snapshot, true);

    let mut saw_gpu_enqueue = false;

    for entry in &exec.timeline {
        if entry.starts_with("ENQUEUE")
            && entry.contains("stream=Gpu")
            && entry.contains("backward:node1")
        {
            saw_gpu_enqueue = true;
        }
    }

    assert!(saw_gpu_enqueue);

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[test]
fn backward_falls_back_to_cpu_when_grad_on_ram() {
    let cache_dir = "./.atenia_cache_test_autograd";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(cache_dir, vram);

    let data: Vec<u8> = (0u8..16u8).collect();

    if let Err(e) = mem.register_tensor_with_data("g1", data.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }
    if let Err(e) = mem.register_tensor_with_data("g2", data.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let snapshot = make_snapshot();

    let plan = match mem.plan_move("g2", MemoryTier::Vram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to VRAM should succeed: {:?}", e),
    };
    if let Err(e) = mem.apply_move("g2", &plan) {
        panic!("apply_move to VRAM should succeed: {:?}", e);
    }

    assert_eq!(mem.get_tier("g1"), Some(MemoryTier::Ram));
    assert_eq!(mem.get_tier("g2"), Some(MemoryTier::Vram));

    let grads = vec![
        TensorGrad {
            id: "g1".to_string(),
            tier: MemoryTier::Ram,
        },
        TensorGrad {
            id: "g2".to_string(),
            tier: MemoryTier::Vram,
        },
    ];

    let mut graph = AutogradGraph::new();
    graph.add_node(make_autograd_node(2, ExecutionTarget::Gpu, grads));

    let mut exec = make_executor();

    execute_backward(&mut exec, &mut mem, &graph, &snapshot, true);

    let mut saw_cpu_enqueue = false;
    let mut saw_gpu_enqueue = false;

    for entry in &exec.timeline {
        if entry.starts_with("ENQUEUE") && entry.contains("backward:node2") {
            if entry.contains("stream=Gpu") {
                saw_gpu_enqueue = true;
            }
            if entry.contains("stream=Cpu") {
                saw_cpu_enqueue = true;
            }
        }
    }

    assert!(saw_cpu_enqueue);
    assert!(!saw_gpu_enqueue);

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[test]
fn backward_moves_grads_to_ram_for_cpu() {
    let cache_dir = "./.atenia_cache_test_autograd";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(cache_dir, vram);

    for id in ["g1", "g2"] {
        // Logical SSD registration without backing data is enough for this test:
        // we only care that HybridMemoryManager updates tiers from Ssd -> Ram.
        mem.register_tensor(id, 16, MemoryTier::Ssd);
        assert_eq!(mem.get_tier(id), Some(MemoryTier::Ssd));
    }

    let grads = vec![
        TensorGrad {
            id: "g1".to_string(),
            tier: MemoryTier::Ssd,
        },
        TensorGrad {
            id: "g2".to_string(),
            tier: MemoryTier::Ssd,
        },
    ];

    let mut graph = AutogradGraph::new();
    graph.add_node(make_autograd_node(3, ExecutionTarget::Cpu, grads));

    let snapshot = make_snapshot();
    let mut exec = make_executor();

    execute_backward(&mut exec, &mut mem, &graph, &snapshot, true);

    assert_eq!(mem.get_tier("g1"), Some(MemoryTier::Ram));
    assert_eq!(mem.get_tier("g2"), Some(MemoryTier::Ram));

    let mut saw_cpu_enqueue = false;
    for entry in &exec.timeline {
        if entry.starts_with("ENQUEUE")
            && entry.contains("stream=Cpu")
            && entry.contains("backward:node3")
        {
            saw_cpu_enqueue = true;
        }
    }

    assert!(saw_cpu_enqueue);

    let _ = std::fs::remove_dir_all(cache_dir);
}
