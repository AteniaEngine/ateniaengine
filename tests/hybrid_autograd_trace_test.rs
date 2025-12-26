use atenia_engine::v13::autograd::{
    execute_backward_with_trace, AutogradGraph, AutogradNode, GradMoveRecord, TensorGrad,
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

struct FailingVramAdapter;

impl VramAdapter for FailingVramAdapter {
    fn is_available(&self) -> bool {
        true
    }

    fn upload(&self, _id: &str, _data: &[u8]) -> Result<(), MoveError> {
        Err(MoveError::BackendUnavailable(
            "Simulated upload failure".to_string(),
        ))
    }

    fn download(&self, _id: &str) -> Result<Vec<u8>, MoveError> {
        Err(MoveError::BackendUnavailable(
            "Simulated download failure".to_string(),
        ))
    }

    fn free(&self, _id: &str) -> Result<(), MoveError> {
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
    reason: &str,
) -> AutogradNode {
    AutogradNode {
        node_id,
        forward_target,
        backward_target: ExecutionTarget::Cpu,
        input_grads,
        output_grads: Vec::new(),
        reason: reason.to_string(),
    }
}

#[test]
fn trace_records_moves_and_target_gpu_success() {
    let cache_dir = "./.atenia_cache_test_autograd_trace";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(cache_dir, vram);

    let data: Vec<u8> = (0u8..16u8).collect();

    for id in ["g1", "g2"] {
        if let Err(e) = mem.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
            panic!("register_tensor_with_data should succeed: {:?}", e);
        }
        assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));
    }

    // Logical tiers for planning: we want GPU backward, so mark grads as Vram
    // even though the physical memory starts in RAM. execute_backward_with_trace
    // will plan and apply the RAM -> VRAM moves.
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
    graph.add_node(make_autograd_node(1, ExecutionTarget::Gpu, grads, "gpu success"));

    let snapshot = make_snapshot();
    let mut exec = make_executor();

    let trace = execute_backward_with_trace(&mut exec, &mut mem, &graph, &snapshot, true);

    assert_eq!(trace.nodes.len(), 1);
    let node_trace = &trace.nodes[0];

    assert_eq!(node_trace.requested_backward_target, ExecutionTarget::Gpu);
    assert_eq!(node_trace.final_backward_target, ExecutionTarget::Gpu);

    // Expect planned and applied moves for both gradients.
    let mut planned_ids = Vec::new();
    let mut applied_ids = Vec::new();

    for rec in &node_trace.moves {
        match rec {
            GradMoveRecord::Planned { grad_id, from, to } => {
                planned_ids.push(grad_id.clone());
                assert_eq!(*from, MemoryTier::Ram);
                assert_eq!(*to, MemoryTier::Vram);
            }
            GradMoveRecord::Applied { grad_id, from, to } => {
                applied_ids.push(grad_id.clone());
                assert_eq!(*from, MemoryTier::Ram);
                assert_eq!(*to, MemoryTier::Vram);
            }
            GradMoveRecord::Failed { .. } => {
                panic!("did not expect any Failed move records in gpu_success test");
            }
            GradMoveRecord::Skipped { .. } => {}
        }
    }

    assert!(planned_ids.contains(&"g1".to_string()));
    assert!(planned_ids.contains(&"g2".to_string()));
    assert!(applied_ids.contains(&"g1".to_string()));
    assert!(applied_ids.contains(&"g2".to_string()));

    let mut saw_autograd_line = false;
    for entry in &exec.timeline {
        if entry.starts_with("AUTOGRAD") && entry.contains("node=1") {
            saw_autograd_line = true;
            assert!(entry.contains("target=Gpu"));
            assert!(entry.contains("fallback=0"));
        }
    }

    assert!(saw_autograd_line);

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[test]
fn trace_records_fallback_when_gpu_move_fails() {
    let cache_dir = "./.atenia_cache_test_autograd_trace";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FailingVramAdapter);
    let mut mem = HybridMemoryManager::new_with_vram(cache_dir, vram);

    let data: Vec<u8> = (0u8..16u8).collect();

    for id in ["g1", "g2"] {
        if let Err(e) = mem.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
            panic!("register_tensor_with_data should succeed: {:?}", e);
        }
        assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));
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
    graph.add_node(make_autograd_node(2, ExecutionTarget::Gpu, grads, "gpu may fail"));

    let snapshot = make_snapshot();
    let mut exec = make_executor();

    let trace = execute_backward_with_trace(&mut exec, &mut mem, &graph, &snapshot, true);

    assert_eq!(trace.nodes.len(), 1);
    let node_trace = &trace.nodes[0];

    assert_eq!(node_trace.requested_backward_target, ExecutionTarget::Gpu);
    assert_eq!(node_trace.final_backward_target, ExecutionTarget::Cpu);
    assert!(node_trace.reason.to_lowercase().contains("fallback"));

    let mut saw_failed = false;
    for rec in &node_trace.moves {
        if let GradMoveRecord::Failed { reason, .. } = rec {
            saw_failed = true;
            assert!(reason.to_lowercase().contains("simulated upload failure")
                || reason.to_lowercase().contains("backendunavailable"));
        }
    }

    assert!(saw_failed);

    let mut saw_autograd_line = false;
    for entry in &exec.timeline {
        if entry.starts_with("AUTOGRAD") && entry.contains("node=2") {
            saw_autograd_line = true;
            assert!(entry.contains("target=Cpu"));
            assert!(entry.contains("fallback=1"));
        }
    }

    assert!(saw_autograd_line);

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[test]
fn trace_skips_moves_when_already_in_target() {
    let cache_dir = "./.atenia_cache_test_autograd_trace";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(cache_dir, vram);

    for id in ["g1", "g2"] {
        mem.register_tensor(id, 16, MemoryTier::Ram);
        assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));
    }

    let grads = vec![
        TensorGrad {
            id: "g1".to_string(),
            tier: MemoryTier::Ram,
        },
        TensorGrad {
            id: "g2".to_string(),
            tier: MemoryTier::Ram,
        },
    ];

    let mut graph = AutogradGraph::new();
    graph.add_node(make_autograd_node(3, ExecutionTarget::Cpu, grads, "cpu no moves"));

    let snapshot = make_snapshot();
    let mut exec = make_executor();

    let trace = execute_backward_with_trace(&mut exec, &mut mem, &graph, &snapshot, true);

    assert_eq!(trace.nodes.len(), 1);
    let node_trace = &trace.nodes[0];

    assert_eq!(node_trace.requested_backward_target, ExecutionTarget::Cpu);
    assert_eq!(node_trace.final_backward_target, ExecutionTarget::Cpu);

    // All moves should be Skipped due to already being in the target tier.
    assert!(!node_trace.moves.is_empty());
    for rec in &node_trace.moves {
        match rec {
            GradMoveRecord::Skipped { reason, .. } => {
                assert!(reason.to_lowercase().contains("already in target tier"));
            }
            other => {
                panic!("expected only Skipped records, got {:?}", other);
            }
        }
    }

    let mut saw_autograd_line = false;
    for entry in &exec.timeline {
        if entry.starts_with("AUTOGRAD") && entry.contains("node=3") {
            saw_autograd_line = true;
            assert!(entry.contains("target=Cpu"));
            assert!(entry.contains("fallback=0"));
        }
    }

    assert!(saw_autograd_line);

    let _ = std::fs::remove_dir_all(cache_dir);
}
