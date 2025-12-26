use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::batch_loop::BatchLoopRunner;
use atenia_engine::v13::graph_executor::GraphExecutor;
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::kernel_model::{KernelKind, KernelProfile};
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};
use atenia_engine::v13::offload_engine::SmartOffloadEngine;
use atenia_engine::v13::reconfigurable_graph::ReconfigurableGraph;
use atenia_engine::v13::streams::StreamConfig;

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

#[test]
fn offload_runs_between_ticks_and_moves_ram_to_ssd_when_ram_high() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_batch_loop");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    // Register t1 in RAM with real bytes.
    let data = vec![1u8, 2, 3, 4];
    let _ = mem.register_tensor_with_data("t1", data, MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel = make_kernel("cpu_node", KernelKind::Small);
    graph.add_node_with_tensors(
        kernel,
        vec!["t1".to_string()],
        vec![MemoryTier::Ram],
    );

    let snapshots = vec![
        make_snapshot(0.1, 0.1),  // tick 0: low RAM pressure
        make_snapshot(0.1, 0.99), // tick 1: high RAM pressure
    ];

    let mut offload = SmartOffloadEngine::default();
    offload.max_actions_per_tick = 2;

    let cfg = make_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);
    let mut runner = BatchLoopRunner::new(offload, ge);

    let _timeline = runner.run_ticks(&graph, &mut exec, &mut mem, &snapshots, false);

    // After running both ticks, offload should have moved t1 to SSD.
    assert_eq!(mem.get_tier("t1"), Some(MemoryTier::Ssd));

    // Check timeline markers.
    let mut saw_tick0_start = false;
    let mut saw_tick0_end = false;
    let mut saw_offload_plan_tick1 = false;
    let mut saw_offload_apply_ok_tick1 = false;

    for line in &exec.timeline {
        if line.contains("TICK_START tick=0") {
            saw_tick0_start = true;
        }
        if line.contains("TICK_END tick=0") {
            saw_tick0_end = true;
        }
        if line.contains("OFFLOAD_PLAN tick=1") && line.contains("actions=1") {
            saw_offload_plan_tick1 = true;
        }
        if line.contains("OFFLOAD_APPLY tick=1 ok") {
            saw_offload_apply_ok_tick1 = true;
        }
    }

    assert!(saw_tick0_start);
    assert!(saw_tick0_end);
    assert!(saw_offload_plan_tick1);
    assert!(saw_offload_apply_ok_tick1);
}

#[test]
fn hysteresis_stable_band_does_not_thrash_between_ticks() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_batch_loop_hys");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    let data = vec![5u8, 6, 7, 8];
    let _ = mem.register_tensor_with_data("t2", data, MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel = make_kernel("cpu_node", KernelKind::Small);
    graph.add_node_with_tensors(
        kernel,
        vec!["t2".to_string()],
        vec![MemoryTier::Ram],
    );

    let snapshots = vec![
        make_snapshot(0.1, 0.90),
        make_snapshot(0.1, 0.90),
    ];

    let offload = SmartOffloadEngine::default();

    let cfg = make_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);
    let mut runner = BatchLoopRunner::new(offload, ge);

    let _timeline = runner.run_ticks(&graph, &mut exec, &mut mem, &snapshots, false);

    // t2 should remain in RAM because we are in the hysteresis stable band.
    assert_eq!(mem.get_tier("t2"), Some(MemoryTier::Ram));

    // OFFLOAD_PLAN actions should be 0 for both ticks.
    let mut plan_tick0_zero = false;
    let mut plan_tick1_zero = false;
    for line in &exec.timeline {
        if line.contains("OFFLOAD_PLAN tick=0") && line.contains("actions=0") {
            plan_tick0_zero = true;
        }
        if line.contains("OFFLOAD_PLAN tick=1") && line.contains("actions=0") {
            plan_tick1_zero = true;
        }
    }

    assert!(plan_tick0_zero);
    assert!(plan_tick1_zero);
}

#[test]
fn priority_budget_limits_actions_per_tick() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_batch_loop_pri");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    // Register three tensors in RAM with different sizes.
    let _ = mem.register_tensor_with_data("tA", vec![1u8; 10], MemoryTier::Ram);
    let _ = mem.register_tensor_with_data("tB", vec![2u8; 100], MemoryTier::Ram);
    let _ = mem.register_tensor_with_data("tC", vec![3u8; 50], MemoryTier::Ram);

    let mut graph = ReconfigurableGraph::new();
    let kernel = make_kernel("cpu_node", KernelKind::Small);
    graph.add_node_with_tensors(
        kernel,
        vec!["tA".to_string(), "tB".to_string(), "tC".to_string()],
        vec![MemoryTier::Ram, MemoryTier::Ram, MemoryTier::Ram],
    );

    let snapshots = vec![make_snapshot(0.1, 0.99)];

    let mut offload = SmartOffloadEngine::default();
    offload.max_actions_per_tick = 2;

    let cfg = make_cfg();
    let mut exec = AsyncExecutor::new(cfg);
    let ge = GraphExecutor::new(cfg);
    let mut runner = BatchLoopRunner::new(offload, ge);

    let _timeline = runner.run_ticks(&graph, &mut exec, &mut mem, &snapshots, false);

    // Under high RAM pressure with budget=2, only the two largest tensors
    // should be offloaded to SSD.
    assert_eq!(mem.get_tier("tB"), Some(MemoryTier::Ssd));
    assert_eq!(mem.get_tier("tC"), Some(MemoryTier::Ssd));
    assert_eq!(mem.get_tier("tA"), Some(MemoryTier::Ram));
}
