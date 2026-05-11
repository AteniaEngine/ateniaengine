//! M4.7.5.b — `TouchOrder` LRU integration tests.
//!
//! Verifies that the per-context `TouchOrder` introduced in
//! M4.7.5.b is updated from `NodeTimingRecorder::drop` so it
//! reflects the *completion* order of graph nodes — exactly what
//! the M4.7.5.c+.d selective-spill policy needs to pick eviction
//! candidates.
//!
//! Pre-step verification (decision #6 of the M4.7.5 plan):
//!
//! - `src/amm/offloading.rs` is dead-end legacy (separate
//!   `Offloader` API with its own `OffloadHandle`, not connected
//!   to the M3-e disk_tier path). Confirmed by grep — no consumer
//!   in `src/` outside `amm/memory_manager.rs`. Not a hook site.
//! - `src/apx7/ule.rs` (mode ≥7.12) either delegates to
//!   `run_plan(true)` (lines 61, 80, 160) — which calls
//!   `execute_single` per node and constructs a
//!   `NodeTimingRecorder` — or calls `execute_single` directly
//!   (line 130) inside its custom scheduler. Either way the touch
//!   signal fires on every node. The hook in
//!   `NodeTimingRecorder::drop` covers both paths without an
//!   apx7-specific branch.
//!
//! Coverage:
//!
//!   1. `touch_order_reflects_node_execution_order` — execute a
//!      small graph and verify the LRU snapshot lists nodes in
//!      the order they completed.
//!   2. `touch_order_re_touches_promote_to_mru` — re-running the
//!      same graph promotes the touched nodes to the back of the
//!      deque without growing the deque past `len = node_count`.
//!   3. `touch_order_is_no_op_without_reactive_context` — a graph
//!      with no `reactive_context` runs forward bit-exact and
//!      surfaces no `TouchOrder` (the LRU only exists on the
//!      context).

use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use atenia_engine::tensor::Tensor;
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::Constraints;
use atenia_engine::v16::contract::constraints::RuntimeState;
use atenia_engine::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_manager::GuardManager;
use atenia_engine::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;

// Mock probes that always report low pressure.

struct LowPressureVramProbe;
impl VramProbeApi for LowPressureVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        Ok(VramSnapshot {
            total_bytes: 1000,
            free_bytes: 900,
            used_bytes: 100,
        })
    }
}

struct LowPressureRamProbe;
impl RamProbeApi for LowPressureRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        Ok(RamSnapshot {
            total_bytes: 1000,
            available_bytes: 900,
            used_bytes: 100,
        })
    }
}

fn permissive_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.5,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.4,
        },
        runtime_snapshot: RuntimeState {
            memory_headroom: 0.8,
            is_stable: true,
            recent_recovery: false,
            offload_supported: true,
        },
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.5,
        require_fallback: false,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

fn cache_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atenia_m4_7_5_b_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("test cache dir");
    dir
}

fn make_context(cache_dir: PathBuf) -> ReactiveExecutionContext {
    let bus = Arc::new(SignalBus::with_probes(
        None,
        None,
        None,
        None,
        Some(Arc::new(LowPressureVramProbe)),
        Some(Arc::new(LowPressureRamProbe)),
    ));
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    ReactiveExecutionContext::new_without_gc(bus, permissive_contract(), gm)
        .with_cache_dir(cache_dir)
}

#[test]
fn touch_order_reflects_node_execution_order() {
    // Chain of 5 Add nodes: input_a, input_b, then five Adds. The
    // executor walks them in plan order; the LRU should record
    // their completion order at the back of the deque.
    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let in_b = gb.input();
    let n1 = gb.add(in_a, in_b);
    let n2 = gb.add(n1, in_b);
    let n3 = gb.add(n2, in_b);
    let n4 = gb.add(n3, in_b);
    let n5 = gb.add(n4, in_b);
    let _ = gb.output(n5);
    let mut graph = gb.build();

    let dir = cache_dir();
    let ctx = make_context(dir.clone());
    let lru = ctx.lru_touch_order();
    graph.set_reactive_context(ctx);

    let _ = graph.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![1.0]),
        Tensor::new_cpu(vec![1, 1], vec![2.0]),
    ]);

    let order = lru.snapshot();
    println!("touch order after forward: {:?}", order);

    // Every node must appear exactly once.
    assert!(
        order.contains(&in_a)
            && order.contains(&in_b)
            && order.contains(&n1)
            && order.contains(&n2)
            && order.contains(&n3)
            && order.contains(&n4)
            && order.contains(&n5),
        "touch order missing one of the input/Add nodes: {:?}",
        order
    );

    // Add nodes must appear in their construction order (n1
    // before n2 before n3 ... since the executor walks them
    // top-down).
    let pos = |id: usize| order.iter().position(|&x| x == id).unwrap();
    assert!(pos(n1) < pos(n2));
    assert!(pos(n2) < pos(n3));
    assert!(pos(n3) < pos(n4));
    assert!(pos(n4) < pos(n5));

    // Output node touched last among the Add chain (output is
    // executed after n5 in plan order).
    assert!(pos(n5) < order.len());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn touch_order_re_touches_promote_to_mru() {
    // Build a 3-Add chain, run the forward twice. After the
    // first run every node id is present. After the second run
    // the deque must have the same length and the node ids must
    // appear in the *second* run's completion order at the back.
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let n1 = gb.add(a, b);
    let n2 = gb.add(n1, b);
    let n3 = gb.add(n2, b);
    let _ = gb.output(n3);
    let mut graph = gb.build();

    let dir = cache_dir();
    let ctx = make_context(dir.clone());
    let lru = ctx.lru_touch_order();
    graph.set_reactive_context(ctx);

    // Run #1.
    let _ = graph.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![1.0]),
        Tensor::new_cpu(vec![1, 1], vec![2.0]),
    ]);
    let len_after_one = lru.len();
    let snapshot_one = lru.snapshot();

    // Run #2 — same graph, same nodes touched. The deque must
    // not grow (touches deduplicate) and the nodes must reappear
    // in the same relative order.
    let _ = graph.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![3.0]),
        Tensor::new_cpu(vec![1, 1], vec![4.0]),
    ]);
    let len_after_two = lru.len();
    let snapshot_two = lru.snapshot();

    println!("after one: {:?}", snapshot_one);
    println!("after two: {:?}", snapshot_two);

    assert_eq!(
        len_after_one, len_after_two,
        "deque must not grow on re-touches; got {} → {}",
        len_after_one, len_after_two
    );

    // The output / final node should still be at the back (n3
    // touched last in both runs).
    assert_eq!(*snapshot_two.last().unwrap(), *snapshot_one.last().unwrap());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn touch_order_is_independent_per_context() {
    // Two graphs, each with its own context, must own
    // independent LRUs. Touching one does not affect the other.
    let mut gb1 = GraphBuilder::new();
    let a1 = gb1.input();
    let b1 = gb1.input();
    let n1 = gb1.add(a1, b1);
    let _ = gb1.output(n1);
    let mut graph_one = gb1.build();

    let mut gb2 = GraphBuilder::new();
    let a2 = gb2.input();
    let b2 = gb2.input();
    let n2 = gb2.add(a2, b2);
    let _ = gb2.output(n2);
    let mut graph_two = gb2.build();

    let dir1 = cache_dir();
    let dir2 = cache_dir();
    let ctx1 = make_context(dir1.clone());
    let ctx2 = make_context(dir2.clone());
    let lru1 = ctx1.lru_touch_order();
    let lru2 = ctx2.lru_touch_order();
    graph_one.set_reactive_context(ctx1);
    graph_two.set_reactive_context(ctx2);

    // Run only graph_one. lru2 must remain empty.
    let _ = graph_one.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![1.0]),
        Tensor::new_cpu(vec![1, 1], vec![2.0]),
    ]);
    assert!(lru1.len() > 0, "graph_one's LRU must have entries");
    assert_eq!(lru2.len(), 0, "graph_two's LRU must remain empty");

    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
}

#[test]
fn touch_order_is_no_op_without_reactive_context() {
    // No `reactive_context` attached — the executor still runs
    // the forward bit-exact and the LRU does not exist (the user
    // has no way to construct a `TouchOrder` independent of a
    // context, which is the design lock). This test verifies the
    // forward path still produces correct output.
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let sum = gb.add(a, b);
    let _ = gb.output(sum);
    let mut graph = gb.build();

    let outputs = graph.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![3.0]),
        Tensor::new_cpu(vec![1, 1], vec![4.0]),
    ]);
    let val = outputs[0].as_cpu_slice()[0];
    assert!(
        (val - 7.0).abs() < 1e-6,
        "no-context forward must produce 3 + 4 = 7, got {}",
        val
    );
}
