//! M4.7.5.e — `ensure_cpu` consumer-side audit tests.
//!
//! After M4.7.5.c+.d the reactive loop can spill an arbitrary
//! subset of graph parameters to disk between two consecutive
//! node executions. Every executor arm whose helper consumes
//! `as_cpu_slice` therefore needs a defensive
//! `ensure_cpu` / `ensure_decoded` ahead of the consumption,
//! otherwise a node whose operand was just spilled trips a
//! `Disk-resident tensor` panic inside the Tensor accessor.
//!
//! M4.7.3.d already audited the executor arms whose helpers are
//! kernel functions (`MatMul`, `BatchMatMul`, `RmsNorm`, `SiLU`,
//! `Softmax`, `RoPE`, `LogSoftmax`, `Linear`, `Activation`,
//! `FusedLinearActivation`, `Gather`, `CrossEntropyLoss`,
//! `Reshape`, `Permute`, `TransposeLastTwo`, `Conv2D`,
//! `MaxPool2D`). M4.7.5.e closes the remaining hole: the
//! `Add` / `Sub` / `Mul` arms call Tensor methods (`a.add(&b)`,
//! `a.sub(&b)`, `a.mul(&b)`) which assume Cpu storage and were
//! NOT in the M4.7.3.d audit because under that milestone's
//! workload no producer fed Disk operands into them.
//!
//! Coverage:
//!
//!   1. `add_after_disk_spill_runs_through` — graph with an Add
//!      node consuming a parameter that gets spilled between
//!      executions; second forward succeeds.
//!   2. `sub_after_disk_spill_runs_through` — same for Sub.
//!   3. `mul_after_disk_spill_runs_through` — same for Mul.
//!   4. `chained_arithmetic_post_deep_degrade_round_trip` —
//!      the full Add/Sub/Mul triplet wired in series with all
//!      params spilled, asserts numeric correctness end-to-end.

use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use atenia_engine::tensor::{Tensor, TensorStorage};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_manager::GuardManager;
use atenia_engine::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;

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

fn cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atenia_m4_7_5_e_{}_{}", label, Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn make_low_pressure_context(cache_dir: PathBuf) -> ReactiveExecutionContext {
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

fn assert_at_least_one_disk(graph: &atenia_engine::amg::graph::Graph) {
    let any_disk = graph.nodes.iter().any(|n| match &n.output {
        Some(t) => matches!(t.storage, TensorStorage::Disk(_)),
        None => false,
    });
    assert!(any_disk, "expected at least one tensor on Disk after spill");
}

#[test]
fn add_after_disk_spill_runs_through() {
    let dir = cache_dir("add_post_spill");

    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let p = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![5.0_f32]));
    let s = gb.add(in_a, p);
    let _ = gb.output(s);
    let mut graph = gb.build();

    let ctx = make_low_pressure_context(dir.clone());
    graph.set_reactive_context(ctx);

    // Warmup forward to populate the LRU.
    let _ = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![10.0])]);

    // Force a DeepDegrade (whole-graph fallback because LRU is
    // small enough that bottom-50 % covers the parameter).
    graph
        .deep_degrade_with_lru(&dir)
        .expect("orchestrator must succeed");
    assert_at_least_one_disk(&graph);

    // Re-execute. The Add arm's M4.7.5.e ensure_cpu must
    // restore the spilled parameter on the fly.
    let outputs = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![20.0])]);
    let out = outputs[0].as_cpu_slice()[0];
    assert!(
        (out - 25.0).abs() < 1e-6,
        "expected 20 + 5 = 25 after disk spill + restore, got {}",
        out
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sub_after_disk_spill_runs_through() {
    let dir = cache_dir("sub_post_spill");

    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let p = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![3.0_f32]));
    let s = gb.sub(in_a, p);
    let _ = gb.output(s);
    let mut graph = gb.build();

    let ctx = make_low_pressure_context(dir.clone());
    graph.set_reactive_context(ctx);

    let _ = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![10.0])]);
    graph.deep_degrade_with_lru(&dir).unwrap();
    assert_at_least_one_disk(&graph);

    let outputs = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![15.0])]);
    let out = outputs[0].as_cpu_slice()[0];
    assert!(
        (out - 12.0).abs() < 1e-6,
        "expected 15 - 3 = 12 after disk spill + restore, got {}",
        out
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mul_after_disk_spill_runs_through() {
    let dir = cache_dir("mul_post_spill");

    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let p = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![4.0_f32]));
    let s = gb.mul(in_a, p);
    let _ = gb.output(s);
    let mut graph = gb.build();

    let ctx = make_low_pressure_context(dir.clone());
    graph.set_reactive_context(ctx);

    let _ = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![10.0])]);
    graph.deep_degrade_with_lru(&dir).unwrap();
    assert_at_least_one_disk(&graph);

    let outputs = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![5.0])]);
    let out = outputs[0].as_cpu_slice()[0];
    assert!(
        (out - 20.0).abs() < 1e-6,
        "expected 5 * 4 = 20 after disk spill + restore, got {}",
        out
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn chained_arithmetic_post_deep_degrade_round_trip() {
    // (x + a) * b - c with a, b, c spilled. Asserts the
    // ensure_cpu calls in Add, Mul, Sub all chain correctly.
    let dir = cache_dir("chained");

    let mut gb = GraphBuilder::new();
    let in_x = gb.input();
    let a = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let b = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![3.0_f32]));
    let c = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![2.0_f32]));
    let added = gb.add(in_x, a); // x + 1
    let scaled = gb.mul(added, b); // (x + 1) * 3
    let result = gb.sub(scaled, c); // (x + 1) * 3 - 2
    let _ = gb.output(result);
    let mut graph = gb.build();

    let ctx = make_low_pressure_context(dir.clone());
    graph.set_reactive_context(ctx);

    let _ = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![5.0])]);
    graph.deep_degrade_with_lru(&dir).unwrap();
    assert_at_least_one_disk(&graph);

    // Re-run. Expect (4 + 1) * 3 - 2 = 13.
    let outputs = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![4.0])]);
    let out = outputs[0].as_cpu_slice()[0];
    assert!(
        (out - 13.0).abs() < 1e-6,
        "expected (4 + 1) * 3 - 2 = 13 after deep degrade, got {}",
        out
    );

    let _ = std::fs::remove_dir_all(&dir);
}
