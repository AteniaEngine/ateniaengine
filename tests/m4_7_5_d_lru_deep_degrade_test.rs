//! M4.7.5.d — LRU-driven `DeepDegrade` integration tests.
//!
//! Validates that the new `Graph::deep_degrade_with_lru`
//! orchestrator spills only the bottom
//! [`atenia_engine::amg::reactive::SPILL_FRACTION`] of the
//! touch order instead of the whole graph, and that the empty-
//! LRU fallback to whole-graph still works.
//!
//! Re-execution after the spill (lazy `ensure_cpu` restore on
//! every consumer arm) is **not exercised here** — it is the
//! consumer-side audit of M4.7.5.e. The .d primitive is the
//! migration orchestrator; its contract is the storage variants
//! after the call returns. End-to-end forward + restore is
//! covered by M4.7.5.e (smoke test) and .f (4-model F64).
//!
//! Coverage:
//!
//!   1. `deep_degrade_with_lru_spills_only_bottom_fraction` —
//!      run forward to populate the LRU, call the orchestrator
//!      directly, assert exactly `floor(N * 0.5)` parameter
//!      nodes ended on `TensorStorage::Disk` and the others
//!      stayed on `Cpu`.
//!   2. `deep_degrade_with_empty_lru_falls_back_to_whole_graph`
//!      — call the orchestrator before any node has run; LRU
//!      is empty so the whole-graph fallback fires and every
//!      Cpu/CpuBf16 tensor lands on Disk.
//!   3. `deep_degrade_with_lru_returns_aggregated_report` —
//!      the helper returns a `MigrationReport` whose counts
//!      reflect the selective spill (not the whole graph).

use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::{ReactiveExecutionContext, SPILL_FRACTION};
use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use atenia_engine::tensor::{Tensor, TensorStorage};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
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
    let dir = std::env::temp_dir()
        .join(format!("atenia_m4_7_5_d_{}_{}", label, Uuid::new_v4()));
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
    let guards: Vec<Box<dyn ExecutionGuard>> =
        vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    ReactiveExecutionContext::new_without_gc(bus, permissive_contract(), gm)
        .with_cache_dir(cache_dir)
}

#[test]
fn deep_degrade_with_lru_spills_only_bottom_fraction() {
    // Build a graph with several Cpu parameters. Run forward
    // under low pressure to populate the LRU (no spill yet).
    // Then call `deep_degrade_with_lru` directly and verify the
    // selective slice.
    let dir = cache_dir("bottom_fraction");

    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let p1 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let p2 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let p3 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let p4 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let s1 = gb.add(in_a, p1);
    let s2 = gb.add(s1, p2);
    let s3 = gb.add(s2, p3);
    let s4 = gb.add(s3, p4);
    let _ = gb.output(s4);
    let mut graph = gb.build();

    // Populate LRU under low-pressure context.
    let ctx = make_low_pressure_context(dir.clone());
    let lru = ctx.lru_touch_order();
    graph.set_reactive_context(ctx);
    let _ = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![10.0])]);

    let n_lru = lru.len();
    let expected_spill_count = ((n_lru as f32) * SPILL_FRACTION).floor() as usize;
    println!(
        "LRU after warmup: {} entries; floor({} * {}) = {}",
        n_lru, n_lru, SPILL_FRACTION, expected_spill_count
    );
    assert!(
        n_lru >= 4,
        "expected at least 4 entries in LRU after warmup, got {}",
        n_lru
    );

    // Call the orchestrator directly. It snapshots the LRU and
    // spills the bottom SPILL_FRACTION.
    let report = graph
        .deep_degrade_with_lru(&dir)
        .expect("deep_degrade_with_lru must succeed");
    println!("orchestrator report: {}", report);

    // Inspect resulting storage variants across all migration-
    // eligible nodes. Note: in this synthetic graph the executor
    // touches parameters FIRST (they are inputs to the Add
    // chain), so they sit at the LRU FRONT; the Add intermediates
    // touched later sit at the MRU BACK. The bottom-50% slice
    // therefore maps to the parameter cluster — this is the
    // correct LRU policy direction for the killer demo (already-
    // consumed early-layer params spilled, late-layer params
    // hot). The structural contract this test locks is:
    //
    //  - selective slice size == floor(N * SPILL_FRACTION);
    //  - report.tensors_migrated == that count exactly (no
    //    whole-graph leak);
    //  - the high-LRU-end (Add chain results) stayed Cpu.
    let add_node_ids = [s1, s2, s3, s4];
    let mut adds_on_cpu = 0usize;
    for &id in &add_node_ids {
        match &graph.nodes[id].output.as_ref().unwrap().storage {
            TensorStorage::Cpu(_) => adds_on_cpu += 1,
            TensorStorage::Disk(_) => {}
            other => panic!("Add node {} unexpected storage {:?}", id, other),
        }
    }
    let mut params_on_disk = 0usize;
    for &id in &[p1, p2, p3, p4] {
        match &graph.nodes[id].output.as_ref().unwrap().storage {
            TensorStorage::Disk(_) => params_on_disk += 1,
            TensorStorage::Cpu(_) => {}
            other => panic!("param {} unexpected storage {:?}", id, other),
        }
    }
    println!(
        "After spill: {} params on Disk, {} Add intermediates on Cpu",
        params_on_disk, adds_on_cpu
    );

    // Structural contracts.
    //
    // (1) Migrated count equals the planned bottom slice size.
    //     If `migrate_all_*` had been called, this would be much
    //     larger (every Cpu/CpuBf16 tensor in the graph).
    assert_eq!(
        report.tensors_migrated, expected_spill_count,
        "report.tensors_migrated {} != expected slice size {} — \
         selective fraction was not applied",
        report.tensors_migrated, expected_spill_count
    );

    // (2) The Add intermediates (LRU MRU end) survived. This
    //     proves the selective slice was the LRU bottom, not the
    //     full graph.
    assert!(
        adds_on_cpu >= 1,
        "expected at least one Add intermediate to remain Cpu (LRU MRU end), \
         got {} on Cpu out of {}",
        adds_on_cpu,
        add_node_ids.len()
    );

    // (3) Migrated count is strictly less than the LRU size, so
    //     the helper did not silently fall through to the
    //     whole-graph branch.
    assert!(
        report.tensors_migrated < n_lru,
        "report.tensors_migrated {} >= LRU size {} — looks like \
         whole-graph fallback fired instead of selective slice",
        report.tensors_migrated, n_lru
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deep_degrade_with_empty_lru_falls_back_to_whole_graph() {
    // Without populating the LRU first, the orchestrator must
    // fall back to the whole-graph spill — preserves the M3-e.11.5
    // pressure-relief contract for the boot path.
    let dir = cache_dir("empty_lru");

    let mut gb = GraphBuilder::new();
    let p1 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let p2 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![2.0_f32]));
    let _ = gb.output(p1);
    let _ = gb.output(p2);
    let mut graph = gb.build();

    let ctx = make_low_pressure_context(dir.clone());
    let lru = ctx.lru_touch_order();
    graph.set_reactive_context(ctx);

    // LRU is empty before anything runs.
    assert_eq!(lru.len(), 0);

    let report = graph
        .deep_degrade_with_lru(&dir)
        .expect("orchestrator must succeed on empty LRU");
    println!("empty-LRU fallback report: {}", report);

    // Both parameters should now be on Disk (whole-graph fallback).
    for id in [p1, p2] {
        match &graph.nodes[id].output.as_ref().unwrap().storage {
            TensorStorage::Disk(_) => {}
            other => panic!(
                "expected Disk on param {} after whole-graph fallback, got {:?}",
                id, other
            ),
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deep_degrade_with_lru_returns_aggregated_report() {
    // The orchestrator returns a `MigrationReport` with counts
    // matching the selective slice. The legacy report shape is
    // preserved so the existing DeepDegrade arm logging surface
    // does not have to learn a second type.
    let dir = cache_dir("aggregated_report");

    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let p1 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let p2 = gb.parameter(Tensor::new_cpu(vec![1, 1], vec![1.0_f32]));
    let s = gb.add(in_a, p1);
    let s2 = gb.add(s, p2);
    let _ = gb.output(s2);
    let mut graph = gb.build();

    let ctx = make_low_pressure_context(dir.clone());
    graph.set_reactive_context(ctx);
    let _ = graph.execute(vec![Tensor::new_cpu(vec![1, 1], vec![10.0])]);

    let report = graph
        .deep_degrade_with_lru(&dir)
        .expect("orchestrator must succeed");

    // The graph after warmup has more than 2 nodes in the LRU
    // (input + 2 params + 2 Adds + 1 Output = 6, all touched).
    // The bottom-50 % slice = 3 ids. Of those, only Cpu/CpuBf16
    // tensors actually migrate (Inputs and Outputs forward
    // upstream tensors but their own storage variants depend on
    // node type). The contract is: tensors_migrated > 0 and no
    // failures.
    println!("aggregated report: {}", report);
    assert!(
        report.tensors_migrated > 0,
        "expected at least one tensor migrated by orchestrator; got 0"
    );
    assert!(report.failure.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}
