//! APX v20 M3-e.11.6 — end-to-end integration tests for the
//! reactive disk tier, completing the M3-e.11 arc.
//!
//! These tests exercise the full reaction loop against a
//! realistic graph: parameter materialization, dual-pressure
//! detection, promotion, `migrate_all_to_disk`, and transparent
//! forward reads from disk-resident tensors. Mock VRAM / RAM /
//! CPU probes force specific pressure scenarios deterministically.
//!
//! Coverage matrix:
//!
//! | Test | VRAM | RAM | CPU | Expected |
//! |---|---|---|---|---|
//! | end_to_end_migrates_all_tensors | high | high | neutral | all tensors → Disk |
//! | forward_reads_disk_tensor_transparently | n/a | n/a | n/a | post-migration forward preserves output bit-exact |
//! | single_pressure_vram_shallow_degrade | high | low  | neutral | Degrade, not DeepDegrade |
//! | cpu_veto_in_non_dual_scenarios | high | low  | external | Degrade vetoed, no migration |
//! | dual_pressure_ignores_cpu_veto | high | high | external | DeepDegrade runs despite CPU |
//!
//! The "forward reads from disk transparently" test differs from
//! what the spec sketched. The original sketch expected tensors
//! to transition Disk → Cpu during forward. That does not match
//! the current executor: forward-path ops read via
//! `copy_to_cpu_vec`, which does NOT mutate storage — it reads
//! the bytes off disk each time the op runs. That is the correct
//! behavior for the current architecture (the executor does not
//! call `ensure_cpu` before ops; only `backward_checked` does so
//! via its pre-pass). The test thus validates the real invariant:
//! forward numerical results match the baseline, and the storage
//! stays on Disk. If an explicit Disk → Cpu transition is wanted,
//! that is `ensure_cpu()` at the Tensor level (validated
//! separately in `m3_e_11_2_tensor_storage_disk_test`).

use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::cpu_probe::{CpuProbeApi, CpuProbeError, CpuSnapshot};
use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use atenia_engine::tensor::{DType, Device, Layout, Tensor, TensorStorage};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_manager::GuardManager;
use atenia_engine::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;

// ---------------------------------------------------------------------
// Mock probes — same shape as in m3_e_11_5, copied for self-containment
// ---------------------------------------------------------------------

struct FixedVramProbe {
    fraction: f32,
}

impl VramProbeApi for FixedVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        let total = 1000u64;
        let free = ((1.0 - self.fraction) * total as f32).round() as u64;
        Ok(VramSnapshot {
            total_bytes: total,
            free_bytes: free,
            used_bytes: total - free,
        })
    }
}

struct FixedRamProbe {
    fraction: f32,
}

impl RamProbeApi for FixedRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        let total = 1000u64;
        let avail = ((1.0 - self.fraction) * total as f32).round() as u64;
        Ok(RamSnapshot {
            total_bytes: total,
            available_bytes: avail,
            used_bytes: total - avail,
        })
    }
}

struct FixedCpuProbe {
    total: f32,
    self_: f32,
}

impl CpuProbeApi for FixedCpuProbe {
    fn snapshot(&self) -> Result<CpuSnapshot, CpuProbeError> {
        Ok(CpuSnapshot {
            total_fraction: self.total,
            self_fraction: self.self_,
        })
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn test_cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("atenia_m3_e_11_6_{}_{}", label, Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
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

fn cpu_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    let mut t = Tensor::with_layout(
        shape,
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.set_cpu_data(data);
    t
}

/// Build a graph with 3 parameters, returning the graph after
/// `execute` has materialized every parameter's output. Gives the
/// "multiple Cpu tensors" setup Test 1 wants.
fn build_graph_with_three_params() -> atenia_engine::amg::graph::Graph {
    let mut gb = GraphBuilder::new();
    let _ = gb.parameter(cpu_tensor(vec![2], vec![1.0, 2.0]));
    let _ = gb.parameter(cpu_tensor(vec![3], vec![3.0, 4.0, 5.0]));
    let _ = gb.parameter(cpu_tensor(vec![4], vec![6.0, 7.0, 8.0, 9.0]));
    let mut g = gb.build();
    let _ = g.execute(vec![]);
    g
}

fn make_context(
    vram: Arc<dyn VramProbeApi>,
    ram: Arc<dyn RamProbeApi>,
    cpu: Option<Arc<dyn CpuProbeApi>>,
    cache_dir: PathBuf,
) -> ReactiveExecutionContext {
    let bus = Arc::new(SignalBus::with_probes(
        cpu,
        None,
        None,
        None,
        Some(vram),
        Some(ram),
    ));
    let guards: Vec<Box<dyn ExecutionGuard>> =
        vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    ReactiveExecutionContext::new_without_gc(bus, permissive_contract(), gm)
        .with_cache_dir(cache_dir)
}

fn count_by_storage(graph: &atenia_engine::amg::graph::Graph) -> (usize, usize, usize) {
    let mut cpu = 0usize;
    let mut cuda = 0usize;
    let mut disk = 0usize;
    for n in &graph.nodes {
        if let Some(t) = &n.output {
            match &t.storage {
                TensorStorage::Cpu(_) => cpu += 1,
                // CpuBf16 (M4.7.2) does not appear in the M3-e
                // migration tests, but the tally must remain
                // exhaustive over `TensorStorage`.
                TensorStorage::CpuBf16(_) => cpu += 1,
                TensorStorage::Cuda(_) => cuda += 1,
                TensorStorage::Disk(_) => disk += 1,
            }
        }
    }
    (cpu, cuda, disk)
}

fn count_tensor_files(dir: &PathBuf) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("tensor_") && n.ends_with(".bin"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

// ---------------------------------------------------------------------
// Test 1: end-to-end — dual pressure migrates every tensor to Disk
// ---------------------------------------------------------------------

#[test]
fn test_deep_degrade_end_to_end_migrates_all_tensors() {
    let dir = test_cache_dir("e2e_migrate_all");
    let vram = Arc::new(FixedVramProbe { fraction: 0.90 });
    let ram = Arc::new(FixedRamProbe { fraction: 0.88 });

    let mut g = build_graph_with_three_params();
    let (cpu_pre, _cuda_pre, disk_pre) = count_by_storage(&g);
    assert!(cpu_pre >= 3, "pre-condition: graph has >= 3 Cpu tensors");
    assert_eq!(disk_pre, 0, "pre-condition: no Disk tensors yet");

    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let result = g.execute_checked(vec![]);
    assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

    let (cpu_post, _cuda_post, disk_post) = count_by_storage(&g);
    assert!(
        disk_post >= cpu_pre,
        "every Cpu tensor must end on Disk: expected >= {}, got {}",
        cpu_pre,
        disk_post
    );
    assert_eq!(
        cpu_post, 0,
        "no Cpu tensors should remain after DeepDegrade"
    );

    // Counter checks — DeepDegrade fired, plain Degrade did not.
    let ctx_ref = g.reactive_context().expect("attached");
    assert!(
        ctx_ref.deep_degrade_events_count() >= 1,
        "deep_degrade_events_count must increment"
    );
    assert_eq!(
        ctx_ref.degrade_events_count(),
        0,
        "degrade_events_count must stay 0 when promotion happens"
    );
    assert_eq!(
        ctx_ref.degrade_vetoed_by_cpu_count(),
        0,
        "no CPU veto expected without a CPU probe"
    );

    // The cache dir has actual files.
    let files = count_tensor_files(&dir);
    assert!(
        files >= cpu_pre,
        "cache dir should contain >= {} tensor_*.bin files, got {}",
        cpu_pre,
        files
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// Test 2: forward reads from Disk transparently (no storage mutation)
// ---------------------------------------------------------------------

#[test]
fn test_forward_reads_disk_tensor_transparently() {
    // Build a graph, capture baseline output with all storage on
    // Cpu, migrate the parameter to Disk via the primitive from
    // M3-e.11.4, and re-run forward. Output must match bit-exact
    // because the CPU path reads via `copy_to_cpu_vec` which
    // transparently handles Disk. Storage stays on Disk — the
    // executor does not mutate it during forward.
    let dir = test_cache_dir("forward_reads_disk");

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let mut w = Tensor::with_layout(
        vec![2, 1],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    w.set_cpu_data(vec![3.0, 4.0]);
    let w_id = gb.parameter(w);
    let lin_id = gb.linear(x_id, w_id, None);
    let _out_id = gb.output(lin_id);
    let mut g = gb.build();

    let input_tensor = {
        let mut t = Tensor::with_layout(
            vec![1, 2],
            0.0,
            Device::CPU,
            Layout::Contiguous,
            DType::F32,
        );
        t.set_cpu_data(vec![1.0, 2.0]);
        t
    };

    // Baseline run — everything on Cpu.
    let baseline_outputs = g.execute(vec![input_tensor.clone()]);
    assert_eq!(baseline_outputs.len(), 1);
    let baseline_value = baseline_outputs[0].as_cpu_slice()[0];

    // Expected: 1*3 + 2*4 = 11.
    assert!((baseline_value - 11.0).abs() < 1e-5);

    // Now migrate every eligible Cpu tensor to Disk. After this
    // call ALL materialized outputs sit on Disk (the 1st execute
    // populated outputs for Input / Parameter / Linear / Output,
    // so migrate_all_cpu_to_disk migrates all of them).
    let report = g
        .migrate_all_cpu_to_disk(&dir)
        .expect("migrate must succeed");
    assert!(
        report.tensors_migrated >= 1,
        "at least one tensor should have been migrated"
    );

    // Specifically assert the Parameter is on Disk — it's the
    // node whose storage behavior matters for the "transparent
    // read from disk" invariant we're testing.
    assert!(
        matches!(
            g.nodes[w_id].output.as_ref().unwrap().storage,
            TensorStorage::Disk(_)
        ),
        "parameter must be on Disk after migrate"
    );

    // Forward again. Output must match baseline bit-exactly.
    // The executor re-runs Input / Linear / Output nodes (which
    // produce fresh Cpu outputs, overwriting their prior Disk
    // storage — that is the executor's normal "recompute on
    // execute" behavior, independent of the disk tier). The
    // Parameter node, however, does NOT get recomputed — its
    // output is persistent, so it stays on Disk throughout. The
    // Linear op reads the parameter via `copy_to_cpu_vec`, which
    // handles Disk transparently; the output is therefore
    // numerically identical to the baseline.
    let disk_outputs = g.execute(vec![input_tensor]);
    assert_eq!(disk_outputs.len(), 1);
    let disk_value = disk_outputs[0].as_cpu_slice()[0];
    assert_eq!(
        disk_value.to_bits(),
        baseline_value.to_bits(),
        "forward from Disk must produce bit-exact output (baseline={}, disk={})",
        baseline_value,
        disk_value
    );

    // The Parameter's storage stays on Disk — forward did NOT
    // call `ensure_cpu` to transition it. This is the real
    // invariant of the transparent-read contract: `copy_to_cpu_vec`
    // reads the bytes off disk each time the op runs, without
    // mutating the owning tensor's storage.
    assert!(
        matches!(
            g.nodes[w_id].output.as_ref().unwrap().storage,
            TensorStorage::Disk(_)
        ),
        "parameter must stay on Disk after forward reads it transparently"
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// Test 3: single-pressure VRAM still uses shallow Degrade
// ---------------------------------------------------------------------

#[test]
fn test_single_pressure_vram_still_uses_shallow_degrade() {
    let dir = test_cache_dir("single_vram_shallow");
    let vram = Arc::new(FixedVramProbe { fraction: 0.90 });
    let ram = Arc::new(FixedRamProbe { fraction: 0.30 });

    let mut g = build_graph_with_three_params();
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    // No DeepDegrade — only shallow Degrade fired.
    let ctx_ref = g.reactive_context().expect("attached");
    assert_eq!(
        ctx_ref.deep_degrade_events_count(),
        0,
        "DeepDegrade must NOT fire with only one tier pressured"
    );
    assert!(
        ctx_ref.degrade_events_count() >= 1,
        "shallow Degrade must fire with memory_pressure > threshold"
    );

    // Tensors still on Cpu (or Cuda → Cpu), not Disk.
    let (_, _, disk_count) = count_by_storage(&g);
    assert_eq!(
        disk_count, 0,
        "no tensors should be on Disk after shallow Degrade"
    );
    // No tensor files in cache dir.
    assert_eq!(count_tensor_files(&dir), 0);

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// Test 4: CPU veto still works in non-dual scenarios
// ---------------------------------------------------------------------

#[test]
fn test_cpu_veto_still_works_in_non_dual_scenarios() {
    let dir = test_cache_dir("cpu_veto_nondual");
    let vram = Arc::new(FixedVramProbe { fraction: 0.90 });
    let ram = Arc::new(FixedRamProbe { fraction: 0.30 });
    // External CPU saturation: total=0.95, self=0.20 → share
    // 0.21 < 0.50 → veto condition met.
    let cpu: Arc<dyn CpuProbeApi> = Arc::new(FixedCpuProbe {
        total: 0.95,
        self_: 0.20,
    });

    let mut g = build_graph_with_three_params();
    let ctx = make_context(vram, ram, Some(cpu), dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    let ctx_ref = g.reactive_context().expect("attached");
    assert!(
        ctx_ref.degrade_vetoed_by_cpu_count() >= 1,
        "CPU veto must fire with external saturation under shallow Degrade"
    );
    assert_eq!(
        ctx_ref.degrade_events_count(),
        0,
        "vetoed verdict must NOT be counted as a migration attempt"
    );
    assert_eq!(
        ctx_ref.deep_degrade_events_count(),
        0,
        "no DeepDegrade in non-dual scenario"
    );

    // No migrations happened — tensors unchanged.
    let (_, _, disk_count) = count_by_storage(&g);
    assert_eq!(disk_count, 0);

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// Test 5: dual pressure ignores CPU veto end-to-end
// ---------------------------------------------------------------------

#[test]
fn test_dual_pressure_ignores_cpu_veto_end_to_end() {
    let dir = test_cache_dir("dual_ignores_veto");
    let vram = Arc::new(FixedVramProbe { fraction: 0.90 });
    let ram = Arc::new(FixedRamProbe { fraction: 0.88 });
    // External CPU pressure that would normally veto shallow
    // Degrade — but promotion to DeepDegrade happens first, and
    // the DeepDegrade arm does NOT consult the veto.
    let cpu: Arc<dyn CpuProbeApi> = Arc::new(FixedCpuProbe {
        total: 0.95,
        self_: 0.20,
    });

    let mut g = build_graph_with_three_params();
    let ctx = make_context(vram, ram, Some(cpu), dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    let ctx_ref = g.reactive_context().expect("attached");
    assert!(
        ctx_ref.deep_degrade_events_count() >= 1,
        "DeepDegrade must fire even with external CPU saturation"
    );
    assert_eq!(
        ctx_ref.degrade_vetoed_by_cpu_count(),
        0,
        "CPU veto must NOT apply to DeepDegrade"
    );

    // Tensors reached disk.
    let (_, _, disk_count) = count_by_storage(&g);
    assert!(disk_count >= 3, "expected all 3 parameters on Disk, got {}", disk_count);

    cleanup(&dir);
}
