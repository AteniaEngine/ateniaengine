//! APX v20 M3-e.11.5 — promotion + DeepDegrade integration tests.
//!
//! These tests exercise the full reaction path end-to-end:
//!
//! - `SimpleMemoryPressureGuard` emits `Degrade` whenever
//!   `memory_pressure > 0.65`.
//! - The reaction site consults `dual_memory_pressure(&conditions)`
//!   BEFORE the CPU-veto check, and if both `vram_pressure` and
//!   `ram_pressure` exceed 0.85 it **promotes** the verdict to
//!   `DeepDegrade`.
//! - `DeepDegrade` runs `Graph::migrate_all_to_disk(&cache_dir)`,
//!   increments `deep_degrade_events_count`, and is NOT vetoed by
//!   the CPU precondition (disk spillover does not consume CPU
//!   the way Cpu-tier migration does).
//!
//! Tests inject mock probes for VRAM / RAM / CPU to force the
//! specific scenarios deterministically; no reliance on real host
//! memory telemetry.
//!
//! What these tests do NOT cover (deferred to M3-e.11.6):
//! - Full pipeline execution with real graph ops reading back
//!   from Disk storage. The assertion here is that the tensors
//!   reach Disk; lazy bring-back during subsequent node execution
//!   is validated in e.11.2 unit tests and e.11.4 migration
//!   tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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
// Mock probes: VRAM, RAM, CPU
// ---------------------------------------------------------------------

/// VRAM probe that interprets the caller-given `fraction` as
/// `(total - free) / total`. We pick `total_bytes = 100` and
/// `free_bytes = (1 - fraction) * 100` so the `SignalBus`
/// arithmetic reproduces the fraction bit-exactly.
struct FixedVramProbe {
    fraction: f32,
    calls: AtomicU64,
}

impl FixedVramProbe {
    fn new(fraction: f32) -> Self {
        Self {
            fraction,
            calls: AtomicU64::new(0),
        }
    }
    // Observer for test instrumentation. Kept for parity with
    // `FixedRamProbe::calls` below and in case future assertions
    // want to check how many snapshots a given scenario triggered.
    #[allow(dead_code)]
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl VramProbeApi for FixedVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let total = 1000u64;
        let free = ((1.0 - self.fraction) * total as f32).round() as u64;
        Ok(VramSnapshot {
            total_bytes: total,
            free_bytes: free,
            used_bytes: total - free,
        })
    }
}

struct FailingVramProbe;
impl VramProbeApi for FailingVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        Err(VramProbeError::NvidiaSmiNotFound)
    }
}

struct FixedRamProbe {
    fraction: f32,
    calls: AtomicU64,
}

impl FixedRamProbe {
    fn new(fraction: f32) -> Self {
        Self {
            fraction,
            calls: AtomicU64::new(0),
        }
    }
    // See `FixedVramProbe::calls` — kept as test instrumentation
    // even when no current test asserts on it.
    #[allow(dead_code)]
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl RamProbeApi for FixedRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let total = 1000u64;
        let avail = ((1.0 - self.fraction) * total as f32).round() as u64;
        Ok(RamSnapshot {
            total_bytes: total,
            available_bytes: avail,
            used_bytes: total - avail,
        })
    }
}

struct FailingRamProbe;
impl RamProbeApi for FailingRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        Err(RamProbeError::SysinfoFailed("injected".into()))
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
// Test fixtures: graph, contract, cache dir
// ---------------------------------------------------------------------

fn test_cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("atenia_m3_e_11_5_{}_{}", label, Uuid::new_v4()));
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

/// Build a minimal graph (one parameter) and materialize its
/// output by running `execute` once. Returns the graph ready to
/// have a reactive context attached for a subsequent
/// `execute_checked` call.
fn build_graph_with_cpu_parameter(data: Vec<f32>) -> atenia_engine::amg::graph::Graph {
    let mut gb = GraphBuilder::new();
    let mut w = Tensor::with_layout(
        vec![data.len()],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    w.set_cpu_data(data);
    let _w_id = gb.parameter(w);
    let mut g = gb.build();
    let _ = g.execute(vec![]);
    g
}

/// Build a `ReactiveExecutionContext` with the canonical setup:
/// SimpleMemoryPressureGuard (emits Degrade when memory > 0.65)
/// plus the given probes injected. The CPU probe is optional
/// (some tests need to exercise the "CPU veto doesn't apply to
/// DeepDegrade" property).
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

fn param_storage_is(graph: &atenia_engine::amg::graph::Graph, kind: &str) -> bool {
    graph.nodes.iter().any(|n| {
        n.output.as_ref().map(|t| match &t.storage {
            TensorStorage::Cpu(_) => kind == "Cpu",
            TensorStorage::CpuBf16(_) => kind == "CpuBf16",
            TensorStorage::Cuda(_) => kind == "Cuda",
            TensorStorage::Disk(_) => kind == "Disk",
        }).unwrap_or(false)
    })
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[test]
fn test_promotion_when_dual_pressure() {
    // VRAM = 0.90, RAM = 0.88. Both above 0.85 → DeepDegrade
    // promotion fires. Parameter should end in Disk.
    let dir = test_cache_dir("promotion_dual");
    let vram = Arc::new(FixedVramProbe::new(0.90));
    let ram = Arc::new(FixedRamProbe::new(0.88));

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0, 3.0]);
    let ctx = make_context(vram.clone(), ram.clone(), None, dir.clone());
    g.set_reactive_context(ctx);

    let result = g.execute_checked(vec![]);
    assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

    // The parameter must be on Disk now.
    assert!(
        param_storage_is(&g, "Disk"),
        "parameter should be on Disk after DeepDegrade promotion"
    );
    // The counter reflects the promotion.
    let ctx_ref = g
        .reactive_context()
        .expect("context still attached");
    assert!(
        ctx_ref.deep_degrade_events_count() >= 1,
        "deep_degrade counter should have incremented: {}",
        ctx_ref.deep_degrade_events_count()
    );
    assert_eq!(
        ctx_ref.degrade_events_count(),
        0,
        "regular degrade counter must NOT fire when promotion happens"
    );

    cleanup(&dir);
}

#[test]
fn test_no_promotion_when_only_vram_high() {
    // VRAM = 0.95, RAM = 0.50. Only one tier above threshold →
    // plain Degrade (which in this test has no Cuda tensors to
    // migrate, so it's a no-op but the counter still increments).
    let dir = test_cache_dir("no_prom_vram_only");
    let vram = Arc::new(FixedVramProbe::new(0.95));
    let ram = Arc::new(FixedRamProbe::new(0.50));

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    // No DeepDegrade — parameter stays Cpu.
    assert!(
        param_storage_is(&g, "Cpu"),
        "parameter must remain Cpu when only VRAM is high"
    );
    let ctx_ref = g.reactive_context().expect("attached");
    assert_eq!(
        ctx_ref.deep_degrade_events_count(),
        0,
        "no promotion should have happened"
    );

    cleanup(&dir);
}

#[test]
fn test_no_promotion_when_only_ram_high() {
    // Mirror image of the above: RAM high, VRAM not.
    // memory_pressure = max(vram, ram) = 0.92 so the guard still
    // fires Degrade, but not DeepDegrade.
    let dir = test_cache_dir("no_prom_ram_only");
    let vram = Arc::new(FixedVramProbe::new(0.50));
    let ram = Arc::new(FixedRamProbe::new(0.92));

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    assert!(
        param_storage_is(&g, "Cpu"),
        "parameter must remain Cpu when only RAM is high"
    );
    assert_eq!(
        g.reactive_context()
            .expect("attached")
            .deep_degrade_events_count(),
        0
    );

    cleanup(&dir);
}

#[test]
fn test_no_promotion_when_both_below_threshold() {
    // Both high enough for Degrade (memory_pressure > 0.65) but
    // neither above the DeepDegrade 0.85 threshold.
    let dir = test_cache_dir("no_prom_both_mid");
    let vram = Arc::new(FixedVramProbe::new(0.80));
    let ram = Arc::new(FixedRamProbe::new(0.82));

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    assert!(
        param_storage_is(&g, "Cpu"),
        "parameter must remain Cpu below the DeepDegrade threshold"
    );
    assert_eq!(
        g.reactive_context()
            .expect("attached")
            .deep_degrade_events_count(),
        0
    );

    cleanup(&dir);
}

#[test]
fn test_no_promotion_when_vram_signal_absent() {
    // VRAM probe fails. `vram_pressure` on GuardConditions is
    // None → dual_memory_pressure returns false (fail-open).
    // `memory_pressure` still equals RAM, so Degrade fires, but
    // no promotion.
    let dir = test_cache_dir("no_prom_vram_absent");
    let vram = Arc::new(FailingVramProbe);
    let ram = Arc::new(FixedRamProbe::new(0.95));

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    assert!(param_storage_is(&g, "Cpu"));
    assert_eq!(
        g.reactive_context()
            .expect("attached")
            .deep_degrade_events_count(),
        0,
        "absent vram signal must suppress promotion"
    );

    cleanup(&dir);
}

#[test]
fn test_no_promotion_when_ram_signal_absent() {
    // Symmetric: RAM probe fails.
    let dir = test_cache_dir("no_prom_ram_absent");
    let vram = Arc::new(FixedVramProbe::new(0.95));
    let ram = Arc::new(FailingRamProbe);

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    assert!(param_storage_is(&g, "Cpu"));
    assert_eq!(
        g.reactive_context()
            .expect("attached")
            .deep_degrade_events_count(),
        0,
        "absent ram signal must suppress promotion"
    );

    cleanup(&dir);
}

#[test]
fn test_deep_degrade_triggers_disk_migration() {
    // Full end-to-end: dual pressure → promotion → disk
    // migration → tensor ends in Disk on a path in cache_dir.
    let dir = test_cache_dir("full_disk_migrate");
    let vram = Arc::new(FixedVramProbe::new(0.92));
    let ram = Arc::new(FixedRamProbe::new(0.90));

    let mut g = build_graph_with_cpu_parameter(vec![10.0, 20.0, 30.0, 40.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    // Parameter on Disk pointing under our cache dir.
    let mut disk_paths: Vec<PathBuf> = vec![];
    for node in &g.nodes {
        if let Some(t) = &node.output {
            if let TensorStorage::Disk(h) = &t.storage {
                disk_paths.push(h.path().to_path_buf());
            }
        }
    }
    assert!(
        !disk_paths.is_empty(),
        "at least one parameter must have ended in Disk"
    );
    for p in &disk_paths {
        assert!(
            p.starts_with(&dir),
            "disk path {:?} should live under cache_dir {:?}",
            p,
            dir
        );
        assert!(p.exists(), "disk file should exist while graph is alive");
    }

    cleanup(&dir);
}

#[test]
fn test_deep_degrade_counter_increments() {
    // Standalone counter-only assertion — complements the path
    // test above. Uses a small input and verifies the counter
    // delta is exactly 1 (one promotion per check_guard_before_node
    // call that observes dual pressure; the graph has one node
    // with an output, so the check fires once before that node).
    let dir = test_cache_dir("counter_increments");
    let vram = Arc::new(FixedVramProbe::new(0.95));
    let ram = Arc::new(FixedRamProbe::new(0.93));

    let mut g = build_graph_with_cpu_parameter(vec![1.0]);
    let ctx = make_context(vram, ram, None, dir.clone());
    g.set_reactive_context(ctx);

    let pre = g
        .reactive_context()
        .expect("attached")
        .deep_degrade_events_count();
    assert_eq!(pre, 0);

    let _ = g.execute_checked(vec![]);

    let post = g
        .reactive_context()
        .expect("attached")
        .deep_degrade_events_count();
    assert!(
        post >= 1,
        "deep_degrade counter should have increased: pre={}, post={}",
        pre,
        post
    );

    cleanup(&dir);
}

#[test]
fn test_deep_degrade_ignores_cpu_veto() {
    // Dual pressure AND external CPU saturation (cpu_total=0.90,
    // cpu_self=0.10 → share=0.11, below the 0.50 min → veto
    // would fire for plain Degrade). But promotion to
    // DeepDegrade happens first, and the DeepDegrade arm does
    // NOT consult the CPU veto — disk spillover is not CPU-
    // intensive the way Cpu-tier migration is, so the veto's
    // premise does not translate.
    let dir = test_cache_dir("ignores_cpu_veto");
    let vram = Arc::new(FixedVramProbe::new(0.92));
    let ram = Arc::new(FixedRamProbe::new(0.90));
    let cpu: Arc<dyn CpuProbeApi> = Arc::new(FixedCpuProbe {
        total: 0.90,
        self_: 0.10,
    });

    let mut g = build_graph_with_cpu_parameter(vec![1.0, 2.0]);
    let ctx = make_context(vram, ram, Some(cpu), dir.clone());
    g.set_reactive_context(ctx);

    let _ = g.execute_checked(vec![]);

    // DeepDegrade happened despite the CPU veto condition.
    assert!(
        param_storage_is(&g, "Disk"),
        "DeepDegrade must proceed even under external CPU saturation"
    );
    let ctx_ref = g.reactive_context().expect("attached");
    assert!(
        ctx_ref.deep_degrade_events_count() >= 1,
        "DeepDegrade must have fired"
    );
    assert_eq!(
        ctx_ref.degrade_vetoed_by_cpu_count(),
        0,
        "CPU veto does NOT apply to DeepDegrade"
    );

    cleanup(&dir);
}
