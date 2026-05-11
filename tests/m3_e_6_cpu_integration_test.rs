//! APX v20 M3-e.6 — integration tests for the CPU-precondition skip
//! path in `Graph::check_guard_before_node`.
//!
//! These tests wire an end-to-end reaction loop with a **mock** CPU
//! probe so the decision can be exercised deterministically, without
//! depending on real host CPU load. The mock implements
//! `CpuProbeApi` and returns a caller-specified snapshot; the
//! `SignalBus` is built via `SignalBus::with_cpu_probe` to inject it.
//!
//! Two behaviors are validated:
//!
//! 1. **Veto path**: Degrade triggered by a failures-based fixture
//!    guard; CPU probe fixed at (total=0.85, self=0.10) so the veto
//!    fires. After `execute_checked` we assert:
//!    - `degrade_vetoed_by_cpu_count() >= 1`
//!    - `degrade_events_count() == 0` (the verdict was intercepted
//!      before the migration counter got incremented)
//!    - The Cuda parameter is still on Cuda (no migration happened)
//!
//! 2. **No-veto path**: same fixture guard; CPU probe at
//!    (total=0.40, self=0.30) so no veto. After `execute_checked`:
//!    - `degrade_events_count() >= 1`
//!    - `degrade_vetoed_by_cpu_count() == 0`
//!    - The Cuda parameter is on Cpu (migration ran)
//!
//! A third test exercises probe-call-count monotonicity with a mock
//! probe (risk 3 from the research): rapid-fire collect calls must
//! all land on the cache, preserving probe_calls_count monotonicity
//! and not re-probing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::cpu_probe::{CpuProbeApi, CpuProbeError, CpuSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::gpu::gpu_engine;
use atenia_engine::tensor::{DType, Device, Layout, Tensor, TensorStorage};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_action::GuardAction;
use atenia_engine::v16::guards::guard_conditions::GuardConditions;
use atenia_engine::v16::guards::guard_manager::GuardManager;

// --- mock probe --------------------------------------------------

/// Deterministic probe that always returns a caller-specified pair
/// of fractions. Used to exercise the skip logic under known CPU
/// conditions, independent of the host state.
struct FixedCpuProbe {
    total: f32,
    self_: f32,
    call_count: AtomicU64,
}

impl FixedCpuProbe {
    fn new(total: f32, self_: f32) -> Self {
        Self {
            total,
            self_,
            call_count: AtomicU64::new(0),
        }
    }
    fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }
}

impl CpuProbeApi for FixedCpuProbe {
    fn snapshot(&self) -> Result<CpuSnapshot, CpuProbeError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(CpuSnapshot {
            total_fraction: self.total,
            self_fraction: self.self_,
        })
    }
}

// --- shared fixtures ---------------------------------------------

fn require_gpu(test_name: &str) -> bool {
    if gpu_engine().is_some() {
        true
    } else {
        println!(
            "[TEST:{}] no GPU available (gpu_engine() = None) -> graceful skip",
            test_name
        );
        false
    }
}

fn tensor_from(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    let mut t = Tensor::with_layout(shape, 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    t.as_cpu_slice_mut().copy_from_slice(&data);
    t
}

/// Emits `Degrade` whenever the bus reports any recorded failures.
/// Mirrors the fixture used in the M3-e.3 integration tests; the
/// signal is deterministic from the test (no dependency on real
/// memory pressure).
struct DegradeIfFailuresGuard;

impl ExecutionGuard for DegradeIfFailuresGuard {
    fn name(&self) -> &'static str {
        "degrade_if_failures_guard_m3_e_6_fixture"
    }
    fn evaluate(&self, _contract: &ExecutionContract, conditions: &GuardConditions) -> GuardAction {
        if conditions.recent_failures > 0 {
            GuardAction::Degrade
        } else {
            GuardAction::Continue
        }
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

fn build_linear_param_graph() -> (atenia_engine::amg::graph::Graph, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();

    let mut w = Tensor::with_layout(vec![2, 1], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    w.set_cpu_data(vec![3.0, 4.0]);
    let w_id = gb.parameter(w);

    let lin_id = gb.linear(x_id, w_id, None);
    let _out_id = gb.output(lin_id);

    (gb.build(), x_id, w_id)
}

fn make_x_input() -> Tensor {
    tensor_from(vec![1, 2], vec![1.0, 2.0])
}

fn make_reactive_context_with_probe(
    probe: Arc<dyn CpuProbeApi>,
    record_failure: bool,
) -> ReactiveExecutionContext {
    let bus = Arc::new(SignalBus::with_cpu_probe(probe));
    if record_failure {
        bus.failure_counter().record_failure();
    }
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(DegradeIfFailuresGuard)];
    let guard_manager = GuardManager::new(guards);
    ReactiveExecutionContext::new(bus, permissive_contract(), guard_manager)
}

// --- tests -------------------------------------------------------

#[test]
fn test_veto_fires_when_cpu_saturated_externally() {
    if !require_gpu("test_veto_fires_when_cpu_saturated_externally") {
        return;
    }

    // External pressure scenario: system at 85%, Atenia at 10%
    // (share ~= 0.12, well under 0.50). Degrade must be vetoed.
    let probe = Arc::new(FixedCpuProbe::new(0.85, 0.10));

    let (mut graph, _x_id, w_id) = build_linear_param_graph();
    let _baseline = graph.execute(vec![make_x_input()]);

    // Push the parameter to VRAM so a migration would be observable.
    {
        let w = graph.nodes[w_id]
            .output
            .as_mut()
            .expect("parameter output present");
        w.ensure_gpu().expect("ensure_gpu must succeed");
        assert!(matches!(w.storage, TensorStorage::Cuda(_)));
    }

    let ctx = make_reactive_context_with_probe(probe.clone(), true);
    assert_eq!(ctx.degrade_events_count(), 0);
    assert_eq!(ctx.degrade_vetoed_by_cpu_count(), 0);
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_x_input()]);
    assert!(
        result.is_ok(),
        "execute_checked under veto must succeed (no abort), got {:?}",
        result.err()
    );

    let ctx_after = graph.reactive_context().expect("context still attached");

    // Key veto assertions.
    assert!(
        ctx_after.degrade_vetoed_by_cpu_count() >= 1,
        "veto counter must be >= 1, got {}",
        ctx_after.degrade_vetoed_by_cpu_count()
    );
    assert_eq!(
        ctx_after.degrade_events_count(),
        0,
        "vetoed verdicts must NOT be counted as migration attempts, got {}",
        ctx_after.degrade_events_count()
    );

    // Migration did not happen: parameter stays on VRAM.
    let w_after = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("parameter output still present");
    assert!(
        matches!(w_after.storage, TensorStorage::Cuda(_)),
        "parameter must remain Cuda when Degrade is vetoed by CPU precondition"
    );
}

#[test]
fn test_no_veto_when_cpu_available_migration_happens() {
    if !require_gpu("test_no_veto_when_cpu_available_migration_happens") {
        return;
    }

    // CPU-available scenario: system at 40%, Atenia at 30% (share
    // ~= 0.75 but total is below the pressure threshold anyway).
    // No veto — migration must run exactly as in M3-e.3.
    let probe = Arc::new(FixedCpuProbe::new(0.40, 0.30));

    let (mut graph, _x_id, w_id) = build_linear_param_graph();
    let _baseline = graph.execute(vec![make_x_input()]);

    {
        let w = graph.nodes[w_id]
            .output
            .as_mut()
            .expect("parameter output present");
        w.ensure_gpu().expect("ensure_gpu must succeed");
    }

    let ctx = make_reactive_context_with_probe(probe.clone(), true);
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_x_input()]);
    assert!(result.is_ok(), "execute_checked must succeed");

    let ctx_after = graph.reactive_context().expect("context still attached");

    assert!(
        ctx_after.degrade_events_count() >= 1,
        "Degrade must have been processed at least once, got {}",
        ctx_after.degrade_events_count()
    );
    assert_eq!(
        ctx_after.degrade_vetoed_by_cpu_count(),
        0,
        "veto must NOT fire when CPU is available, got {}",
        ctx_after.degrade_vetoed_by_cpu_count()
    );

    // Migration happened: parameter is back on Cpu.
    let w_after = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("parameter output still present");
    assert!(
        matches!(w_after.storage, TensorStorage::Cpu(_)),
        "parameter must be on Cpu after Degrade-driven migration"
    );
}

// =============================================================
// Risk 3 regression: probe_calls_count monotonicity with a CPU
// probe attached. Rapid-fire collect calls must share a single
// cache slot and not re-probe the CPU either.
// =============================================================

#[test]
fn test_probe_calls_count_monotone_under_rapid_fire_with_cpu_probe() {
    // Mock probe so the test does not depend on real host state.
    let probe = Arc::new(FixedCpuProbe::new(0.20, 0.10));
    let bus = SignalBus::with_cpu_probe(probe.clone());

    // First call: triggers a probe cycle (memory + CPU) and caches
    // the values. If the memory probe is unavailable on this host
    // (e.g. CI without nvidia-smi), skip — the monotonicity property
    // is trivially satisfied on a bus that can never probe.
    let first = bus.collect_guard_conditions();
    if first.is_none() {
        println!(
            "[TEST:test_probe_calls_count_monotone_under_rapid_fire_with_cpu_probe] \
             memory probe unavailable -> graceful skip"
        );
        return;
    }

    let count_after_first = bus.probe_calls_count();
    assert_eq!(
        count_after_first, 1,
        "exactly one probe cycle must have run after the first collect"
    );
    let cpu_calls_after_first = probe.call_count();
    assert_eq!(
        cpu_calls_after_first, 1,
        "CPU probe must have been called exactly once on the cache miss"
    );

    // Twenty rapid-fire calls: all must hit the cache. Both counters
    // must stay flat.
    for _ in 0..20 {
        let _ = bus.collect_guard_conditions();
    }
    assert_eq!(
        bus.probe_calls_count(),
        count_after_first,
        "probe_calls_count must stay flat during rapid-fire cached calls \
         (before={}, after_burst={})",
        count_after_first,
        bus.probe_calls_count()
    );
    assert_eq!(
        probe.call_count(),
        cpu_calls_after_first,
        "CPU probe must not be called during rapid-fire cache hits \
         (before={}, after_burst={})",
        cpu_calls_after_first,
        probe.call_count()
    );

    // Monotonicity: counter never decreased. Trivially true given
    // the equality above, but asserted explicitly so a future
    // refactor that accidentally returns a decremented value trips
    // here first.
    assert!(bus.probe_calls_count() >= count_after_first);
    assert!(probe.call_count() >= cpu_calls_after_first);
}

#[test]
fn test_cpu_fields_populated_on_conditions_when_probe_attached() {
    // Sanity: with a probe attached, GuardConditions carries both
    // CPU fields (Some). Without this the veto logic could never
    // fire even if implemented correctly.
    let probe = Arc::new(FixedCpuProbe::new(0.33, 0.11));
    let bus = SignalBus::with_cpu_probe(probe);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_cpu_fields_populated_on_conditions_when_probe_attached] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.cpu_pressure_total, Some(0.33));
    assert_eq!(c.cpu_pressure_self, Some(0.11));
}
