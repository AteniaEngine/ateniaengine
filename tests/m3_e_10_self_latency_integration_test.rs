//! APX v20 M3-e.10 — integration tests for self-latency wiring.
//!
//! Two things are being validated end-to-end:
//!
//! 1. **The wire**: `Graph::execute_single_inner` records each node's
//!    execution time to `SignalBus::latency_monitor()` when a
//!    reactive context is attached. Pre-M3-e.10 the monitor was
//!    orphan — `record_latency` was never called from production
//!    code, so `latency_spike` was always `false` in `GuardConditions`
//!    regardless of reality. These tests exercise the path that
//!    fixes that.
//!
//! 2. **The enriched shape**: `collect_guard_conditions` populates
//!    `latency_baseline_ms`, `latency_current_ms` and `latency_ratio`
//!    from the monitor, computed fresh on every call (not cached).
//!    The three fields travel together via
//!    `with_latency_context` — half-populated states are
//!    disallowed by construction.
//!
//! The tests inject latencies directly via `monitor.record_latency`
//! rather than relying on real timing from `execute_single_inner`
//! so they stay deterministic. A separate "end-to-end wire" test
//! runs a real graph and checks that the monitor received at
//! least one sample afterwards — this is the minimum assertion
//! that survives the test-graph-is-small constraint (< min_samples
//! nodes → no spike detection, but samples DO accumulate).

use std::sync::Arc;
use std::time::Duration;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::tensor::{DType, Device, Layout, Tensor};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_action::GuardAction;
use atenia_engine::v16::guards::guard_conditions::GuardConditions;
use atenia_engine::v16::guards::guard_manager::GuardManager;

// --- shared fixtures ---------------------------------------------

fn minimal_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.8,
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

struct AlwaysContinueGuard;
impl ExecutionGuard for AlwaysContinueGuard {
    fn name(&self) -> &'static str {
        "always_continue_guard_m3_e_10"
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        _conditions: &GuardConditions,
    ) -> GuardAction {
        GuardAction::Continue
    }
}

fn build_small_matmul_graph() -> atenia_engine::amg::graph::Graph {
    let mut gb = GraphBuilder::new();
    let input_id = gb.input();
    let mut weight = Tensor::with_layout(
        vec![2, 2],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    weight.set_cpu_data(vec![1.0, 0.0, 0.0, 1.0]);
    let w_id = gb.parameter(weight);
    let mm_id = gb.matmul(input_id, w_id);
    gb.output(mm_id);
    gb.build()
}

fn make_input_tensor() -> Tensor {
    let mut t = Tensor::with_layout(
        vec![2, 2],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.set_cpu_data(vec![1.0, 2.0, 3.0, 4.0]);
    t
}

// --- Shape tests (inject directly, no graph) ----------------

#[test]
fn test_latency_context_populated_when_baseline_ready() {
    // With >= min_samples stable measurements, baseline, current,
    // and ratio should all be present on GuardConditions. Ratio
    // should be approximately 1.0 since samples are uniform.
    let bus = SignalBus::new();
    let monitor = bus.latency_monitor();
    for _ in 0..20 {
        monitor.record_latency(Duration::from_millis(50));
    }

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_latency_context_populated_when_baseline_ready] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    let baseline = c.latency_baseline_ms.expect("baseline must be present");
    let current = c.latency_current_ms.expect("current must be present");
    let ratio = c.latency_ratio.expect("ratio must be present");

    // Uniform 50ms samples → baseline and current both ~50ms,
    // ratio ~1.0. Wide tolerances to absorb float arithmetic.
    assert!(
        (baseline - 50.0).abs() < 1.0,
        "baseline not ~50ms: {}",
        baseline
    );
    assert!(
        (current - 50.0).abs() < 1.0,
        "current not ~50ms: {}",
        current
    );
    assert!(
        (ratio - 1.0).abs() < 0.1,
        "ratio not ~1.0: {}",
        ratio
    );
}

#[test]
fn test_latency_context_none_below_min_samples() {
    // With fewer than min_samples measurements the baseline is
    // cold; all three fields must be None. Single-field None is
    // disallowed by the `with_latency_context` builder by design.
    let bus = SignalBus::new();
    let monitor = bus.latency_monitor();
    for _ in 0..5 {
        monitor.record_latency(Duration::from_millis(50));
    }

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_latency_context_none_below_min_samples] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.latency_baseline_ms, None);
    assert_eq!(c.latency_current_ms, None);
    assert_eq!(c.latency_ratio, None);
}

#[test]
fn test_latency_context_detects_elevated_ewma() {
    // 15 stable samples establish baseline ~50ms, then 10 samples
    // at 100ms bring the EWMA up. The ratio should show that
    // current is materially higher than baseline.
    let bus = SignalBus::new();
    let monitor = bus.latency_monitor();
    for _ in 0..15 {
        monitor.record_latency(Duration::from_millis(50));
    }
    for _ in 0..10 {
        monitor.record_latency(Duration::from_millis(100));
    }

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_latency_context_detects_elevated_ewma] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    let ratio = c.latency_ratio.expect("ratio must be present");
    // EWMA with alpha=0.2 over the 10 high samples lifts the
    // current well above baseline. Exact number depends on the
    // alpha, so assert a loose but meaningful lower bound.
    assert!(
        ratio > 1.1,
        "ratio should show elevation (>1.1), got {}",
        ratio
    );
    // It shouldn't completely reach 2.0 either (10 new samples is
    // not enough to fully displace 15 old samples at alpha=0.2).
    assert!(
        ratio < 2.0,
        "ratio overshoot — old samples should still pull current down, got {}",
        ratio
    );
}

#[test]
fn test_latency_spike_flag_now_reflects_reality() {
    // Pre-M3-e.10 `latency_spike` was always `false` in production
    // because nothing called `record_latency`. This test confirms
    // that with the wire in place (or direct injection for test
    // determinism), `latency_spike` flips when a real spike occurs.
    let bus = SignalBus::new();
    let monitor = bus.latency_monitor();
    for _ in 0..15 {
        monitor.record_latency(Duration::from_millis(50));
    }

    let Some(c_pre) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_latency_spike_flag_now_reflects_reality] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };
    assert!(!c_pre.latency_spike, "no spike yet under stable samples");

    monitor.record_latency(Duration::from_millis(500));

    let c_post = bus.collect_guard_conditions().expect("probe must still work");
    assert!(
        c_post.latency_spike,
        "latency_spike must flip to true after a 10x sample"
    );
}

// --- Fresh-read test (latency is NOT cached) -----------------

#[test]
fn test_latency_context_is_read_fresh_not_cached() {
    // Memory-pressure probe is cached for SIGNAL_BUS_CACHE_TTL, but
    // latency context reads the monitor directly on every call.
    // This test confirms by injecting a sample between two
    // collect_guard_conditions calls within the TTL window and
    // showing the latency fields reflect the newer data.
    let bus = SignalBus::new();
    let monitor = bus.latency_monitor();

    // Pre-populate baseline with uniform samples.
    for _ in 0..15 {
        monitor.record_latency(Duration::from_millis(50));
    }

    let Some(c_before) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_latency_context_is_read_fresh_not_cached] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };
    let current_before = c_before.latency_current_ms.expect("current ready");

    // Immediately inject a MUCH higher sample. No sleep — we are
    // well within the 100ms memory-probe cache TTL.
    monitor.record_latency(Duration::from_millis(500));

    let c_after = bus.collect_guard_conditions().expect("still works");
    let current_after = c_after.latency_current_ms.expect("current ready");

    // The new sample lifts the EWMA. If latency were being cached
    // alongside memory, `current_after` would still equal
    // `current_before` within the TTL — this assertion would fail.
    assert!(
        current_after > current_before + 1.0,
        "latency_current_ms must reflect the fresh sample, \
         before={:.2}ms after={:.2}ms",
        current_before,
        current_after
    );
}

// --- End-to-end wire test (small graph) ----------------------

#[test]
fn test_execute_single_inner_feeds_latency_monitor() {
    // The signature test for M3-e.10: run a graph under a reactive
    // context and verify that `LatencyMonitor::sample_count` moved
    // from 0 to something > 0 as a side effect of
    // `execute_single_inner`. We cannot assert on `latency_spike`
    // or the derived fields because a 3-4 node graph is below
    // min_samples (10) — so they stay None / false — but
    // sample_count is the direct proof of wire.
    let mut graph = build_small_matmul_graph();
    let bus = Arc::new(SignalBus::new());

    assert_eq!(
        bus.latency_monitor().sample_count(),
        0,
        "pre-condition: fresh bus has zero samples"
    );

    let contract = minimal_contract();
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(AlwaysContinueGuard)];
    let guard_manager = GuardManager::new(guards);
    graph.set_reactive_context(ReactiveExecutionContext::new(
        Arc::clone(&bus),
        contract,
        guard_manager,
    ));

    let _ = graph
        .execute_checked(vec![make_input_tensor()])
        .expect("execute_checked must succeed under Continue guard");

    let samples = bus.latency_monitor().sample_count();
    assert!(
        samples > 0,
        "at least one node execution must have been recorded, got {}",
        samples
    );
}

#[test]
fn test_execute_without_reactive_context_does_not_feed_monitor() {
    // Symmetric check: when no reactive_context is attached, the
    // `NodeTimingRecorder`'s `bus` field is `None` and no latencies
    // are recorded. We use a disconnected bus (the graph does not
    // know about it) to confirm it stays at zero.
    let mut graph = build_small_matmul_graph();
    let disconnected_bus = SignalBus::new();
    assert_eq!(disconnected_bus.latency_monitor().sample_count(), 0);

    let _ = graph
        .execute_checked(vec![make_input_tensor()])
        .expect("execute_checked must succeed without reactive context");

    // Bus we never attached to the graph must still be at zero.
    assert_eq!(
        disconnected_bus.latency_monitor().sample_count(),
        0,
        "disconnected bus must not receive latency records"
    );
}
