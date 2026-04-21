//! Integration tests for APX v20 M2: the AMG executor consults
//! `SignalBus` → `GuardManager` before each node and aborts cleanly
//! when a guard triggers.
//!
//! Two tests cover distinct controllable signals:
//! - `abort_when_failure_counter_triggers_guard`: records failures on
//!   `SignalBus::failure_counter()` and uses a guard that fires on
//!   `recent_failures > 0`.
//! - `abort_when_latency_monitor_triggers_guard`: records enough
//!   latencies to produce a spike on `SignalBus::latency_monitor()`
//!   and uses a guard that fires on `latency_spike == true`.
//!
//! We never use `Graph::execute` (which panics on abort) — we use
//! `Graph::execute_checked` to obtain `Result` and assert on the
//! `Err` variant.
//!
//! Signals not exercised here:
//! - `memory_pressure` / `pre_oom_signal`: would require mocking the
//!   NVIDIA + system probes, which cannot be done without modifying
//!   v19. The two signals above provide full coverage of the
//!   abort path without mocks.

use std::sync::Arc;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::{ExecutionAbortReason, ReactiveExecutionContext};
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

// ---------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------

/// Small graph: 2x2 input × 2x2 identity-weight MatMul → output.
/// Structure is intentionally trivial; tests verify the abort path,
/// not numerical behavior.
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
    weight.data = vec![1.0, 0.0, 0.0, 1.0]; // identity
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
    t.data = vec![1.0, 2.0, 3.0, 4.0];
    t
}

/// Minimal contract with `require_stability = false` so we don't
/// accidentally trigger `IllegalAction` from the `GuardManager`'s
/// own legality check (which needs pre_oom_signal AND require_stability).
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

// ---------------------------------------------------------------------
// Test guards
// ---------------------------------------------------------------------

/// Aborts when `recent_failures > 0`.
struct FailureCountAbortGuard;

impl ExecutionGuard for FailureCountAbortGuard {
    fn name(&self) -> &'static str {
        "failure_count_abort_guard"
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> GuardAction {
        if conditions.recent_failures > 0 {
            GuardAction::Abort
        } else {
            GuardAction::Continue
        }
    }
}

/// Aborts when `latency_spike == true`.
struct LatencySpikeAbortGuard;

impl ExecutionGuard for LatencySpikeAbortGuard {
    fn name(&self) -> &'static str {
        "latency_spike_abort_guard"
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> GuardAction {
        if conditions.latency_spike {
            GuardAction::Abort
        } else {
            GuardAction::Continue
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[test]
fn abort_when_failure_counter_triggers_guard() {
    let mut graph = build_small_matmul_graph();

    let bus = Arc::new(SignalBus::new());
    // Record 3 failures — the guard will fire since threshold is > 0.
    for _ in 0..3 {
        bus.failure_counter().record_failure();
    }

    let contract = minimal_contract();
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(FailureCountAbortGuard)];
    let guard_manager = GuardManager::new(guards);

    let ctx = ReactiveExecutionContext::new(bus, contract, guard_manager);
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_input_tensor()]);

    // Assert the abort variant and extract for field inspection.
    match result {
        Err(ExecutionAbortReason::GuardAborted { at_node, conditions }) => {
            // Recent failures must reflect what we recorded.
            assert_eq!(
                conditions.recent_failures, 3,
                "expected 3 recorded failures, got {}",
                conditions.recent_failures
            );
            // at_node should be a valid node id in the graph.
            assert!(
                at_node < graph.nodes.len(),
                "at_node {} out of bounds (graph has {} nodes)",
                at_node,
                graph.nodes.len()
            );
        }
        Err(other) => panic!("expected GuardAborted, got {:?}", other),
        Ok(_) => panic!("expected abort, got Ok(_)"),
    }

    // last_abort must match what execute_checked returned.
    let last = graph.last_abort().expect("last_abort must be populated after abort");
    match last {
        ExecutionAbortReason::GuardAborted { conditions, .. } => {
            assert_eq!(conditions.recent_failures, 3);
        }
        other => panic!("last_abort should be GuardAborted, got {:?}", other),
    }
}

#[test]
fn abort_when_latency_monitor_triggers_guard() {
    let mut graph = build_small_matmul_graph();

    let bus = Arc::new(SignalBus::new());

    // Record 15 normal latencies to establish a baseline, then 1 spike.
    // 15 ≥ min_samples (default 10) and 500ms ≫ 50ms × 2.0 multiplier.
    let monitor = bus.latency_monitor();
    for _ in 0..15 {
        monitor.record_latency(std::time::Duration::from_millis(50));
    }
    monitor.record_latency(std::time::Duration::from_millis(500));
    assert!(
        monitor.has_recent_spike(),
        "test setup: spike should have been detected"
    );

    let contract = minimal_contract();
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(LatencySpikeAbortGuard)];
    let guard_manager = GuardManager::new(guards);

    let ctx = ReactiveExecutionContext::new(bus, contract, guard_manager);
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_input_tensor()]);

    match result {
        Err(ExecutionAbortReason::GuardAborted { at_node, conditions }) => {
            assert!(
                conditions.latency_spike,
                "expected latency_spike=true in conditions, got false"
            );
            assert!(at_node < graph.nodes.len());
        }
        Err(other) => panic!("expected GuardAborted, got {:?}", other),
        Ok(_) => panic!("expected abort, got Ok(_)"),
    }

    assert!(
        graph.last_abort().is_some(),
        "last_abort must be populated after abort"
    );
}
