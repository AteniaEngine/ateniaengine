//! Integration tests for APX v20 M2: the AMG executor completes
//! normally when guards are attached but do not trigger.
//!
//! Two tests cover the non-abort paths:
//! - `no_abort_with_continue_guard_matches_baseline`: reactive
//!   context is attached, conditions are clean, and a guard that
//!   always returns `Continue` is installed. The output must be
//!   numerically identical (within 1e-6) to the same graph executed
//!   WITHOUT any reactive context.
//! - `no_abort_without_reactive_context`: baseline path — no
//!   reactive context set. Confirms `execute_checked` returns
//!   `Ok(_)` and `last_abort()` is `None`. Ensures the checked API
//!   is zero-impact when the reactive layer is absent.
//!
//! Skipped on this host (documented):
//! - The fail-open "probe unavailable" branch, which requires
//!   `SignalBus::collect_guard_conditions` to return `None`. On an
//!   NVIDIA host with working `nvidia-smi` + `sysinfo`, probes never
//!   fail, so this branch cannot be exercised without mocking v19.
//!   The branch is covered by code inspection in
//!   `Graph::check_guard_before_node` (early-return on `None`).

use std::sync::Arc;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
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

const NUMERICAL_TOLERANCE: f32 = 1e-6;

// ---------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------

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
    weight.set_cpu_data(vec![1.0, 0.0, 0.0, 1.0]); // identity
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
// Test guard
// ---------------------------------------------------------------------

struct AlwaysContinueGuard;

impl ExecutionGuard for AlwaysContinueGuard {
    fn name(&self) -> &'static str {
        "always_continue_guard"
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        _conditions: &GuardConditions,
    ) -> GuardAction {
        GuardAction::Continue
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", ctx);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff < NUMERICAL_TOLERANCE,
            "{}: idx {}: got={} want={} diff={}",
            ctx, i, g, w, diff
        );
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[test]
fn no_abort_with_continue_guard_matches_baseline() {
    // --- Baseline: same graph, no reactive context ---
    let mut baseline = build_small_matmul_graph();
    let baseline_outputs = baseline
        .execute_checked(vec![make_input_tensor()])
        .expect("baseline execution must not abort");
    assert!(
        baseline.last_abort().is_none(),
        "baseline must not record an abort"
    );
    assert_eq!(baseline_outputs.len(), 1, "baseline should produce one output");

    // --- Guarded: same graph + reactive context with AlwaysContinueGuard ---
    let mut guarded = build_small_matmul_graph();
    let bus = Arc::new(SignalBus::new());
    // No failures recorded, no latency spikes — conditions are clean.
    let contract = minimal_contract();
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(AlwaysContinueGuard)];
    let guard_manager = GuardManager::new(guards);
    guarded.set_reactive_context(ReactiveExecutionContext::new(
        bus,
        contract,
        guard_manager,
    ));

    let guarded_outputs = guarded
        .execute_checked(vec![make_input_tensor()])
        .expect("guarded execution with continue guard must not abort");
    assert!(
        guarded.last_abort().is_none(),
        "guarded execution with continue guard must not record an abort"
    );
    assert_eq!(guarded_outputs.len(), 1);

    // --- Outputs must be numerically identical ---
    assert_close(
        guarded_outputs[0].as_cpu_slice(),
        baseline_outputs[0].as_cpu_slice(),
        "guarded vs baseline output",
    );
}

#[test]
fn no_abort_without_reactive_context() {
    let mut graph = build_small_matmul_graph();

    // No reactive_context set. execute_checked takes the fast path
    // (check_guard_before_node returns Ok immediately when
    // reactive_context is None) and must succeed.
    let result = graph.execute_checked(vec![make_input_tensor()]);

    assert!(
        result.is_ok(),
        "execute_checked without reactive_context must return Ok"
    );
    assert!(
        graph.last_abort().is_none(),
        "last_abort must remain None"
    );

    let outputs = result.unwrap();
    assert_eq!(outputs.len(), 1);
    // Identity matmul: output = input.
    assert_close(
        outputs[0].as_cpu_slice(),
        &[1.0, 2.0, 3.0, 4.0],
        "identity matmul output",
    );
}
