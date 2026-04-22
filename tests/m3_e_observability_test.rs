//! APX v20 M3-e.5 — observability tests for the Degrade reaction path.
//!
//! Covers the `degrade_events_count` counter exposed on
//! `ReactiveExecutionContext`: it starts at zero, increments exactly
//! when `GuardAction::Degrade` is processed by `check_guard_before_node`,
//! and stays put when the guard verdict is `Continue`.
//!
//! The counter is incremented before migration runs, so it counts
//! *attempts*, not successes. Testing the failure case (Degrade
//! processed, migration returns Err) would require forcing an
//! `ensure_cpu` failure cleanly — we do not have a clean way to do
//! that from the test crate without inventing a fake failure mode
//! that does not map to a real scenario, so that case is
//! intentionally not covered here.
//!
//! The positive and negative setups mirror `m3_e_degrade_test.rs`
//! (the M3-e.3 integration tests): a linear graph with a Parameter
//! pushed to VRAM, a `DegradeIfFailuresGuard` that fires on
//! `recent_failures > 0`, and a permissive `ExecutionContract`.

use std::sync::Arc;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::gpu::gpu_engine;
use atenia_engine::tensor::{DType, Device, Layout, Tensor, TensorStorage};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_action::GuardAction;
use atenia_engine::v16::guards::guard_conditions::GuardConditions;
use atenia_engine::v16::guards::guard_manager::GuardManager;

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
    let mut t = Tensor::with_layout(
        shape,
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.as_cpu_slice_mut().copy_from_slice(&data);
    t
}

struct DegradeIfFailuresGuard;

impl ExecutionGuard for DegradeIfFailuresGuard {
    fn name(&self) -> &'static str {
        "degrade_if_failures_guard_m3_e_5_fixture"
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> GuardAction {
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

    (gb.build(), x_id, w_id)
}

fn make_x_input() -> Tensor {
    tensor_from(vec![1, 2], vec![1.0, 2.0])
}

fn make_reactive_context(record_failure: bool) -> ReactiveExecutionContext {
    let bus = Arc::new(SignalBus::new());
    if record_failure {
        bus.failure_counter().record_failure();
    }
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(DegradeIfFailuresGuard)];
    let guard_manager = GuardManager::new(guards);
    ReactiveExecutionContext::new(bus, permissive_contract(), guard_manager)
}

#[test]
fn test_degrade_events_count_starts_at_zero() {
    // A freshly constructed context must report zero events. This is
    // independent of GPU availability — it tests only the counter's
    // initial state.
    let ctx = make_reactive_context(false);
    assert_eq!(
        ctx.degrade_events_count(),
        0,
        "a fresh ReactiveExecutionContext must start with zero degrade events"
    );
}

#[test]
fn test_degrade_events_count_increments_on_trigger() {
    if !require_gpu("test_degrade_events_count_increments_on_trigger") {
        return;
    }

    // Mirror the positive M3-e.3 setup: parameter on VRAM, guard that
    // fires on recent_failures > 0, and a failure recorded on the bus.
    // After `execute_checked` the counter must have incremented at
    // least once (the guard runs before every node, so on a 4-node
    // graph it can run more than once; we assert >= 1 which is the
    // contract).
    let (mut graph, _x_id, w_id) = build_linear_param_graph();
    let _baseline = graph.execute(vec![make_x_input()]);

    {
        let w_tensor = graph.nodes[w_id]
            .output
            .as_mut()
            .expect("parameter output present");
        w_tensor
            .ensure_gpu()
            .expect("ensure_gpu on parameter must succeed");
        assert!(
            matches!(w_tensor.storage, TensorStorage::Cuda(_)),
            "setup: parameter should be Cuda after ensure_gpu"
        );
    }

    let ctx = make_reactive_context(true);
    assert_eq!(
        ctx.degrade_events_count(),
        0,
        "pre-condition: counter must be zero before execute_checked"
    );
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_x_input()]);
    assert!(
        result.is_ok(),
        "execute_checked under Degrade must succeed, got {:?}",
        result.err()
    );

    let count = graph
        .reactive_context()
        .expect("context still attached after execute_checked")
        .degrade_events_count();
    assert!(
        count >= 1,
        "degrade_events_count must be >= 1 after a Degrade-driving execute, got {}",
        count
    );
}

#[test]
fn test_degrade_events_count_does_not_increment_without_trigger() {
    if !require_gpu("test_degrade_events_count_does_not_increment_without_trigger") {
        return;
    }

    // Mirror the negative M3-e.3 setup: parameter on VRAM, reactive
    // context attached, but no failure recorded. Guard emits Continue
    // on every call; counter must stay at zero across the whole
    // execution.
    let (mut graph, _x_id, w_id) = build_linear_param_graph();
    let _baseline = graph.execute(vec![make_x_input()]);

    {
        let w_tensor = graph.nodes[w_id]
            .output
            .as_mut()
            .expect("parameter output present");
        w_tensor
            .ensure_gpu()
            .expect("ensure_gpu on parameter must succeed");
    }

    let ctx = make_reactive_context(false);
    assert_eq!(ctx.degrade_events_count(), 0);
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_x_input()]);
    assert!(
        result.is_ok(),
        "execute_checked with Continue guard must succeed, got {:?}",
        result.err()
    );

    let count = graph
        .reactive_context()
        .expect("context still attached")
        .degrade_events_count();
    assert_eq!(
        count, 0,
        "degrade_events_count must stay at 0 when guard never emits Degrade, got {}",
        count
    );
}

// NOTE: a fourth test covering the migration-failure path
// (Degrade processed, `ensure_cpu` returns Err, counter still
// incremented) is intentionally omitted. We would need to force an
// `ensure_cpu` failure cleanly from the test crate, and there is no
// public API to inject D→H transfer errors. Inventing an artificial
// failure mode that does not map to a real runtime scenario would
// test the mock, not the production path. If a realistic hook appears
// later (e.g. a test-only `FaultyGpuEngine`), this case should be
// added then.
