#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::executor::execution_context::{ExecutionContext, RuntimeFacade};
use v16::feedback::event_emitter::EventEmitter;
use v16::planner::execution_planner::ExecutionPlanner;
use v16::replay::execution_replay::ExecutionReplay;
use v16::replay::replay_context::ReplayContext;
use v16::replay::replay_errors::ReplayError;

#[derive(Debug, Clone, PartialEq, Default)]
struct MockRuntime {
    calls: Vec<&'static str>,
}

impl RuntimeFacade for MockRuntime {
    fn ensure_memory_headroom(&mut self) -> Result<(), String> {
        self.calls.push("ensure_memory_headroom");
        Ok(())
    }

    fn select_backend_candidate(&mut self) -> Result<(), String> {
        self.calls.push("select_backend_candidate");
        Ok(())
    }

    fn prepare_fallback(&mut self) -> Result<(), String> {
        self.calls.push("prepare_fallback");
        Ok(())
    }

    fn mark_tensors_movable(&mut self) -> Result<(), String> {
        self.calls.push("mark_tensors_movable");
        Ok(())
    }
}

fn make_contract() -> ExecutionContract {
    let bias = DecisionBias {
        risk_weight: 0.3,
        latency_weight: 0.4,
        stability_weight: 0.8,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.4,
    };

    let state = RuntimeState {
        memory_headroom: 0.8,
        is_stable: true,
        recent_recovery: false,
        offload_supported: true,
    };

    let constraints = Constraints {
        items: vec![
            Constraint::hard(ConstraintKind::MemoryHeadroom { min: 0.2 }),
            Constraint::hard(ConstraintKind::RequireStability),
        ],
    };

    ExecutionContract {
        bias,
        runtime_snapshot: state,
        allowed_backends: vec![ExecutionBackend::Local, ExecutionBackend::Offload],
        forbidden_backends: vec![],
        max_aggressiveness: 0.7,
        require_fallback: true,
        require_stability: true,
        constraints,
    }
}

fn make_plan() -> v16::planner::execution_plan::ExecutionPlan {
    let contract = make_contract();
    ExecutionPlanner::build_plan(&contract).expect("plan should be produced")
}

#[test]
fn replay_reproduces_original_execution_order_and_outcome() {
    let contract = make_contract();
    let plan = make_plan();

    let executed_steps: Vec<usize> = (0..plan.steps.len()).collect();
    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let replay_ctx = ReplayContext::new("replay".to_string(), ctx);

    let original_events = events.clone();
    let original_outcome = outcome.clone();

    let mut replay = ExecutionReplay::new(contract, plan, events, outcome, replay_ctx);

    replay.replay().expect("replay should succeed");

    // Replay must not have mutated the original events or outcome copies.
    assert_eq!(original_events, replay.events);
    assert_eq!(original_outcome, replay.outcome);
}

#[test]
fn replay_is_deterministic() {
    let contract = make_contract();
    let plan = make_plan();

    let executed_steps: Vec<usize> = (0..plan.steps.len()).collect();
    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let runtime1 = MockRuntime::default();
    let runtime2 = MockRuntime::default();

    let ctx1 = ExecutionContext::new(runtime1);
    let ctx2 = ExecutionContext::new(runtime2);

    let replay_ctx1 = ReplayContext::new("replay1".to_string(), ctx1);
    let replay_ctx2 = ReplayContext::new("replay2".to_string(), ctx2);

    let mut r1 = ExecutionReplay::new(
        contract.clone(),
        plan.clone(),
        events.clone(),
        outcome.clone(),
        replay_ctx1,
    );
    let mut r2 = ExecutionReplay::new(contract, plan, events, outcome, replay_ctx2);

    r1.replay().expect("first replay");
    r2.replay().expect("second replay");

    assert_eq!(
        r1.context.context.runtime.calls,
        r2.context.context.runtime.calls
    );
}

#[test]
fn replay_aborts_on_inconsistent_history() {
    let contract = make_contract();
    let plan = make_plan();

    let executed_steps: Vec<usize> = (0..plan.steps.len()).collect();
    let (mut events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    // Introduce an inconsistency: swap timestamps of two events.
    if events.len() >= 2 {
        events[0].logical_timestamp = 10;
        events[1].logical_timestamp = 0;
    }

    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let replay_ctx = ReplayContext::new("replay".to_string(), ctx);

    let mut replay = ExecutionReplay::new(contract, plan, events, outcome, replay_ctx);

    let err = replay.replay().unwrap_err();
    assert!(matches!(err, ReplayError::InconsistentHistory(_)));
}

#[test]
fn replay_has_no_side_effects_on_inputs() {
    let contract = make_contract();
    let plan = make_plan();

    let executed_steps: Vec<usize> = (0..plan.steps.len()).collect();
    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let contract_before = contract.clone();
    let plan_before = plan.clone();
    let events_before = events.clone();
    let outcome_before = outcome.clone();

    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let replay_ctx = ReplayContext::new("replay".to_string(), ctx);

    let mut replay = ExecutionReplay::new(contract, plan, events, outcome, replay_ctx);

    let _ = replay.replay();

    assert_eq!(replay.contract, contract_before);
    assert_eq!(replay.plan, plan_before);
    assert_eq!(replay.events, events_before);
    assert_eq!(replay.outcome, outcome_before);
}
