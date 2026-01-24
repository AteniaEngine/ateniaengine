#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::executor::executor_state::ExecutorStatus;
use v16::feedback::event_emitter::EventEmitter;
use v16::feedback::feedback_collector::FeedbackCollector;
use v16::feedback::feedback_errors::FeedbackError;
use v16::feedback::execution_event::ExecutionEventKind;
use v16::feedback::execution_outcome::ExecutionOutcomeKind;
use v16::planner::execution_planner::ExecutionPlanner;

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

fn make_plan_len() -> usize {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");
    plan.steps.len()
}

#[test]
fn events_are_emitted_in_execution_order() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");
    let executed_steps: Vec<usize> = (0..plan.steps.len()).collect();

    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &ExecutorStatus::Completed,
        None,
    )
    .expect("emission should succeed");

    // Logical timestamps must be strictly increasing.
    for w in events.windows(2) {
        assert!(w[1].logical_timestamp > w[0].logical_timestamp);
    }

    // First event is ExecutionStarted, last is ExecutionCompleted.
    assert!(matches!(events.first().unwrap().kind, ExecutionEventKind::ExecutionStarted));
    assert!(matches!(events.last().unwrap().kind, ExecutionEventKind::ExecutionCompleted));

    assert!(matches!(outcome.kind, ExecutionOutcomeKind::Completed));
}

#[test]
fn outcome_matches_execution_result() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");

    // Completed
    let executed_all: Vec<usize> = (0..plan.steps.len()).collect();
    let (_, out_completed) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_all,
        &ExecutorStatus::Completed,
        None,
    )
    .expect("completed");
    assert!(matches!(out_completed.kind, ExecutionOutcomeKind::Completed));

    // Failed after partial execution
    let executed_partial: Vec<usize> = vec![0];
    let (_, out_failed) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_partial,
        &ExecutorStatus::Failed,
        Some("error"),
    )
    .expect("failed");
    assert!(matches!(out_failed.kind, ExecutionOutcomeKind::PartiallyCompleted));

    // Aborted with no steps
    let (_, out_aborted) = EventEmitter::emit_for_snapshot(
        &plan,
        &[],
        &ExecutorStatus::Aborted,
        Some("aborted"),
    )
    .expect("aborted");
    assert!(matches!(out_aborted.kind, ExecutionOutcomeKind::Aborted));
}

#[test]
fn feedback_is_deterministic_and_pure() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");
    let executed_steps: Vec<usize> = vec![0, 1];

    let plan_before = plan.clone();
    let executed_before = executed_steps.clone();

    let r1 = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &ExecutorStatus::Completed,
        None,
    )
    .expect("first emission");
    let r2 = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &ExecutorStatus::Completed,
        None,
    )
    .expect("second emission");

    assert_eq!(r1, r2);

    // Inputs are unchanged.
    assert_eq!(plan, plan_before);
    assert_eq!(executed_steps, executed_before);
}

#[test]
fn feedback_collector_records_snapshot() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");
    let executed_steps: Vec<usize> = vec![0, 1];

    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &ExecutorStatus::Completed,
        None,
    )
    .expect("emission");

    let mut collector = FeedbackCollector::new();
    collector.record(events.clone(), outcome.clone());

    let snapshot = collector.snapshot().expect("snapshot");
    assert_eq!(snapshot.events, events);
    assert_eq!(snapshot.outcome, outcome);
}

#[test]
fn invalid_event_sequences_yield_error() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");

    // Out-of-bounds index.
    let bad_steps = vec![plan.steps.len()];
    let err = EventEmitter::emit_for_snapshot(
        &plan,
        &bad_steps,
        &ExecutorStatus::Completed,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, FeedbackError::InvalidEvent(_)));

    // Non-monotonic indices.
    let bad_order = vec![1, 0];
    let err2 = EventEmitter::emit_for_snapshot(
        &plan,
        &bad_order,
        &ExecutorStatus::Completed,
        None,
    )
    .unwrap_err();
    assert!(matches!(err2, FeedbackError::LogicalOrderViolation(_)));
}
