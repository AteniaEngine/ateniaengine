#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::feedback::event_emitter::EventEmitter;
use v16::guards::guard_action::GuardAction;
use v16::explain::explanation_builder::ExplanationBuilder;
use v16::explain::explanation_formatter::{format_explanation_json, format_explanation_text};
use v16::speculative::speculative_plan::SpeculativePlan;
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

#[test]
fn explanation_matches_execution_events() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan");

    let executed_steps: Vec<usize> = (0..plan.steps.len()).collect();
    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let guard_actions = vec![(0usize, GuardAction::Continue)];
    let expl = ExplanationBuilder::build(
        &contract.bias,
        &contract,
        "test plan".to_string(),
        events.clone(),
        outcome.clone(),
        guard_actions,
        None,
    )
    .expect("explanation");

    assert_eq!(expl.events, events);
    assert_eq!(expl.outcome, outcome);
}

#[test]
fn explanation_reflects_contract_constraints_and_guard_actions() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan");
    let executed_steps: Vec<usize> = vec![0];

    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Aborted,
        Some("aborted"),
    )
    .expect("events");

    let guard_actions = vec![(0usize, GuardAction::Abort)];

    let expl = ExplanationBuilder::build(
        &contract.bias,
        &contract,
        "test plan".to_string(),
        events,
        outcome,
        guard_actions,
        None,
    )
    .expect("explanation");

    // Outcome must reflect an aborted or partially completed execution, and
    // the guard action must record the abort decision.
    use v16::feedback::execution_outcome::ExecutionOutcomeKind;
    assert!(matches!(
        expl.outcome.kind,
        ExecutionOutcomeKind::Aborted | ExecutionOutcomeKind::PartiallyCompleted
    ));
    assert_eq!(expl.steps.len(), 1);
    assert_eq!(expl.steps[0].guard_action, Some(GuardAction::Abort));
}

#[test]
fn explanation_includes_speculation_when_present() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan");
    let executed_steps: Vec<usize> = vec![0, 1];

    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let guard_actions = vec![(0usize, GuardAction::Continue), (1usize, GuardAction::Degrade)];
    let speculative_plan = Some(SpeculativePlan::from_base(&plan));

    let expl = ExplanationBuilder::build(
        &contract.bias,
        &contract,
        "test plan".to_string(),
        events,
        outcome,
        guard_actions,
        speculative_plan,
    )
    .expect("explanation");

    assert!(expl.speculative_plan.is_some());
    assert!(expl.steps.iter().any(|s| s.speculative));
}

#[test]
fn explanation_is_deterministic() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan");
    let executed_steps: Vec<usize> = vec![0, 1];

    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let guard_actions = vec![(0usize, GuardAction::Continue), (1usize, GuardAction::Continue)];

    let e1 = ExplanationBuilder::build(
        &contract.bias,
        &contract,
        "test plan".to_string(),
        events.clone(),
        outcome.clone(),
        guard_actions.clone(),
        None,
    )
    .expect("e1");

    let e2 = ExplanationBuilder::build(
        &contract.bias,
        &contract,
        "test plan".to_string(),
        events,
        outcome,
        guard_actions,
        None,
    )
    .expect("e2");

    assert_eq!(e1, e2);
}

#[test]
fn formatting_output_is_stable() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan");
    let executed_steps: Vec<usize> = vec![0, 1];

    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &plan,
        &executed_steps,
        &v16::executor::executor_state::ExecutorStatus::Completed,
        None,
    )
    .expect("events");

    let guard_actions = vec![(0usize, GuardAction::Continue), (1usize, GuardAction::Continue)];

    let expl = ExplanationBuilder::build(
        &contract.bias,
        &contract,
        "test plan".to_string(),
        events,
        outcome,
        guard_actions,
        None,
    )
    .expect("explanation");

    let t1 = format_explanation_text(&expl);
    let t2 = format_explanation_text(&expl);
    let j1 = format_explanation_json(&expl);
    let j2 = format_explanation_json(&expl);

    assert_eq!(t1, t2);
    assert_eq!(j1, j2);
}
