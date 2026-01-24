#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::planner::execution_planner::ExecutionPlanner;
use v16::planner::plan_step::PlanStepKind;
use v16::planner::planner_errors::PlannerError;

fn make_runtime_state() -> RuntimeState {
    RuntimeState {
        memory_headroom: 0.8,
        is_stable: true,
        recent_recovery: false,
        offload_supported: true,
    }
}

fn make_constraints() -> Constraints {
    Constraints {
        items: vec![
            Constraint::hard(ConstraintKind::MemoryHeadroom { min: 0.2 }),
            Constraint::hard(ConstraintKind::RequireStability),
        ],
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

    let state = make_runtime_state();
    let constraints = make_constraints();

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
fn plan_is_produced_from_valid_contract() {
    let contract = make_contract();

    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");

    assert_eq!(plan.contract, contract);
    assert!(!plan.steps.is_empty());
    assert!(plan.globally_abortable);
}

#[test]
fn plan_respects_contract_constraints() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");

    // If the contract requires stability, there should be a memory headroom/check step and
    // backend selection that are conceptually consistent.
    assert!(contract.require_stability);

    let kinds: Vec<PlanStepKind> = plan.steps.iter().map(|s| s.kind.clone()).collect();

    assert!(kinds.contains(&PlanStepKind::EnsureMemoryHeadroom));
    assert!(kinds.contains(&PlanStepKind::SelectBackendCandidate));
    if contract.require_fallback {
        assert!(kinds.contains(&PlanStepKind::PrepareFallback));
    }
}

#[test]
fn planner_is_deterministic() {
    let contract = make_contract();

    let p1 = ExecutionPlanner::build_plan(&contract).expect("first plan");
    let p2 = ExecutionPlanner::build_plan(&contract).expect("second plan");

    assert_eq!(p1, p2);
}

#[test]
fn invalid_contract_yields_explicit_error() {
    let mut contract = make_contract();
    contract.allowed_backends.clear();

    let err = ExecutionPlanner::build_plan(&contract).unwrap_err();
    assert!(matches!(err, PlannerError::InvalidContract(_)));
}

#[test]
fn plan_steps_are_ordered_and_abortable() {
    let contract = make_contract();
    let plan = ExecutionPlanner::build_plan(&contract).expect("plan should be produced");

    // All steps must be conceptually abortable.
    assert!(!plan.steps.is_empty());
    for step in &plan.steps {
        assert!(step.abortable);
    }

    // Ensure the steps include the expected order of high-level phases.
    let kinds: Vec<PlanStepKind> = plan.steps.iter().map(|s| s.kind.clone()).collect();

    let idx_mem = kinds
        .iter()
        .position(|k| *k == PlanStepKind::EnsureMemoryHeadroom)
        .expect("EnsureMemoryHeadroom step missing");
    let idx_backend = kinds
        .iter()
        .position(|k| *k == PlanStepKind::SelectBackendCandidate)
        .expect("SelectBackendCandidate step missing");

    assert!(idx_mem < idx_backend);
}
