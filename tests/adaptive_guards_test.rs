#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::guards::execution_guard::ExecutionGuard;
use v16::guards::guard_action::GuardAction;
use v16::guards::guard_conditions::GuardConditions;
use v16::guards::guard_errors::GuardError;
use v16::guards::guard_manager::GuardManager;

fn make_contract(require_stability: bool) -> ExecutionContract {
    let bias = DecisionBias {
        risk_weight: 0.3,
        latency_weight: 0.4,
        stability_weight: 0.8,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.4,
    };

    let state = RuntimeState {
        memory_headroom: 0.8,
        is_stable: !require_stability,
        recent_recovery: false,
        offload_supported: true,
    };

    let mut items = vec![Constraint::hard(ConstraintKind::MemoryHeadroom { min: 0.2 })];
    if require_stability {
        items.push(Constraint::hard(ConstraintKind::RequireStability));
    }

    let constraints = Constraints { items };

    ExecutionContract {
        bias,
        runtime_snapshot: state,
        allowed_backends: vec![ExecutionBackend::Local, ExecutionBackend::Offload],
        forbidden_backends: vec![],
        max_aggressiveness: 0.7,
        require_fallback: true,
        require_stability,
        constraints,
    }
}

fn make_conditions_high_risk() -> GuardConditions {
    GuardConditions::new(0.9, 3, true, true)
}

fn make_conditions_low_risk() -> GuardConditions {
    GuardConditions::new(0.1, 0, false, false)
}

struct MemoryPressureGuard {
    threshold: f32,
}

impl ExecutionGuard for MemoryPressureGuard {
    fn name(&self) -> &'static str {
        "memory_pressure_guard"
    }

    fn evaluate(&self, _contract: &ExecutionContract, conditions: &GuardConditions) -> GuardAction {
        if conditions.memory_pressure > self.threshold {
            GuardAction::Abort
        } else {
            GuardAction::Continue
        }
    }
}

struct FailureCountGuard {
    max_failures: u32,
}

impl ExecutionGuard for FailureCountGuard {
    fn name(&self) -> &'static str {
        "failure_count_guard"
    }

    fn evaluate(&self, _contract: &ExecutionContract, conditions: &GuardConditions) -> GuardAction {
        if conditions.recent_failures > self.max_failures {
            GuardAction::Degrade
        } else {
            GuardAction::Continue
        }
    }
}

struct AlwaysContinueGuard;

impl ExecutionGuard for AlwaysContinueGuard {
    fn name(&self) -> &'static str {
        "always_continue_guard"
    }

    fn evaluate(&self, _contract: &ExecutionContract, _conditions: &GuardConditions) -> GuardAction {
        GuardAction::Continue
    }
}

#[test]
fn guards_detect_risk_and_recommend_action() {
    let contract = make_contract(true);
    let conditions = make_conditions_high_risk();

    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(MemoryPressureGuard { threshold: 0.7 })];
    let manager = GuardManager::new(guards);

    let action = manager
        .evaluate(&contract, &conditions)
        .expect("evaluation should succeed");

    assert_eq!(action, GuardAction::Abort);
}

#[test]
fn abort_dominates_over_degrade() {
    let contract = make_contract(true);
    let conditions = make_conditions_high_risk();

    let guards: Vec<Box<dyn ExecutionGuard>> = vec![
        Box::new(MemoryPressureGuard { threshold: 0.7 }),
        Box::new(FailureCountGuard { max_failures: 1 }),
    ];
    let manager = GuardManager::new(guards);

    let action = manager
        .evaluate(&contract, &conditions)
        .expect("evaluation should succeed");

    assert_eq!(action, GuardAction::Abort);
}

#[test]
fn guards_never_violate_execution_contract() {
    let contract = make_contract(true);
    let conditions = make_conditions_high_risk();

    // This guard would like to continue even under high risk.
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(AlwaysContinueGuard)];
    let manager = GuardManager::new(guards);

    let err = manager.evaluate(&contract, &conditions).unwrap_err();
    assert!(matches!(err, GuardError::IllegalAction(_)));
}

#[test]
fn guard_evaluation_is_deterministic_and_pure() {
    let contract = make_contract(false);
    let conditions = make_conditions_high_risk();

    let guards: Vec<Box<dyn ExecutionGuard>> = vec![
        Box::new(MemoryPressureGuard { threshold: 0.7 }),
        Box::new(FailureCountGuard { max_failures: 1 }),
    ];
    let manager = GuardManager::new(guards);

    let contract_before = contract.clone();
    let conditions_before = conditions.clone();

    let a1 = manager.evaluate(&contract, &conditions).expect("first");
    let a2 = manager.evaluate(&contract, &conditions).expect("second");

    assert_eq!(a1, a2);
    assert_eq!(contract, contract_before);
    assert_eq!(conditions, conditions_before);
}

#[test]
fn guard_evaluation_handles_low_risk_as_continue() {
    let contract = make_contract(true);
    let conditions = make_conditions_low_risk();

    let guards: Vec<Box<dyn ExecutionGuard>> = vec![
        Box::new(MemoryPressureGuard { threshold: 0.7 }),
        Box::new(FailureCountGuard { max_failures: 1 }),
    ];
    let manager = GuardManager::new(guards);

    let action = manager
        .evaluate(&contract, &conditions)
        .expect("evaluation should succeed");

    assert_eq!(action, GuardAction::Continue);
}
