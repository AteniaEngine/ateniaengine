//! APX v20 M3-e.2 — unit tests for `SimpleMemoryPressureGuard`.
//!
//! The guard emits `GuardAction::Degrade` when `memory_pressure`
//! strictly exceeds `DEGRADE_MEMORY_PRESSURE_THRESHOLD` (default 0.65);
//! otherwise it emits `Continue`. Follows the `#[path]` pattern used
//! by `adaptive_guards_test.rs` so v16 submodules are reachable from
//! the test crate without exporting them publicly.

#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::guards::execution_guard::ExecutionGuard;
use v16::guards::guard_action::GuardAction;
use v16::guards::guard_conditions::GuardConditions;
use v16::guards::simple_memory_pressure_guard::{
    DEGRADE_MEMORY_PRESSURE_THRESHOLD, SimpleMemoryPressureGuard,
};

fn make_permissive_contract() -> ExecutionContract {
    // A contract that does not require stability and has no hard
    // constraints: isolates the guard's decision from contract legality
    // checks that live on `GuardManager::evaluate`, not on the guard
    // itself.
    let bias = DecisionBias {
        risk_weight: 0.3,
        latency_weight: 0.4,
        stability_weight: 0.5,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.4,
    };

    let state = RuntimeState {
        memory_headroom: 0.8,
        is_stable: true,
        recent_recovery: false,
        offload_supported: true,
    };

    ExecutionContract {
        bias,
        runtime_snapshot: state,
        allowed_backends: vec![ExecutionBackend::Local, ExecutionBackend::Offload],
        forbidden_backends: vec![],
        max_aggressiveness: 0.7,
        require_fallback: true,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

fn conditions_with_pressure(p: f32) -> GuardConditions {
    // Only memory_pressure is relevant for this guard; the other
    // fields are zeroed / false.
    GuardConditions::new(p, 0, false, false)
}

#[test]
fn test_guard_below_threshold() {
    let guard = SimpleMemoryPressureGuard::new();
    let contract = make_permissive_contract();
    let conditions = conditions_with_pressure(0.5);

    let action = guard.evaluate(&contract, &conditions);
    assert!(
        matches!(action, GuardAction::Continue),
        "pressure 0.5 (< {}) must yield Continue, got {:?}",
        DEGRADE_MEMORY_PRESSURE_THRESHOLD,
        action
    );
}

#[test]
fn test_guard_above_threshold() {
    let guard = SimpleMemoryPressureGuard::new();
    let contract = make_permissive_contract();
    let conditions = conditions_with_pressure(0.9);

    let action = guard.evaluate(&contract, &conditions);
    assert!(
        matches!(action, GuardAction::Degrade),
        "pressure 0.9 (> {}) must yield Degrade, got {:?}",
        DEGRADE_MEMORY_PRESSURE_THRESHOLD,
        action
    );
}

#[test]
fn test_guard_at_threshold() {
    // The trigger is strictly `>`, so exactly-at-threshold is Continue.
    let guard = SimpleMemoryPressureGuard::new();
    let contract = make_permissive_contract();
    let conditions = conditions_with_pressure(DEGRADE_MEMORY_PRESSURE_THRESHOLD);

    let action = guard.evaluate(&contract, &conditions);
    assert!(
        matches!(action, GuardAction::Continue),
        "pressure == {} must yield Continue (trigger is strict >), got {:?}",
        DEGRADE_MEMORY_PRESSURE_THRESHOLD,
        action
    );
}

#[test]
fn test_guard_just_above_threshold() {
    // A small amount above the threshold already fires.
    let guard = SimpleMemoryPressureGuard::new();
    let contract = make_permissive_contract();
    let conditions = conditions_with_pressure(DEGRADE_MEMORY_PRESSURE_THRESHOLD + 0.001);

    let action = guard.evaluate(&contract, &conditions);
    assert!(
        matches!(action, GuardAction::Degrade),
        "pressure just above {} must yield Degrade, got {:?}",
        DEGRADE_MEMORY_PRESSURE_THRESHOLD,
        action
    );
}

#[test]
fn test_guard_custom_threshold() {
    // `with_threshold` should change the trigger point. Using 0.3:
    // - 0.25 → Continue
    // - 0.35 → Degrade
    let guard = SimpleMemoryPressureGuard::with_threshold(0.3);
    let contract = make_permissive_contract();

    assert_eq!(guard.threshold(), 0.3);

    let low = guard.evaluate(&contract, &conditions_with_pressure(0.25));
    assert!(matches!(low, GuardAction::Continue));

    let high = guard.evaluate(&contract, &conditions_with_pressure(0.35));
    assert!(matches!(high, GuardAction::Degrade));
}
