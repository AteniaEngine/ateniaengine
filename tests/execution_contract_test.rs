#![allow(dead_code)]

#[path = "../src/v15/mod.rs"]
mod v15;

#[path = "../src/v16/mod.rs"]
mod v16;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{ConstraintKind, RuntimeState};
use v16::contract::contract_errors::ContractError;
use v16::contract::contract_resolver::ContractResolver;

fn make_normalized_bias() -> DecisionBias {
    DecisionBias {
        risk_weight: 0.3,
        latency_weight: 0.4,
        stability_weight: 0.8,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.6,
    }
}

fn make_runtime_state() -> RuntimeState {
    RuntimeState {
        memory_headroom: 0.8,
        is_stable: true,
        recent_recovery: false,
        offload_supported: true,
    }
}

#[test]
fn contract_is_produced_from_valid_bias() {
    let bias = make_normalized_bias();
    let state = make_runtime_state();

    let contract =
        ContractResolver::resolve_contract(&bias, &state).expect("contract should be produced");

    assert_eq!(contract.bias, bias);
    assert_eq!(contract.runtime_snapshot, state);
    assert!(!contract.allowed_backends.is_empty());
}

#[test]
fn invalid_bias_or_state_yields_explicit_error() {
    // Bias with invalid range.
    let mut bad_bias = make_normalized_bias();
    bad_bias.risk_weight = 1.5;
    let state = make_runtime_state();

    let err = ContractResolver::resolve_contract(&bad_bias, &state).unwrap_err();
    assert!(matches!(err, ContractError::InvariantViolation(_)));

    // State with invalid headroom.
    let bias = make_normalized_bias();
    let bad_state = RuntimeState {
        memory_headroom: 1.5,
        ..make_runtime_state()
    };

    let err2 = ContractResolver::resolve_contract(&bias, &bad_state).unwrap_err();
    assert!(matches!(err2, ContractError::InvariantViolation(_)));
}

#[test]
fn contract_is_deterministic() {
    let bias = make_normalized_bias();
    let state = make_runtime_state();

    let c1 = ContractResolver::resolve_contract(&bias, &state).expect("first contract");
    let c2 = ContractResolver::resolve_contract(&bias, &state).expect("second contract");

    assert_eq!(c1, c2);
}

#[test]
fn no_side_effects_during_resolution() {
    let bias = make_normalized_bias();
    let state = make_runtime_state();

    let bias_before = bias.clone();
    let state_before = state.clone();

    let _ = ContractResolver::resolve_contract(&bias, &state);

    assert_eq!(bias, bias_before);
    assert_eq!(state, state_before);
}

#[test]
fn constraints_reflect_bias_priorities() {
    // Strong stability preference with unstable state should require stability.
    let mut bias = make_normalized_bias();
    bias.stability_weight = 0.9;
    let unstable_state = RuntimeState {
        is_stable: false,
        ..make_runtime_state()
    };

    let contract = ContractResolver::resolve_contract(&bias, &unstable_state).expect("contract");

    assert!(contract.require_stability);
    let has_require_stability = contract
        .constraints
        .items
        .iter()
        .any(|c| matches!(c.kind, ConstraintKind::RequireStability));
    assert!(has_require_stability);

    // Very aggressive bias with low headroom should be capped.
    let aggressive_bias = DecisionBias {
        risk_weight: 0.9,
        latency_weight: 0.9,
        stability_weight: 0.1,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.2,
    };

    let low_headroom_state = RuntimeState {
        memory_headroom: 0.1,
        ..make_runtime_state()
    };

    let contract2 = ContractResolver::resolve_contract(&aggressive_bias, &low_headroom_state)
        .expect("contract");

    assert!(contract2.max_aggressiveness <= 0.5);
    let has_limit_aggressiveness = contract2
        .constraints
        .items
        .iter()
        .any(|c| matches!(c.kind, ConstraintKind::LimitAggressiveness { .. }));
    assert!(has_limit_aggressiveness);
}
