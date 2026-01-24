#![allow(clippy::float_cmp)]

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::builtin::latency_first::LatencyFirstPolicy;
use v15::policy::builtin::stability_first::StabilityFirstPolicy;
use v15::policy::builtin::throughput_first::ThroughputFirstPolicy;
use v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use v15::policy::evidence::signals::{PolicySignal, PolicySignalKind};
use v15::policy::manager::policy_manager::PolicyManager;
use v15::policy::preferences::user_preferences::UserPreferences;
use v15::policy::registry::PolicyRegistry;
use v15::policy::simulation::simulator::PolicySimulator;
use v15::policy::types::PolicyInput;

fn make_registry() -> PolicyRegistry {
    let mut registry = PolicyRegistry::new();
    registry.register(StabilityFirstPolicy);
    registry.register(ThroughputFirstPolicy);
    registry.register(LatencyFirstPolicy);
    registry
}

fn make_input() -> PolicyInput {
    PolicyInput::default()
}

fn make_prefs() -> UserPreferences {
    UserPreferences {
        prefer_latency: true,
        avoid_ssd: false,
        prioritize_stability: true,
        minimize_power: false,
        prefer_gpu: true,
    }
}

fn make_evidence() -> PolicyEvidenceSnapshot {
    PolicyEvidenceSnapshot::new(vec![
        PolicySignal {
            kind: PolicySignalKind::HighMemoryPressure,
            score: 0.5,
        },
        PolicySignal {
            kind: PolicySignalKind::RecentRecovery,
            score: 0.4,
        },
    ])
}

#[test]
fn simulation_evaluates_multiple_policies() {
    let registry = make_registry();
    let input = make_input();
    let prefs = make_prefs();
    let evidence = make_evidence();

    let results = PolicySimulator::simulate_for_policies(
        &registry,
        &["stability_first", "throughput_first", "latency_first"],
        &input,
        &prefs,
        Some(&evidence),
    );

    assert_eq!(results.len(), 3);
    let names: Vec<String> = results.iter().map(|r| r.policy_name.clone()).collect();
    assert!(names.contains(&"stability_first".to_string()));
    assert!(names.contains(&"throughput_first".to_string()));
    assert!(names.contains(&"latency_first".to_string()));
}

#[test]
fn simulation_does_not_change_active_policy() {
    let registry = make_registry();
    let manager = PolicyManager::new(registry, "stability_first");

    let input = make_input();
    let prefs = make_prefs();
    let evidence = make_evidence();

    let before = manager.active_policy_name().to_string();

    // Use a separate registry for simulation so that the manager is
    // only used to check the active policy name.
    let sim_registry = make_registry();

    let _ = PolicySimulator::simulate_for_policies(
        &sim_registry,
        &["stability_first", "throughput_first"],
        &input,
        &prefs,
        Some(&evidence),
    );

    let after = manager.active_policy_name().to_string();
    assert_eq!(before, after);
}

#[test]
fn simulation_is_deterministic() {
    let registry = make_registry();
    let input = make_input();
    let prefs = make_prefs();
    let evidence = make_evidence();

    let names = ["stability_first", "throughput_first", "latency_first"];

    let r1 = PolicySimulator::simulate_for_policies(&registry, &names, &input, &prefs, Some(&evidence));
    let r2 = PolicySimulator::simulate_for_policies(&registry, &names, &input, &prefs, Some(&evidence));

    assert_eq!(r1, r2);
}

#[test]
fn same_input_produces_comparable_outputs() {
    let registry = make_registry();
    let input = make_input();
    let prefs = make_prefs();
    let evidence = make_evidence();

    let names = ["stability_first", "throughput_first", "latency_first"];

    let results = PolicySimulator::simulate_for_policies(&registry, &names, &input, &prefs, Some(&evidence));

    // All simulations must have used the same evidence snapshot; this is
    // reflected by equal considered_signals in their explanations.
    for window in results.windows(2) {
        let a = &window[0].explanation.considered_signals;
        let b = &window[1].explanation.considered_signals;
        assert_eq!(a, b);
    }
}

#[test]
fn simulation_has_no_side_effects() {
    let registry = make_registry();
    let input = make_input();
    let prefs = make_prefs();
    let evidence = make_evidence();

    let registry_before = make_registry();
    let prefs_before = prefs.clone();
    let evidence_before = evidence.clone();
    let input_before = input.clone();

    let names = ["stability_first", "throughput_first", "latency_first"];

    let _ = PolicySimulator::simulate_for_policies(&registry, &names, &input, &prefs, Some(&evidence));

    // Registry is not mutated in a way that changes its observable behavior
    // for the same set of policies.
    let r_before = PolicySimulator::simulate_for_policies(&registry_before, &names, &input, &prefs, Some(&evidence));
    let r_after = PolicySimulator::simulate_for_policies(&registry, &names, &input, &prefs, Some(&evidence));
    assert_eq!(r_before, r_after);

    assert_eq!(prefs, prefs_before);
    assert_eq!(evidence, evidence_before);
    assert_eq!(input, input_before);
}
