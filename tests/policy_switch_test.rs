#![allow(clippy::float_cmp)]

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::builtin::latency_first::LatencyFirstPolicy;
use v15::policy::builtin::stability_first::StabilityFirstPolicy;
use v15::policy::builtin::throughput_first::ThroughputFirstPolicy;
use v15::policy::manager::policy_manager::PolicyManager;
use v15::policy::registry::PolicyRegistry;
use v15::policy::types::{DecisionBias, PolicyInput};

fn make_registry() -> PolicyRegistry {
    let mut registry = PolicyRegistry::new();
    registry.register(StabilityFirstPolicy);
    registry.register(ThroughputFirstPolicy);
    registry.register(LatencyFirstPolicy);
    registry
}

fn sample_bias(manager: &PolicyManager) -> DecisionBias {
    let policy = manager.active_policy();
    policy.evaluate(&PolicyInput::default())
}

#[test]
fn initial_policy_is_defined() {
    let registry = make_registry();
    let manager = PolicyManager::new(registry, "stability_first");

    assert_eq!(manager.active_policy_name(), "stability_first");
}

#[test]
fn switch_changes_active_policy() {
    let registry = make_registry();
    let mut manager = PolicyManager::new(registry, "stability_first");

    let ok = manager.set_active_policy("latency_first");
    assert!(ok);
    assert_eq!(manager.active_policy_name(), "latency_first");
}

#[test]
fn invalid_switch_is_ignored() {
    let registry = make_registry();
    let mut manager = PolicyManager::new(registry, "stability_first");

    let before = manager.active_policy_name().to_string();

    let ok = manager.set_active_policy("does_not_exist");
    assert!(!ok);
    assert_eq!(manager.active_policy_name(), before);
}

#[test]
fn switch_does_not_reset_state() {
    let registry = make_registry();
    let mut manager = PolicyManager::new(registry, "stability_first");

    let bias_before = sample_bias(&manager);

    // Perform a sequence of switches and then return to the original
    // policy. The sampled bias must be identical, showing that the
    // manager does not reset policy behavior or external state.
    manager.set_active_policy("throughput_first");
    manager.set_active_policy("latency_first");
    manager.set_active_policy("stability_first");

    let bias_after = sample_bias(&manager);
    assert_eq!(bias_before, bias_after);
}

#[test]
fn determinism_under_repeated_switches() {
    let registry1 = make_registry();
    let registry2 = make_registry();

    let mut m1 = PolicyManager::new(registry1, "stability_first");
    let mut m2 = PolicyManager::new(registry2, "stability_first");

    let sequence = [
        "throughput_first",
        "latency_first",
        "stability_first",
        "latency_first",
        "stability_first",
    ];

    for name in &sequence {
        let _ = m1.set_active_policy(name);
        let _ = m2.set_active_policy(name);
    }

    assert_eq!(m1.active_policy_name(), m2.active_policy_name());
}
