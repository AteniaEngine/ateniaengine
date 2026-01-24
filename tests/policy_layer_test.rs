#![allow(clippy::float_cmp)]

use std::collections::HashSet;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::builtin::constrained_hardware::ConstrainedHardwarePolicy;
use v15::policy::builtin::latency_first::LatencyFirstPolicy;
use v15::policy::builtin::power_saving::PowerSavingPolicy;
use v15::policy::builtin::stability_first::StabilityFirstPolicy;
use v15::policy::builtin::throughput_first::ThroughputFirstPolicy;
use v15::policy::policy::ExecutionPolicy;
use v15::policy::registry::PolicyRegistry;
use v15::policy::types::{DecisionBias, PolicyInput};

fn assert_normalized(bias: &DecisionBias) {
    assert!(bias.is_normalized());
}

#[test]
fn policies_are_deterministic_for_same_input() {
    let input = PolicyInput::default();

    let policies: Vec<Box<dyn v15::policy::policy::ExecutionPolicy>> = vec![
        Box::new(StabilityFirstPolicy),
        Box::new(ThroughputFirstPolicy),
        Box::new(LatencyFirstPolicy),
        Box::new(ConstrainedHardwarePolicy),
        Box::new(PowerSavingPolicy),
    ];

    for policy in policies.iter() {
        let first = policy.evaluate(&input);
        let second = policy.evaluate(&input);
        assert_eq!(first, second, "policy {} must be deterministic", policy.name());
        assert_normalized(&first);
    }
}

#[test]
fn repeated_evaluation_has_no_side_effects() {
    let input = PolicyInput::default();
    let policy = StabilityFirstPolicy;

    let first = policy.evaluate(&input);

    for _ in 0..10 {
        let again = policy.evaluate(&input);
        assert_eq!(first, again);
    }
}

#[test]
fn registry_registers_and_lists_all_policies() {
    let mut registry = PolicyRegistry::new();

    registry.register(StabilityFirstPolicy);
    registry.register(ThroughputFirstPolicy);
    registry.register(LatencyFirstPolicy);
    registry.register(ConstrainedHardwarePolicy);
    registry.register(PowerSavingPolicy);

    let names: HashSet<&'static str> = registry.list().into_iter().collect();

    let expected: HashSet<&'static str> = vec![
        "stability_first",
        "throughput_first",
        "latency_first",
        "constrained_hardware",
        "power_saving",
    ]
    .into_iter()
    .collect();

    assert_eq!(names, expected);

    for name in &expected {
        let policy = registry.get(name).expect("policy must be present in registry");
        let bias = policy.evaluate(&PolicyInput::default());
        assert_normalized(&bias);
    }
}
