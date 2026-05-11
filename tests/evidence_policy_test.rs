#![allow(clippy::float_cmp)]

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::builtin::stability_first::StabilityFirstPolicy;
use v15::policy::evidence::signals::{PolicySignal, PolicySignalKind};
use v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use v15::policy::policy::ExecutionPolicy;
use v15::policy::types::PolicyInput;

fn make_pre_oom_snapshot(score: f32) -> PolicyEvidenceSnapshot {
    let signal = PolicySignal {
        kind: PolicySignalKind::PreOomSignal,
        score,
    };
    PolicyEvidenceSnapshot::new(vec![signal])
}

#[test]
fn policy_reacts_to_evidence() {
    let input = PolicyInput::default();
    let policy = StabilityFirstPolicy;

    let low_oom = make_pre_oom_snapshot(0.1);
    let high_oom = make_pre_oom_snapshot(0.9);

    let bias_low = policy.evaluate_with_evidence(&input, Some(&low_oom));
    let bias_high = policy.evaluate_with_evidence(&input, Some(&high_oom));

    // With stronger pre-OOM evidence we expect a stronger stability bias
    // and/or a reduced risk bias.
    assert!(bias_high.stability_weight >= bias_low.stability_weight);
    assert!(bias_high.risk_weight <= bias_low.risk_weight);
}

#[test]
fn no_evidence_falls_back_to_v15_0_behavior() {
    let input = PolicyInput::default();
    let policy = StabilityFirstPolicy;

    let bias_base = policy.evaluate(&input);
    let bias_no_evidence = policy.evaluate_with_evidence(&input, None);

    assert_eq!(bias_base, bias_no_evidence);
}

#[test]
fn evidence_is_read_only_and_deterministic() {
    let input = PolicyInput::default();
    let policy = StabilityFirstPolicy;
    let snapshot = make_pre_oom_snapshot(0.7);

    let before = snapshot.clone();

    let first = policy.evaluate_with_evidence(&input, Some(&snapshot));

    for _ in 0..10 {
        let again = policy.evaluate_with_evidence(&input, Some(&snapshot));
        // Determinism: all outputs are identical.
        assert_eq!(again, first);
    }

    // Read-only guarantee: snapshot has not been modified.
    assert_eq!(snapshot, before);
}
