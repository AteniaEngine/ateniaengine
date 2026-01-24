#![allow(clippy::float_cmp)]

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use v15::policy::evidence::signals::{PolicySignal, PolicySignalKind};
use v15::policy::preferences::preference_weights::apply_user_preferences;
use v15::policy::preferences::user_preferences::UserPreferences;
use v15::policy::types::DecisionBias;

fn make_bias() -> DecisionBias {
    DecisionBias {
        risk_weight: 0.4,
        latency_weight: 0.4,
        stability_weight: 0.4,
        memory_pressure_weight: 0.4,
        offload_cost_weight: 0.4,
    }
}

fn pre_oom_evidence(score: f32) -> PolicyEvidenceSnapshot {
    PolicyEvidenceSnapshot::new(vec![PolicySignal {
        kind: PolicySignalKind::PreOomSignal,
        score,
    }])
}

#[test]
fn preferences_are_soft() {
    let base = make_bias();
    let prefs = UserPreferences {
        prefer_latency: true,
        avoid_ssd: true,
        prioritize_stability: true,
        minimize_power: true,
        prefer_gpu: true,
    };

    let adjusted = apply_user_preferences(&base, &prefs, None);

    // Preferences may move weights, but they must stay in a reasonable range
    // and not saturate purely due to preferences.
    for w in [
        adjusted.risk_weight,
        adjusted.latency_weight,
        adjusted.stability_weight,
        adjusted.memory_pressure_weight,
        adjusted.offload_cost_weight,
    ] {
        assert!(w >= 0.0 && w <= 1.0);
    }
}

#[test]
fn preferences_are_overridable_by_risk() {
    let base = make_bias();
    let prefs = UserPreferences {
        prefer_latency: true,
        avoid_ssd: true,
        prioritize_stability: true,
        minimize_power: true,
        prefer_gpu: true,
    };

    let high_risk = pre_oom_evidence(0.9);

    let adjusted = apply_user_preferences(&base, &prefs, Some(&high_risk));

    // Under very high risk, preferences should be effectively ignored.
    assert_eq!(adjusted, base);
}

#[test]
fn no_preferences_fallback_to_bias() {
    let base = make_bias();
    let prefs = UserPreferences::default();

    let adjusted_no_evidence = apply_user_preferences(&base, &prefs, None);
    let adjusted_with_evidence = apply_user_preferences(&base, &prefs, Some(&pre_oom_evidence(0.5)));

    // Without any active preferences, behavior must match the base bias,
    // regardless of evidence.
    assert_eq!(adjusted_no_evidence, base);
    assert_eq!(adjusted_with_evidence, base);
}

#[test]
fn preferences_are_deterministic() {
    let base = make_bias();
    let prefs = UserPreferences {
        prefer_latency: true,
        avoid_ssd: false,
        prioritize_stability: true,
        minimize_power: false,
        prefer_gpu: true,
    };

    let evidence = pre_oom_evidence(0.2);

    let first = apply_user_preferences(&base, &prefs, Some(&evidence));

    for _ in 0..10 {
        let again = apply_user_preferences(&base, &prefs, Some(&evidence));
        assert_eq!(again, first);
    }
}

#[test]
fn safety_dominance_under_conflict() {
    let base = make_bias();
    let prefs = UserPreferences {
        prefer_latency: true,
        avoid_ssd: false,
        prioritize_stability: false,
        minimize_power: false,
        prefer_gpu: false,
    };

    // Medium risk: preferences may apply but must not reduce stability or
    // increase risk beyond the base bias.
    let medium_risk = pre_oom_evidence(0.6);

    let adjusted = apply_user_preferences(&base, &prefs, Some(&medium_risk));

    assert!(adjusted.stability_weight >= base.stability_weight);
    assert!(adjusted.risk_weight <= base.risk_weight);
}
