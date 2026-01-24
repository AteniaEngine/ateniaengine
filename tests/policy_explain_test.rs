#![allow(clippy::float_cmp)]

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use v15::policy::evidence::signals::{PolicySignal, PolicySignalKind};
use v15::policy::explain::explanation::{
    PolicyExplanation, PreferenceStatus,
};
use v15::policy::explain::formatter::{
    format_explanation_json, format_explanation_text,
};
use v15::policy::preferences::user_preferences::UserPreferences;
use v15::policy::types::DecisionBias;

fn base_bias() -> DecisionBias {
    DecisionBias {
        risk_weight: 0.2,
        latency_weight: 0.5,
        stability_weight: 0.8,
        memory_pressure_weight: 0.4,
        offload_cost_weight: 0.3,
    }
}

fn final_bias() -> DecisionBias {
    DecisionBias {
        risk_weight: 0.25,
        latency_weight: 0.55,
        stability_weight: 0.85,
        memory_pressure_weight: 0.45,
        offload_cost_weight: 0.28,
    }
}

fn make_evidence() -> PolicyEvidenceSnapshot {
    PolicyEvidenceSnapshot::new(vec![
        PolicySignal {
            kind: PolicySignalKind::PreOomSignal,
            score: 0.4,
        },
        PolicySignal {
            kind: PolicySignalKind::StableLatency,
            score: 0.9,
        },
    ])
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

#[test]
fn explanation_matches_bias() {
    let expl = PolicyExplanation::from_bias_and_context(
        "stability_first",
        final_bias(),
        base_bias(),
        &make_prefs(),
        Some(&make_evidence()),
    );

    assert_eq!(expl.final_bias, final_bias());
    assert_eq!(expl.policy_name, "stability_first");
}

#[test]
fn explanation_is_deterministic() {
    let e1 = PolicyExplanation::from_bias_and_context(
        "stability_first",
        final_bias(),
        base_bias(),
        &make_prefs(),
        Some(&make_evidence()),
    );

    let e2 = PolicyExplanation::from_bias_and_context(
        "stability_first",
        final_bias(),
        base_bias(),
        &make_prefs(),
        Some(&make_evidence()),
    );

    assert_eq!(e1, e2);

    let t1 = format_explanation_text(&e1);
    let t2 = format_explanation_text(&e2);
    assert_eq!(t1, t2);

    let j1 = format_explanation_json(&e1);
    let j2 = format_explanation_json(&e2);
    assert_eq!(j1, j2);
}

#[test]
fn explanation_has_no_side_effects() {
    let prefs = make_prefs();
    let evidence = make_evidence();
    let base = base_bias();
    let final_b = final_bias();

    let prefs_before = prefs.clone();
    let evidence_before = evidence.clone();
    let base_before = base.clone();
    let final_before = final_b.clone();

    let _ = PolicyExplanation::from_bias_and_context(
        "stability_first",
        final_b,
        base,
        &prefs,
        Some(&evidence),
    );

    assert_eq!(prefs, prefs_before);
    assert_eq!(evidence, evidence_before);
    assert_eq!(base_bias(), base_before);
    assert_eq!(final_bias(), final_before);
}

#[test]
fn ignored_preferences_are_explained() {
    // High-risk evidence so that preferences are considered ignored.
    let evidence = PolicyEvidenceSnapshot::new(vec![PolicySignal {
        kind: PolicySignalKind::PreOomSignal,
        score: 0.9,
    }]);

    let prefs = make_prefs();

    let expl = PolicyExplanation::from_bias_and_context(
        "stability_first",
        final_bias(),
        base_bias(),
        &prefs,
        Some(&evidence),
    );

    // All active preferences should be marked as ignored_due_to_risk.
    for p in &expl.preference_explanations {
        if p.name == "prefer_latency" || p.name == "prioritize_stability" || p.name == "prefer_gpu" {
            assert_eq!(p.status, PreferenceStatus::IgnoredDueToRisk);
        }
    }
}

#[test]
fn formatter_output_is_stable() {
    let expl = PolicyExplanation::from_bias_and_context(
        "stability_first",
        final_bias(),
        base_bias(),
        &make_prefs(),
        Some(&make_evidence()),
    );

    let text1 = format_explanation_text(&expl);
    let text2 = format_explanation_text(&expl);
    assert_eq!(text1, text2);

    let json1 = format_explanation_json(&expl);
    let json2 = format_explanation_json(&expl);
    assert_eq!(json1, json2);
}
