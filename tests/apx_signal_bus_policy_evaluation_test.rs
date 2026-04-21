//! APX v19 milestone B: end-to-end sensor → evidence → policy cycle.
//!
//! Demonstrates SignalBus producing PolicyEvidenceSnapshot from live
//! memory telemetry, and StabilityFirstPolicy (v15) reacting to it via
//! `evaluate_with_evidence`. First real-evidence policy evaluation in
//! the suite.

use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::v15::policy::builtin::stability_first::StabilityFirstPolicy;
use atenia_engine::v15::policy::evidence::signals::PolicySignalKind;
use atenia_engine::v15::policy::policy::ExecutionPolicy;
use atenia_engine::v15::policy::types::PolicyInput;

#[test]
fn test_signal_bus_drives_policy_evaluation() {
    let bus = SignalBus::new();

    let evidence = match bus.collect_policy_evidence() {
        Some(e) => e,
        None => {
            eprintln!("SKIPPED: memory probe unavailable");
            return;
        }
    };

    let policy = StabilityFirstPolicy;
    let input = PolicyInput::default();

    let bias_with = policy.evaluate_with_evidence(&input, Some(&evidence));
    let bias_without = policy.evaluate_with_evidence(&input, None);

    assert!(
        bias_with.is_normalized(),
        "resulting DecisionBias must be normalized"
    );

    // StabilityFirstPolicy reacts only to PreOomSignal. If the current
    // machine state produced one, the evidence-aware bias must show a
    // stability weight no lower than the baseline, and a risk weight
    // no higher. If not, the default impl returns the base bias and
    // the comparison is trivially skipped.
    let has_pre_oom = evidence
        .all_signals()
        .iter()
        .any(|s| s.kind == PolicySignalKind::PreOomSignal);

    if has_pre_oom {
        assert!(
            bias_with.stability_weight >= bias_without.stability_weight,
            "stability_weight must not decrease under PreOomSignal \
             (with = {}, without = {})",
            bias_with.stability_weight,
            bias_without.stability_weight,
        );
        assert!(
            bias_with.risk_weight <= bias_without.risk_weight,
            "risk_weight must not increase under PreOomSignal \
             (with = {}, without = {})",
            bias_with.risk_weight,
            bias_without.risk_weight,
        );
    }
}
