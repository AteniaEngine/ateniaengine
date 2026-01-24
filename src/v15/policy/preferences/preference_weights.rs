#![allow(dead_code)]

use crate::v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use crate::v15::policy::evidence::signals::PolicySignalKind;
use crate::v15::policy::preferences::user_preferences::UserPreferences;
use crate::v15::policy::types::DecisionBias;

/// Safely applies user preferences as soft adjustments over an existing
/// (policy + evidence) decision bias. Preferences are always bounded,
/// optional, and overridden by strong risk signals.
pub fn apply_user_preferences(
    base: &DecisionBias,
    prefs: &UserPreferences,
    evidence: Option<&PolicyEvidenceSnapshot>,
) -> DecisionBias {
    // Start from the existing bias.
    let mut out = base.clone();

    // Compute an aggregate "risk" score from evidence.
    let mut risk_score = 0.0f32;

    if let Some(snapshot) = evidence {
        for signal in snapshot.all_signals() {
            match signal.kind {
                PolicySignalKind::PreOomSignal | PolicySignalKind::RecentRecovery => {
                    if signal.score > risk_score {
                        risk_score = signal.score;
                    }
                }
                PolicySignalKind::HighMemoryPressure
                | PolicySignalKind::FragmentationWarning => {
                    // Memory-related risk also contributes, but slightly less.
                    let contrib = 0.7 * signal.score;
                    if contrib > risk_score {
                        risk_score = contrib;
                    }
                }
                PolicySignalKind::StableLatency => {
                    // Stable latency does not increase risk.
                }
            }
        }
    }

    // Safety dominance: if risk is very high, ignore user preferences entirely.
    if risk_score >= 0.7 {
        return out;
    }

    // For medium risk, attenuate preferences.
    let preference_scale = if risk_score >= 0.3 { 0.5 } else { 1.0 };

    let apply_delta = |value: &mut f32, delta: f32| {
        let scaled = delta * preference_scale;
        *value = (*value + scaled).clamp(0.0, 1.0);
    };

    // Apply soft preferences. All adjustments are small and bounded.
    if prefs.prefer_latency {
        // Favor lower latency, but never reduce stability directly. Under
        // medium/high risk, this effect is also attenuated by
        // `preference_scale`.
        apply_delta(&mut out.latency_weight, 0.15);
    }

    if prefs.prioritize_stability {
        apply_delta(&mut out.stability_weight, 0.15);
        apply_delta(&mut out.risk_weight, -0.05);
    }

    if prefs.minimize_power {
        apply_delta(&mut out.offload_cost_weight, -0.1);
        apply_delta(&mut out.latency_weight, -0.05);
    }

    if prefs.prefer_gpu {
        // Interpreted abstractly as being more willing to pay offload cost,
        // without binding to a specific backend.
        apply_delta(&mut out.offload_cost_weight, -0.05);
    }

    if prefs.avoid_ssd {
        // Favor avoiding heavy reliance on lower tiers by slightly
        // increasing memory_pressure_weight and risk_weight.
        apply_delta(&mut out.memory_pressure_weight, 0.05);
        apply_delta(&mut out.risk_weight, 0.05);
    }

    out
}
