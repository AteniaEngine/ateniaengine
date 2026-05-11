#![allow(dead_code)]

use crate::v15::policy::evidence::signals::PolicySignalKind;
use crate::v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use crate::v15::policy::preferences::user_preferences::UserPreferences;
use crate::v15::policy::types::DecisionBias;

#[derive(Debug, Clone, PartialEq)]
pub enum PreferenceStatus {
    Applied,
    IgnoredDueToRisk,
    Inactive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceExplanation {
    pub name: &'static str,
    pub status: PreferenceStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalExplanation {
    pub kind: PolicySignalKind,
    pub score: f32,
}

/// Structured explanation of why a particular DecisionBias was produced.
///
/// This is a pure, derived view and must not alter any state or bias.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyExplanation {
    pub policy_name: String,
    pub final_bias: DecisionBias,
    pub considered_signals: Vec<SignalExplanation>,
    pub preference_explanations: Vec<PreferenceExplanation>,
    pub notes: Vec<String>,
}

impl PolicyExplanation {
    /// Builds an explanation from the final bias, user preferences, and
    /// optional evidence snapshot. This function mirrors, but does not
    /// execute, the logic used to apply preferences.
    pub fn from_bias_and_context(
        policy_name: &str,
        final_bias: DecisionBias,
        base_bias_before_prefs: DecisionBias,
        prefs: &UserPreferences,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> Self {
        let mut considered_signals: Vec<SignalExplanation> = Vec::new();
        let mut risk_score = 0.0f32;

        if let Some(snapshot) = evidence {
            for signal in snapshot.all_signals() {
                considered_signals.push(SignalExplanation {
                    kind: signal.kind.clone(),
                    score: signal.score,
                });

                match signal.kind {
                    PolicySignalKind::PreOomSignal | PolicySignalKind::RecentRecovery => {
                        if signal.score > risk_score {
                            risk_score = signal.score;
                        }
                    }
                    PolicySignalKind::HighMemoryPressure
                    | PolicySignalKind::FragmentationWarning => {
                        let contrib = 0.7 * signal.score;
                        if contrib > risk_score {
                            risk_score = contrib;
                        }
                    }
                    PolicySignalKind::StableLatency => {}
                }
            }
        }

        let prefs_ignored = risk_score >= 0.7;

        let mut preference_explanations: Vec<PreferenceExplanation> = Vec::new();

        let push_pref = |vec: &mut Vec<PreferenceExplanation>, name: &'static str, active: bool| {
            let status = if !active {
                PreferenceStatus::Inactive
            } else if prefs_ignored {
                PreferenceStatus::IgnoredDueToRisk
            } else {
                PreferenceStatus::Applied
            };
            vec.push(PreferenceExplanation { name, status });
        };

        push_pref(
            &mut preference_explanations,
            "prefer_latency",
            prefs.prefer_latency,
        );
        push_pref(&mut preference_explanations, "avoid_ssd", prefs.avoid_ssd);
        push_pref(
            &mut preference_explanations,
            "prioritize_stability",
            prefs.prioritize_stability,
        );
        push_pref(
            &mut preference_explanations,
            "minimize_power",
            prefs.minimize_power,
        );
        push_pref(&mut preference_explanations, "prefer_gpu", prefs.prefer_gpu);

        let mut notes: Vec<String> = Vec::new();

        if prefs_ignored {
            notes.push("User preferences were ignored due to high risk score".to_string());
        }

        if final_bias != base_bias_before_prefs {
            notes.push(
                "Final bias reflects both evidence and (possibly scaled) user preferences"
                    .to_string(),
            );
        } else {
            notes.push("Final bias matches base bias; either no preferences were active or risk overrode them".to_string());
        }

        PolicyExplanation {
            policy_name: policy_name.to_string(),
            final_bias,
            considered_signals,
            preference_explanations,
            notes,
        }
    }
}
