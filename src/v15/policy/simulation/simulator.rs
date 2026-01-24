#![allow(dead_code)]

use crate::v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use crate::v15::policy::explain::explanation::PolicyExplanation;
use crate::v15::policy::preferences::preference_weights::apply_user_preferences;
use crate::v15::policy::preferences::user_preferences::UserPreferences;
use crate::v15::policy::registry::PolicyRegistry;
use crate::v15::policy::simulation::simulation_result::SimulationResult;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

/// Pure what-if simulator for execution policies.
///
/// PolicySimulator does not track state, does not touch the active
/// policy, and does not interact with the runtime. It only evaluates
/// policies against a given input, evidence snapshot and user
/// preferences, producing comparable SimulationResult values.
pub struct PolicySimulator;

impl PolicySimulator {
    pub fn simulate_for_policies(
        registry: &PolicyRegistry,
        policy_names: &[&str],
        input: &PolicyInput,
        prefs: &UserPreferences,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> Vec<SimulationResult> {
        let mut results: Vec<SimulationResult> = Vec::new();

        for name in policy_names {
            if let Some(policy) = registry.get(name) {
                // Policy + evidence bias.
                let base_bias: DecisionBias = policy.evaluate_with_evidence(input, evidence);

                // Apply user preferences in a pure, bounded way.
                let final_bias = apply_user_preferences(&base_bias, prefs, evidence);

                // Build explanation from the final bias and context.
                let explanation = PolicyExplanation::from_bias_and_context(
                    policy.name(),
                    final_bias.clone(),
                    base_bias.clone(),
                    prefs,
                    evidence,
                );

                let sim_result = SimulationResult::new(
                    policy.name(),
                    final_bias,
                    explanation,
                    &base_bias,
                );

                results.push(sim_result);
            }
        }

        results
    }
}
