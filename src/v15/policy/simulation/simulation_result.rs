#![allow(dead_code)]

use crate::v15::policy::explain::explanation::PolicyExplanation;
use crate::v15::policy::types::DecisionBias;

#[derive(Debug, Clone, PartialEq)]
pub struct SimulationResult {
    pub policy_name: String,
    pub bias: DecisionBias,
    pub explanation: PolicyExplanation,
    pub more_conservative_than_base: bool,
    pub more_aggressive_than_base: bool,
}

impl SimulationResult {
    pub fn new(
        policy_name: &str,
        bias: DecisionBias,
        explanation: PolicyExplanation,
        base_bias_before_prefs: &DecisionBias,
    ) -> Self {
        let more_conservative_than_base = bias.stability_weight
            >= base_bias_before_prefs.stability_weight
            && bias.risk_weight <= base_bias_before_prefs.risk_weight;

        let more_aggressive_than_base = bias.latency_weight > base_bias_before_prefs.latency_weight
            && bias.risk_weight >= base_bias_before_prefs.risk_weight;

        SimulationResult {
            policy_name: policy_name.to_string(),
            bias,
            explanation,
            more_conservative_than_base,
            more_aggressive_than_base,
        }
    }
}
