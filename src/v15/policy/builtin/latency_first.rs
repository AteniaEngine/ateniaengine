#![allow(dead_code)]

use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

pub struct LatencyFirstPolicy;

impl ExecutionPolicy for LatencyFirstPolicy {
    fn name(&self) -> &'static str {
        "latency_first"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.4,
            latency_weight: 1.0,
            stability_weight: 0.6,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.5,
        }
    }
}
