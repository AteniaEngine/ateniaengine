#![allow(dead_code)]

use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

pub struct ThroughputFirstPolicy;

impl ExecutionPolicy for ThroughputFirstPolicy {
    fn name(&self) -> &'static str {
        "throughput_first"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.9,
            stability_weight: 0.5,
            memory_pressure_weight: 0.6,
            offload_cost_weight: 0.2,
        }
    }
}
