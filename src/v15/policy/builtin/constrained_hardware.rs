#![allow(dead_code)]

use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

pub struct ConstrainedHardwarePolicy;

impl ExecutionPolicy for ConstrainedHardwarePolicy {
    fn name(&self) -> &'static str {
        "constrained_hardware"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.5,
            latency_weight: 0.5,
            stability_weight: 0.7,
            memory_pressure_weight: 1.0,
            offload_cost_weight: 0.3,
        }
    }
}
