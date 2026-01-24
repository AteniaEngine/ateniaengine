#![allow(dead_code)]

use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

pub struct PowerSavingPolicy;

impl ExecutionPolicy for PowerSavingPolicy {
    fn name(&self) -> &'static str {
        "power_saving"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.6,
            latency_weight: 0.4,
            stability_weight: 0.7,
            memory_pressure_weight: 0.8,
            offload_cost_weight: 0.9,
        }
    }
}
