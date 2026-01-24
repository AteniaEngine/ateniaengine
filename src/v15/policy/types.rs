#![allow(dead_code)]

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyInput {
    // APX 15.0: intentionally empty. Future versions will extend this.
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionBias {
    pub risk_weight: f32,
    pub latency_weight: f32,
    pub stability_weight: f32,
    pub memory_pressure_weight: f32,
    pub offload_cost_weight: f32,
}

impl DecisionBias {
    pub fn is_normalized(&self) -> bool {
        fn in_range(v: f32) -> bool {
            (0.0..=1.0).contains(&v)
        }

        in_range(self.risk_weight)
            && in_range(self.latency_weight)
            && in_range(self.stability_weight)
            && in_range(self.memory_pressure_weight)
            && in_range(self.offload_cost_weight)
    }
}
