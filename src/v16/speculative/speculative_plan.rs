#![allow(dead_code)]

use crate::v16::planner::execution_plan::ExecutionPlan;

/// A speculative plan derived from a base execution plan.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativePlan {
    pub base_plan: ExecutionPlan,
    /// Marker indicating that this plan is intended for speculative execution.
    pub speculative_only: bool,
}

impl SpeculativePlan {
    pub fn from_base(plan: &ExecutionPlan) -> Self {
        Self {
            base_plan: plan.clone(),
            speculative_only: true,
        }
    }
}
