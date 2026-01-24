#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;

use super::plan_step::PlanStep;

/// Immutable planning artifact derived from an ExecutionContract.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPlan {
    /// The contract that this plan is based on.
    pub contract: ExecutionContract,
    /// Ordered sequence of high-level steps that would be attempted.
    pub steps: Vec<PlanStep>,
    /// Whether the planner considers this plan globally abortable at any time.
    pub globally_abortable: bool,
}
