#![allow(dead_code)]

use crate::v15::policy::types::DecisionBias;
use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::feedback::execution_event::ExecutionEvent;
use crate::v16::feedback::execution_outcome::ExecutionOutcome;
use crate::v16::guards::guard_action::GuardAction;
use crate::v16::speculative::speculative_plan::SpeculativePlan;

#[derive(Debug, Clone, PartialEq)]
pub struct StepExecutionExplanation {
    pub step_index: usize,
    pub description: String,
    pub guard_action: Option<GuardAction>,
    pub speculative: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionExplanation {
    pub summary: String,
    pub decision_bias: DecisionBias,
    pub contract: ExecutionContract,
    pub plan_summary: String,
    pub steps: Vec<StepExecutionExplanation>,
    pub events: Vec<ExecutionEvent>,
    pub outcome: ExecutionOutcome,
    pub speculative_plan: Option<SpeculativePlan>,
}
