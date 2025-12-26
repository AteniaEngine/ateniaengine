use super::execution_planner::{DecisionRule, ExecutionTarget};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionDecisionTrace {
    pub kernel_name: String,
    pub evaluated_rules: Vec<DecisionRule>,
    pub winning_rule: DecisionRule,
    pub target: ExecutionTarget,
    pub reason: String,
}
