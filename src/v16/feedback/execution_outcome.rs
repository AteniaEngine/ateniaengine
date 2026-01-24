#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionOutcomeKind {
    Completed,
    Failed,
    Aborted,
    PartiallyCompleted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionOutcome {
    pub kind: ExecutionOutcomeKind,
    /// Indices of successfully executed steps.
    pub executed_steps: Vec<usize>,
    /// Final error message, if any.
    pub final_error: Option<String>,
}
