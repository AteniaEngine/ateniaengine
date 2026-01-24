#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorStatus {
    Running,
    Aborted,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutorState {
    pub current_step: usize,
    pub status: ExecutorStatus,
    /// Indices of successfully executed steps in the plan.
    pub executed_steps: Vec<usize>,
}

impl ExecutorState {
    pub fn new() -> Self {
        Self {
            current_step: 0,
            status: ExecutorStatus::Running,
            executed_steps: Vec::new(),
        }
    }
}
