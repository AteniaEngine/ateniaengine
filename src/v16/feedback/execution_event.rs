#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionEventKind {
    ExecutionStarted,
    StepStarted,
    StepSucceeded,
    ExecutionFailed,
    ExecutionAborted,
    ExecutionCompleted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionEvent {
    pub kind: ExecutionEventKind,
    /// Index of the associated step in the plan, if applicable.
    pub step_index: Option<usize>,
    /// Logical timestamp (monotonic counter) assigned by the emitter.
    pub logical_timestamp: u64,
    pub severity: EventSeverity,
    /// Minimal, human-readable description.
    pub message: String,
}
