#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorError {
    /// Preconditions for a step were not satisfied.
    PreconditionFailed(String),
    /// The step would be unsafe to execute under current conditions.
    UnsafeToExecute(String),
    /// The underlying runtime reported a failure while executing the step.
    StepFailed(String),
    /// Execution was explicitly aborted.
    Aborted(String),
}
