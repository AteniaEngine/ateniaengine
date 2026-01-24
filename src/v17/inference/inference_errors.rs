#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    InvalidArtifact(String),
    LoadFailed(String),
    ContractError(String),
    PlanningError(String),
    AdapterError(String),
    FeedbackError(String),
}
