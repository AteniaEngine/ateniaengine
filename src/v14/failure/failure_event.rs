#![allow(dead_code)]

use super::failure_kind::FailureKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureEvent {
    pub kind: FailureKind,
    pub timestamp: u64,
    pub message: String,
    pub device: Option<String>,
    pub tensor_id: Option<String>,
    pub kernel_id: Option<String>,
    pub severity: FailureSeverity,
}

impl FailureEvent {
    pub fn new(
        kind: FailureKind,
        timestamp: u64,
        message: String,
        device: Option<String>,
        tensor_id: Option<String>,
        kernel_id: Option<String>,
        severity: FailureSeverity,
    ) -> Self {
        FailureEvent {
            kind,
            timestamp,
            message,
            device,
            tensor_id,
            kernel_id,
            severity,
        }
    }
}
