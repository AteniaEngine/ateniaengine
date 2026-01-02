#![allow(dead_code)]

use super::failure_event::FailureEvent;
use super::recovery_action::RecoveryAction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryResult {
    Recovered,
    Degraded,
    Failed,
    Avoided,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryRecord {
    pub failure_event: FailureEvent,
    pub action_taken: RecoveryAction,
    pub action_reason: String,
    pub result: RecoveryResult,
    pub timestamp: u64,
}

impl RecoveryRecord {
    pub fn new(
        failure_event: FailureEvent,
        action_taken: RecoveryAction,
        action_reason: String,
        result: RecoveryResult,
        timestamp: u64,
    ) -> Self {
        RecoveryRecord {
            failure_event,
            action_taken,
            action_reason,
            result,
            timestamp,
        }
    }
}
