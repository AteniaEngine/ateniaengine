#![allow(dead_code)]

use crate::v16::executor::executor_state::ExecutorStatus;
use crate::v16::feedback::execution_event::{EventSeverity, ExecutionEvent, ExecutionEventKind};
use crate::v16::feedback::execution_outcome::{ExecutionOutcome, ExecutionOutcomeKind};
use crate::v16::feedback::feedback_errors::FeedbackError;
use crate::v16::planner::execution_plan::ExecutionPlan;

pub struct EventEmitter;

impl EventEmitter {
    /// Build a sequence of execution events and a final outcome from a snapshot
    /// of plan execution.
    pub fn emit_for_snapshot(
        plan: &ExecutionPlan,
        executed_steps: &[usize],
        status: &ExecutorStatus,
        final_error: Option<&str>,
    ) -> Result<(Vec<ExecutionEvent>, ExecutionOutcome), FeedbackError> {
        // Validate executed_steps: indices within range and non-decreasing
        if executed_steps
            .iter()
            .any(|&idx| idx >= plan.steps.len())
        {
            return Err(FeedbackError::InvalidEvent(
                "executed step index out of bounds".to_string(),
            ));
        }

        for window in executed_steps.windows(2) {
            if window[1] < window[0] {
                return Err(FeedbackError::LogicalOrderViolation(
                    "executed steps must be in non-decreasing order".to_string(),
                ));
            }
        }

        let mut events = Vec::new();
        let mut ts: u64 = 0;

        // Execution started event.
        events.push(ExecutionEvent {
            kind: ExecutionEventKind::ExecutionStarted,
            step_index: None,
            logical_timestamp: ts,
            severity: EventSeverity::Info,
            message: "Execution started".to_string(),
        });
        ts += 1;

        // Per-step events.
        for &idx in executed_steps {
            events.push(ExecutionEvent {
                kind: ExecutionEventKind::StepStarted,
                step_index: Some(idx),
                logical_timestamp: ts,
                severity: EventSeverity::Info,
                message: format!("Step {} started", idx),
            });
            ts += 1;

            events.push(ExecutionEvent {
                kind: ExecutionEventKind::StepSucceeded,
                step_index: Some(idx),
                logical_timestamp: ts,
                severity: EventSeverity::Info,
                message: format!("Step {} succeeded", idx),
            });
            ts += 1;
        }

        // Outcome and terminal event.
        let (outcome_kind, terminal_kind, severity) = match status {
            ExecutorStatus::Completed => (
                ExecutionOutcomeKind::Completed,
                ExecutionEventKind::ExecutionCompleted,
                EventSeverity::Info,
            ),
            ExecutorStatus::Failed => (
                if executed_steps.len() == plan.steps.len() {
                    ExecutionOutcomeKind::Failed
                } else {
                    ExecutionOutcomeKind::PartiallyCompleted
                },
                ExecutionEventKind::ExecutionFailed,
                EventSeverity::Error,
            ),
            ExecutorStatus::Aborted => (
                if executed_steps.is_empty() {
                    ExecutionOutcomeKind::Aborted
                } else {
                    ExecutionOutcomeKind::PartiallyCompleted
                },
                ExecutionEventKind::ExecutionAborted,
                EventSeverity::Warning,
            ),
            ExecutorStatus::Running => (
                ExecutionOutcomeKind::PartiallyCompleted,
                ExecutionEventKind::ExecutionFailed,
                EventSeverity::Warning,
            ),
        };

        let message = match status {
            ExecutorStatus::Completed => "Execution completed".to_string(),
            ExecutorStatus::Failed => final_error
                .map(|s| format!("Execution failed: {}", s))
                .unwrap_or_else(|| "Execution failed".to_string()),
            ExecutorStatus::Aborted => final_error
                .map(|s| format!("Execution aborted: {}", s))
                .unwrap_or_else(|| "Execution aborted".to_string()),
            ExecutorStatus::Running => "Execution snapshot while still running".to_string(),
        };

        events.push(ExecutionEvent {
            kind: terminal_kind,
            step_index: None,
            logical_timestamp: ts,
            severity,
            message,
        });

        let outcome = ExecutionOutcome {
            kind: outcome_kind,
            executed_steps: executed_steps.to_vec(),
            final_error: final_error.map(|s| s.to_string()),
        };

        Ok((events, outcome))
    }
}
