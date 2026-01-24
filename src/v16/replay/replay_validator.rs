#![allow(dead_code)]

use crate::v16::feedback::execution_event::{ExecutionEvent, ExecutionEventKind};
use crate::v16::feedback::execution_outcome::ExecutionOutcome;
use crate::v16::planner::execution_plan::ExecutionPlan;

use super::replay_errors::ReplayError;

pub struct ReplayValidator;

impl ReplayValidator {
    pub fn validate(
        plan: &ExecutionPlan,
        events: &[ExecutionEvent],
        _outcome: &ExecutionOutcome,
    ) -> Result<(), ReplayError> {
        if events.is_empty() {
            return Err(ReplayError::MissingInformation(
                "No events provided for replay".to_string(),
            ));
        }

        // Ensure timestamps are non-decreasing.
        let mut ts = 0u64;
        for e in events {
            if e.logical_timestamp < ts {
                return Err(ReplayError::InconsistentHistory(
                    "Events are not ordered by logical timestamp".to_string(),
                ));
            }
            ts = e.logical_timestamp;
        }

        // Collect succeeded step indices from events and compare shape with plan.
        let mut succeeded_indices: Vec<usize> = Vec::new();
        for e in events {
            if let ExecutionEventKind::StepSucceeded = e.kind {
                if let Some(idx) = e.step_index {
                    succeeded_indices.push(idx);
                }
            }
        }

        if succeeded_indices.len() != plan.steps.len() {
            return Err(ReplayError::InconsistentHistory(
                "Number of succeeded steps does not match plan".to_string(),
            ));
        }

        for (i, idx) in succeeded_indices.iter().enumerate() {
            if *idx != i {
                return Err(ReplayError::InconsistentHistory(
                    "Succeeded steps do not follow plan order".to_string(),
                ));
            }
        }

        // Outcome must at least be compatible with having executed all steps.
        // Here we accept any outcome kind; more detailed checks can be added in
        // future versions.

        Ok(())
    }
}
