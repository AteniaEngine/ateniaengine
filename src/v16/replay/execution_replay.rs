#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::executor::execution_context::RuntimeFacade;
use crate::v16::executor::step_executor::StepExecutor;
use crate::v16::feedback::execution_event::{ExecutionEvent, ExecutionEventKind};
use crate::v16::feedback::execution_outcome::ExecutionOutcome;
use crate::v16::planner::execution_plan::ExecutionPlan;

use super::replay_context::ReplayContext;
use super::replay_errors::ReplayError;
use super::replay_validator::ReplayValidator;

pub struct ExecutionReplay<R: RuntimeFacade> {
    pub contract: ExecutionContract,
    pub plan: ExecutionPlan,
    pub events: Vec<ExecutionEvent>,
    pub outcome: ExecutionOutcome,
    pub context: ReplayContext<R>,
}

impl<R: RuntimeFacade> ExecutionReplay<R> {
    pub fn new(
        contract: ExecutionContract,
        plan: ExecutionPlan,
        events: Vec<ExecutionEvent>,
        outcome: ExecutionOutcome,
        context: ReplayContext<R>,
    ) -> Self {
        Self {
            contract,
            plan,
            events,
            outcome,
            context,
        }
    }

    pub fn replay(&mut self) -> Result<(), ReplayError> {
        ReplayValidator::validate(&self.plan, &self.events, &self.outcome)?;

        // Derive the order of steps to replay from the events.
        let mut step_indices: Vec<usize> = Vec::new();
        for e in &self.events {
            if let ExecutionEventKind::StepSucceeded = e.kind {
                if let Some(idx) = e.step_index {
                    step_indices.push(idx);
                }
            }
        }

        // Execute steps in the same order using the isolated replay context.
        for idx in step_indices {
            let step = &self.plan.steps[idx];
            StepExecutor::execute_step(step, &mut self.context.context)
                .map_err(|e| ReplayError::DivergentOutcome(format!(
                    "replay step failed: {:?}",
                    e
                )))?;
        }

        Ok(())
    }
}
