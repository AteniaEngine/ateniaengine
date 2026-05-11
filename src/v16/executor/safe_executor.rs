#![allow(dead_code)]

use crate::v16::executor::execution_context::{ExecutionContext, RuntimeFacade};
use crate::v16::executor::executor_errors::ExecutorError;
use crate::v16::executor::executor_state::{ExecutorState, ExecutorStatus};
use crate::v16::executor::step_executor::StepExecutor;
use crate::v16::planner::execution_plan::ExecutionPlan;

pub struct SafeExecutor<R: RuntimeFacade> {
    pub state: ExecutorState,
    pub context: ExecutionContext<R>,
    pub plan: ExecutionPlan,
}

impl<R: RuntimeFacade> SafeExecutor<R> {
    pub fn new(plan: ExecutionPlan, context: ExecutionContext<R>) -> Self {
        Self {
            state: ExecutorState::new(),
            context,
            plan,
        }
    }

    pub fn status(&self) -> ExecutorStatus {
        self.state.status.clone()
    }

    /// Executes a single step from the plan, if any remain.
    pub fn step(&mut self) -> Result<(), ExecutorError> {
        match self.state.status {
            ExecutorStatus::Running => {}
            ExecutorStatus::Aborted => {
                return Err(ExecutorError::Aborted(
                    "Executor is aborted; no further steps may run".to_string(),
                ));
            }
            ExecutorStatus::Completed | ExecutorStatus::Failed => {
                // No further execution; treat as a no-op.
                return Ok(());
            }
        }

        if self.state.current_step >= self.plan.steps.len() {
            self.state.status = ExecutorStatus::Completed;
            return Ok(());
        }

        let idx = self.state.current_step;
        let step = &self.plan.steps[idx];

        // Basic validation: every step must be marked abortable.
        if !step.abortable {
            self.state.status = ExecutorStatus::Failed;
            return Err(ExecutorError::UnsafeToExecute(
                "Non-abortable step encountered in plan".to_string(),
            ));
        }

        // In APX 16.2 we do not attempt to interpret the textual preconditions;
        // we assume the planner and contract already ensured conceptual
        // consistency. Additional structural checks could be added here.

        match StepExecutor::execute_step(step, &mut self.context) {
            Ok(()) => {
                self.state.executed_steps.push(idx);
                self.state.current_step += 1;
                if self.state.current_step >= self.plan.steps.len() {
                    self.state.status = ExecutorStatus::Completed;
                }
                Ok(())
            }
            Err(e) => {
                self.state.status = ExecutorStatus::Failed;
                Err(e)
            }
        }
    }

    /// Explicitly aborts execution. After calling this, no further steps will be
    /// executed.
    pub fn abort(&mut self, reason: &str) -> ExecutorError {
        self.state.status = ExecutorStatus::Aborted;
        ExecutorError::Aborted(reason.to_string())
    }
}
