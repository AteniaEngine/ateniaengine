#![allow(dead_code)]

use crate::v16::executor::execution_context::{ExecutionContext, RuntimeFacade};
use crate::v16::executor::step_executor::StepExecutor;
use crate::v16::planner::execution_plan::ExecutionPlan;

use super::rollback_manager::RollbackManager;
use super::speculative_errors::SpeculativeError;
use super::speculative_plan::SpeculativePlan;

/// Executor for speculative plans. It executes steps using a dedicated
/// ExecutionContext and relies on RollbackManager to restore the runtime
/// facade on failure.
#[derive(Debug)]
pub struct SpeculativeExecutor<R: RuntimeFacade + Clone> {
    pub plan: SpeculativePlan,
    pub context: ExecutionContext<R>,
    rollback: RollbackManager<R>,
}

impl<R: RuntimeFacade + Clone> SpeculativeExecutor<R> {
    pub fn new(base_plan: &ExecutionPlan, context: ExecutionContext<R>) -> Self {
        let rollback = RollbackManager::new(&context.runtime);
        Self {
            plan: SpeculativePlan::from_base(base_plan),
            context,
            rollback,
        }
    }

    /// Run the speculative execution. On any failure, rollback is applied and
    /// an error is returned. On success, the speculative context may have
    /// observed effects, but the original runtime can always be restored using
    /// the rollback manager.
    pub fn run(&mut self) -> Result<(), SpeculativeError> {
        if !self.plan.base_plan.globally_abortable {
            return Err(SpeculativeError::ContractViolation(
                "Base plan is not globally abortable".to_string(),
            ));
        }

        for step in &self.plan.base_plan.steps {
            let result = StepExecutor::execute_step(step, &mut self.context);
            if let Err(e) = result {
                // On any failure, rollback and surface an error.
                self.rollback.rollback(&mut self.context.runtime);
                return Err(SpeculativeError::ExecutionFailed(format!(
                    "speculative step failed: {:?}",
                    e
                )));
            }
        }

        Ok(())
    }
}
