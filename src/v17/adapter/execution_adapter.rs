#![allow(dead_code)]

use crate::v16::planner::execution_plan::ExecutionPlan;

use crate::v17::compute::cpu_backend::CpuBackend;
use crate::v17::compute::tensor::Tensor;

use super::adapter_context::AdapterContext;
use super::adapter_errors::AdapterError;
use super::step_dispatcher::StepDispatcher;

/// Bridges an `ExecutionPlan` from APX 16.1 to the v17 CPU backend.
pub struct ExecutionAdapter {
    backend: CpuBackend,
}

impl ExecutionAdapter {
    pub fn new(backend: CpuBackend) -> Self {
        Self { backend }
    }

    /// Execute all steps in the plan in order, using the provided adapter
    /// context and input tensor. The adapter does not modify the plan or
    /// contract; it only follows the prescribed steps.
    pub fn execute_plan(
        &self,
        plan: &ExecutionPlan,
        ctx: &mut AdapterContext,
        input: &Tensor,
    ) -> Result<Tensor, AdapterError> {
        for (idx, step) in plan.steps.iter().enumerate() {
            if ctx.aborted {
                break;
            }

            // Guards may have requested abort; surface this as an adapter error.
            if let crate::v16::guards::guard_action::GuardAction::Abort = ctx.guard_action {
                ctx.aborted = true;
                return Err(AdapterError::AbortedByGuard(
                    "execution aborted by guard before step".to_string(),
                ));
            }

            StepDispatcher::dispatch_step(&self.backend, ctx, idx, &step.kind, input)?;
        }

        ctx.last_output
            .clone()
            .ok_or_else(|| AdapterError::BackendFailure("no output produced by plan".to_string()))
    }
}
