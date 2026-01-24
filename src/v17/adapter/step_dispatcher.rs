#![allow(dead_code)]

use crate::v16::planner::plan_step::PlanStepKind;
use crate::v17::compute::cpu_backend::CpuBackend;
use crate::v17::compute::tensor::Tensor;

use super::adapter_context::AdapterContext;
use super::adapter_errors::AdapterError;

pub struct StepDispatcher;

impl StepDispatcher {
    pub fn dispatch_step(
        backend: &CpuBackend,
        ctx: &mut AdapterContext,
        step_index: usize,
        kind: &PlanStepKind,
        input: &Tensor,
    ) -> Result<(), AdapterError> {
        // Record that this step was visited.
        ctx.executed_steps.push(step_index);

        match kind {
            // Treat this as the primary compute step: run the backend.
            PlanStepKind::MarkTensorsMovable => {
                let out = backend
                    .run_inference(&ctx.model, input, &ctx.contract, ctx.guard_action.clone())
                    .map_err(|e| AdapterError::BackendFailure(format!("backend: {:?}", e)))?;
                ctx.last_output = Some(out);
                Ok(())
            }
            // Known non-compute planning steps are treated as adapter-level no-ops.
            PlanStepKind::EnsureMemoryHeadroom
            | PlanStepKind::SelectBackendCandidate
            | PlanStepKind::PrepareFallback => Ok(()),
        }
    }
}
