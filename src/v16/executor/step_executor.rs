#![allow(dead_code)]

use super::execution_context::RuntimeFacade;
use super::executor_errors::ExecutorError;
use crate::v16::planner::plan_step::PlanStep;
use crate::v16::planner::plan_step::PlanStepKind;

pub struct StepExecutor;

impl StepExecutor {
    pub fn execute_step<R: RuntimeFacade>(
        step: &PlanStep,
        ctx: &mut crate::v16::executor::execution_context::ExecutionContext<R>,
    ) -> Result<(), ExecutorError> {
        // In APX 16.2 we delegate the actual effects to the RuntimeFacade
        // implementation and only map errors into typed ExecutorError values.
        match step.kind {
            PlanStepKind::EnsureMemoryHeadroom => ctx
                .runtime
                .ensure_memory_headroom()
                .map_err(ExecutorError::StepFailed),
            PlanStepKind::SelectBackendCandidate => ctx
                .runtime
                .select_backend_candidate()
                .map_err(ExecutorError::StepFailed),
            PlanStepKind::PrepareFallback => ctx
                .runtime
                .prepare_fallback()
                .map_err(ExecutorError::StepFailed),
            PlanStepKind::MarkTensorsMovable => ctx
                .runtime
                .mark_tensors_movable()
                .map_err(ExecutorError::StepFailed),
        }
    }
}
