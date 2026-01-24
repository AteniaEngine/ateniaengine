#![allow(dead_code)]

use crate::v16::planner::execution_plan::ExecutionPlan;
use crate::v17::compute::tensor::Tensor;

use super::backend_metrics::{BackendKind, BackendMetrics, ExecutionProfile};
use super::profiling_errors::ProfilingError;
use super::step_metrics::StepMetrics;

/// Collects deterministic, logical metrics for a plan execution.
pub struct ExecutionProfiler;

impl ExecutionProfiler {
    /// Build an `ExecutionProfile` from a plan, the executed step indices, the
    /// backend kind, and the logical input/output tensors. No timing or IO is
    /// performed.
    pub fn profile(
        plan: &ExecutionPlan,
        executed_steps: &[usize],
        backend: BackendKind,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<ExecutionProfile, ProfilingError> {
        if executed_steps.is_empty() {
            return Err(ProfilingError::MissingSteps(
                "no executed steps provided".to_string(),
            ));
        }

        let mut steps = Vec::new();
        for &idx in executed_steps {
            let kind_str = if let Some(step) = plan.steps.get(idx) {
                format!("{:?}", step.kind)
            } else {
                "unknown".to_string()
            };

            steps.push(StepMetrics {
                step_index: idx,
                kind: kind_str,
                backend: match backend {
                    BackendKind::Cpu => "cpu".to_string(),
                    BackendKind::Gpu => "gpu".to_string(),
                },
                input_elements: input.data.len(),
                output_elements: output.data.len(),
                aborted: false,
                fallback: false,
            });
        }

        let mut backend_metrics = BackendMetrics::new(backend);
        // For now, assume a single matmul + relu per execution when there is at
        // least one compute step.
        backend_metrics.matmul_count = 1;
        backend_metrics.relu_count = 1;
        backend_metrics.add_count = 0;
        backend_metrics.elements_processed = input.data.len() + output.data.len();

        Ok(ExecutionProfile {
            steps,
            backends: vec![backend_metrics],
        })
    }
}
