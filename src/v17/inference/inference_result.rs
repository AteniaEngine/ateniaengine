#![allow(dead_code)]

use crate::v16::feedback::execution_event::ExecutionEvent;
use crate::v16::feedback::execution_outcome::ExecutionOutcome;
use crate::v17::compute::tensor::Tensor;
use crate::v17::profiling::backend_metrics::ExecutionProfile;

/// Structured result of an end-to-end inference run.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub output: Tensor,
    pub outcome: ExecutionOutcome,
    pub executed_steps: Vec<usize>,
    pub explanation_text: String,
    pub explanation_json: String,
    pub replay_events: Vec<ExecutionEvent>,
    pub replay_outcome: ExecutionOutcome,
    pub profile: Option<ExecutionProfile>,
}
