#![allow(dead_code)]

/// Per-step profiling data collected during execution.
#[derive(Debug, Clone, PartialEq)]
pub struct StepMetrics {
    pub step_index: usize,
    pub kind: String,
    pub backend: String,
    pub input_elements: usize,
    pub output_elements: usize,
    pub aborted: bool,
    pub fallback: bool,
}
