#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum PlanStepKind {
    EnsureMemoryHeadroom,
    SelectBackendCandidate,
    PrepareFallback,
    MarkTensorsMovable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanStep {
    pub kind: PlanStepKind,
    /// Human-readable description of what this step intends to check or prepare.
    pub description: String,
    /// Preconditions that must conceptually hold before this step would run.
    pub preconditions: Vec<String>,
    /// Expected postconditions after this step would succeed.
    pub postconditions: Vec<String>,
    /// Whether this step can be safely aborted before or during execution.
    pub abortable: bool,
    /// Whether an explicit verification step is required before considering this step safe.
    pub requires_verification: bool,
}
