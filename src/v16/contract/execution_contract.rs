#![allow(dead_code)]

use crate::v15::policy::types::DecisionBias;

use super::constraints::{Constraints, RuntimeState};

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionBackend {
    Local,
    Offload,
}

/// Immutable description of what is legally allowed for a future execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionContract {
    /// The intention coming from APX 15 (policy + evidence + preferences).
    pub bias: DecisionBias,
    /// Passive snapshot of the runtime state used to derive this contract.
    pub runtime_snapshot: RuntimeState,
    /// Abstract backends that are explicitly allowed.
    pub allowed_backends: Vec<ExecutionBackend>,
    /// Abstract backends that are explicitly forbidden.
    pub forbidden_backends: Vec<ExecutionBackend>,
    /// Upper bound on allowed aggressiveness in [0.0, 1.0].
    pub max_aggressiveness: f32,
    /// Whether any aggressive choice must have a safe fallback.
    pub require_fallback: bool,
    /// Whether the contract requires stability-oriented behavior.
    pub require_stability: bool,
    /// Explicit constraints derived from bias and runtime state.
    pub constraints: Constraints,
}
