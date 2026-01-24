#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ContractError {
    /// No legal execution is possible under the given bias and runtime state.
    NoLegalExecution(String),
    /// The requested intent is incompatible with the observed state.
    IncompatibleIntent(String),
    /// A basic invariant was violated (e.g. invalid ranges).
    InvariantViolation(String),
}
