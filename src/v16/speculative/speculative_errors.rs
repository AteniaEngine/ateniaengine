#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum SpeculativeError {
    /// Speculative execution would violate the execution contract.
    ContractViolation(String),
    /// Rollback is not available or failed.
    RollbackUnavailable(String),
    /// The speculative execution failed.
    ExecutionFailed(String),
}
