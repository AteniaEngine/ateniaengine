#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum AdapterError {
    UnsupportedStep(String),
    ContractViolation(String),
    AbortedByGuard(String),
    BackendFailure(String),
}
