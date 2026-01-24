#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ComputeError {
    ShapeMismatch(String),
    ContractViolation(String),
    AbortedByGuard(String),
}
