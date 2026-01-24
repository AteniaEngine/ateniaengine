#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    MissingInformation(String),
    InconsistentHistory(String),
    DivergentOutcome(String),
}
