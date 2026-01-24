#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ExplainError {
    MissingInformation(String),
    InconsistentEvents(String),
}
