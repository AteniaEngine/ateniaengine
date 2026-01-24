#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    MissingProfile(String),
    MissingExplanation(String),
    IncompleteExecution(String),
}
