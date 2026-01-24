#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum GuardError {
    /// Recommended action would violate the execution contract.
    IllegalAction(String),
    /// Guards produced recommendations that are logically inconsistent.
    InconsistentGuards(String),
}
