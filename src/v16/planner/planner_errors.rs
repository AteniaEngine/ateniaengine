#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum PlannerError {
    /// The contract cannot be planned because constraints are incompatible.
    UnplannableContract(String),
    /// The contract is missing essential information for planning.
    InvalidContract(String),
}
