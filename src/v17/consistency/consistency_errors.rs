#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyError {
    InvalidBaseline(String),
    IncompatibleSnapshots(String),
}
