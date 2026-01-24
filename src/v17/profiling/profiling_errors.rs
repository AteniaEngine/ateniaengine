#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ProfilingError {
    MissingSteps(String),
}
