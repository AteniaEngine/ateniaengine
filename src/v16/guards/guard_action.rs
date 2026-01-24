#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum GuardAction {
    Continue,
    Degrade,
    Abort,
}
