#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackError {
    InconsistentEvents(String),
    InvalidEvent(String),
    LogicalOrderViolation(String),
}
