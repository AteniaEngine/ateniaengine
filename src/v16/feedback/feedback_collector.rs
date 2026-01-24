#![allow(dead_code)]

use crate::v16::feedback::execution_event::ExecutionEvent;
use crate::v16::feedback::execution_outcome::ExecutionOutcome;

#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackSnapshot {
    pub events: Vec<ExecutionEvent>,
    pub outcome: ExecutionOutcome,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct FeedbackCollector {
    events: Vec<ExecutionEvent>,
    outcome: Option<ExecutionOutcome>,
}

impl FeedbackCollector {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            outcome: None,
        }
    }

    pub fn record(&mut self, events: Vec<ExecutionEvent>, outcome: ExecutionOutcome) {
        self.events = events;
        self.outcome = Some(outcome);
    }

    pub fn snapshot(&self) -> Option<FeedbackSnapshot> {
        self.outcome.as_ref().map(|o| FeedbackSnapshot {
            events: self.events.clone(),
            outcome: o.clone(),
        })
    }
}
