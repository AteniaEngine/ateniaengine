#![allow(dead_code)]

use super::decision_event::DecisionEvent;
use super::reasoning_factors::ReasoningFactors;

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub event: DecisionEvent,
    pub factors: ReasoningFactors,
    pub avoided_alternative: Option<String>,
    pub justification_code: u32,
    pub timestamp: u64,
}

impl DecisionRecord {
    pub fn new(
        event: DecisionEvent,
        factors: ReasoningFactors,
        avoided_alternative: Option<String>,
        justification_code: u32,
        timestamp: u64,
    ) -> Self {
        DecisionRecord {
            event,
            factors,
            avoided_alternative,
            justification_code,
            timestamp,
        }
    }
}
