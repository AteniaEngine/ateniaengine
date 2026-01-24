#![allow(dead_code)]

use crate::v15::policy::evidence::signals::PolicySignal;

/// Immutable, clonable snapshot of passive evidence signals.
///
/// In APX 15.1 this is built from synthetic or pre-aggregated data.
/// There are no references to live runtime objects.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyEvidenceSnapshot {
    pub signals: Vec<PolicySignal>,
}

impl PolicyEvidenceSnapshot {
    pub fn new(signals: Vec<PolicySignal>) -> Self {
        Self { signals }
    }

    pub fn all_signals(&self) -> &[PolicySignal] {
        &self.signals
    }

    pub fn is_normalized(&self) -> bool {
        self.signals.iter().all(|s| s.is_normalized())
    }
}
