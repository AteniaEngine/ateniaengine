#![allow(dead_code)]

use super::evidence::snapshot::PolicyEvidenceSnapshot;
use super::types::{DecisionBias, PolicyInput};

pub trait ExecutionPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    /// Base evaluation that does not consider evidence (APX 15.0 behavior).
    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias;

    /// Evidence-aware evaluation (APX 15.1).
    ///
    /// Default implementation falls back to the 15.0 behavior when no
    /// evidence is required, preserving existing policies.
    fn evaluate_with_evidence(
        &self,
        input: &PolicyInput,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> DecisionBias {
        let _ = evidence;
        self.evaluate(input)
    }
}
