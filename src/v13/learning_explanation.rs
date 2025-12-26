use crate::v13::learning_snapshot::{BackendKind, LearningContextSnapshot};

#[derive(Debug, Clone)]
pub struct DecisionExplanation {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub confidence: f32,
    pub explanation: String,
}

impl DecisionExplanation {
    pub fn clamp_confidence(raw: f32) -> f32 {
        if raw < 0.0 {
            0.0
        } else if raw > 1.0 {
            1.0
        } else {
            raw
        }
    }
}
