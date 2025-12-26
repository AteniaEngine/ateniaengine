use crate::v13::learning_snapshot::{BackendKind, LearningContextSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFactorKind {
    HistoricalSuccessRate,
    DriftPenalty,
    ObservationCount,
    MemoryStability,
}

#[derive(Debug, Clone)]
pub struct DecisionFactor {
    pub kind: DecisionFactorKind,
    pub weight: f32,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct StructuredDecisionExplanation {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub confidence: f32,
    pub factors: Vec<DecisionFactor>,
}

fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

impl DecisionFactor {
    pub fn new(kind: DecisionFactorKind, weight: f32, description: String) -> Self {
        DecisionFactor {
            kind,
            weight: clamp01(weight),
            description,
        }
    }
}

impl StructuredDecisionExplanation {
    pub fn clamp_confidence(raw: f32) -> f32 {
        clamp01(raw)
    }
}
