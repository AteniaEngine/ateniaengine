use crate::v13::learning_explanation::DecisionExplanation;
use crate::v13::learning_factors::{DecisionFactorKind, StructuredDecisionExplanation};
use crate::v13::learning_snapshot::{BackendKind, LearningContextSnapshot};

#[derive(Debug, Clone)]
pub struct NarrativeExplanation {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub confidence: f32,
    pub narrative: String,
}

fn qualitative_label(weight: f32) -> &'static str {
    if weight >= 0.75 {
        "high"
    } else if weight >= 0.40 {
        "medium"
    } else {
        "low"
    }
}

pub fn build_narrative(
    text: &DecisionExplanation,
    structured: &StructuredDecisionExplanation,
) -> NarrativeExplanation {
    let context = text.context;
    let recommended_backend = text.recommended_backend;

    // Use the confidence from the textual explanation; assume both views are
    // consistent. Clamp to [0.0, 1.0] defensively.
    let mut confidence = text.confidence;
    if confidence < 0.0 {
        confidence = 0.0;
    } else if confidence > 1.0 {
        confidence = 1.0;
    }
    let confidence_pct = (confidence * 100.0).round() as i32;

    // Paragraph 1: decision summary.
    let backend_str = match recommended_backend {
        BackendKind::Cpu => "CPU",
        BackendKind::Gpu => "GPU",
    };

    let p1 = format!(
        "Atenia selected {} execution with a confidence of {}%. {}",
        backend_str,
        confidence_pct,
        text.explanation.trim(),
    );

    // Extract simple evidence from factors: success rate, drift.
    let mut success_rate: Option<f32> = None;
    let mut has_drift: bool = false;

    for f in &structured.factors {
        match f.kind {
            DecisionFactorKind::HistoricalSuccessRate => {
                // Try to parse episodes and successes from the description in a
                // best-effort, non-panicking way. If parsing fails, fall back
                // to qualitative text only.
                // The description is stable but we avoid depending on its exact
                // formatting: we only reuse it in a high-level way.
                // For the narrative we only need to mention that there is a
                // strong/weak success rate, so we can also infer from weight.
                success_rate = Some(f.weight);
            }
            DecisionFactorKind::DriftPenalty => {
                if f.weight > 0.0 {
                    has_drift = true;
                }
            }
            DecisionFactorKind::ObservationCount => {
                // We do not parse numbers here; we just reflect qualitatively
                // that the observation count is small/medium/large.
                // Episodes are already indirectly represented by the weight.
                // Keep episodes as None and talk about sample size in words.
            }
            DecisionFactorKind::MemoryStability => {}
        }
    }

    let success_phrase = match success_rate {
        Some(w) if w >= 0.75 => "a strong historical success rate",
        Some(w) if w >= 0.40 => "a moderate historical success rate",
        Some(_) => "a limited historical success rate",
        None => "historical performance information",
    };

    let drift_phrase = if has_drift {
        "Some drift or instability has been observed in this context."
    } else {
        "No significant drift or instability has been observed in this context."
    };

    let p2 = format!(
        "The decision is based on {} under similar memory conditions. {}",
        success_phrase,
        drift_phrase,
    );

    // Paragraph 3: factors, in deterministic order.
    fn factor_label(kind: DecisionFactorKind) -> &'static str {
        match kind {
            DecisionFactorKind::HistoricalSuccessRate => "historical success",
            DecisionFactorKind::DriftPenalty => "drift impact",
            DecisionFactorKind::ObservationCount => "observation count",
            DecisionFactorKind::MemoryStability => "memory stability",
        }
    }

    let mut factors_desc: Vec<String> = Vec::new();
    for kind in [
        DecisionFactorKind::HistoricalSuccessRate,
        DecisionFactorKind::DriftPenalty,
        DecisionFactorKind::ObservationCount,
        DecisionFactorKind::MemoryStability,
    ] {
        let mut weight: f32 = 0.0;
        for f in &structured.factors {
            if f.kind == kind {
                weight = f.weight;
                break;
            }
        }
        let label = qualitative_label(weight);
        let name = factor_label(kind);
        factors_desc.push(format!("{} factor has {} influence.", name, label));
    }

    let p3 = format!(
        "Key contributing factors include {}",
        factors_desc.join(" "),
    );

    let narrative = format!("{}\n\n{}\n\n{}", p1, p2, p3);

    NarrativeExplanation {
        context,
        recommended_backend,
        confidence,
        narrative,
    }
}
