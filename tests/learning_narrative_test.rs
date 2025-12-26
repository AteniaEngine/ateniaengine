use atenia_engine::v13::learning_explanation::DecisionExplanation;
use atenia_engine::v13::learning_factors::{DecisionFactor, DecisionFactorKind, StructuredDecisionExplanation};
use atenia_engine::v13::learning_narrative::build_narrative;
use atenia_engine::v13::learning_snapshot::{BackendKind, LearningContextSnapshot};

fn make_sample_decision_explanation(backend: BackendKind, confidence: f32) -> DecisionExplanation {
    DecisionExplanation {
        context: LearningContextSnapshot {
            gpu_available: true,
            vram_band: 0,
            ram_band: 0,
        },
        recommended_backend: backend,
        confidence,
        explanation: "Base explanation for this context.".to_string(),
    }
}

fn make_sample_structured_explanation(drift_weight: f32) -> StructuredDecisionExplanation {
    let ctx = LearningContextSnapshot {
        gpu_available: true,
        vram_band: 0,
        ram_band: 0,
    };

    let factors = vec![
        DecisionFactor {
            kind: DecisionFactorKind::HistoricalSuccessRate,
            weight: 0.8,
            description: "Strong historical success rate.".to_string(),
        },
        DecisionFactor {
            kind: DecisionFactorKind::DriftPenalty,
            weight: drift_weight,
            description: if drift_weight > 0.0 {
                "Drift observed in several episodes, indicating instability.".to_string()
            } else {
                "No drift observed.".to_string()
            },
        },
        DecisionFactor {
            kind: DecisionFactorKind::ObservationCount,
            weight: 0.6,
            description: "Observed more than 30 episodes.".to_string(),
        },
        DecisionFactor {
            kind: DecisionFactorKind::MemoryStability,
            weight: 1.0 - drift_weight,
            description: "Memory behavior has been stable for most executions.".to_string(),
        },
    ];

    StructuredDecisionExplanation {
        context: ctx,
        recommended_backend: BackendKind::Gpu,
        confidence: 0.8,
        factors,
    }
}

#[test]
fn narrative_contains_backend_and_confidence() {
    let text = make_sample_decision_explanation(BackendKind::Gpu, 0.82);
    let structured = make_sample_structured_explanation(0.1);

    let narrative = build_narrative(&text, &structured);

    assert!(narrative.narrative.contains("GPU"));
    assert!(narrative.narrative.contains("%"));
}

#[test]
fn narrative_mentions_drift_when_present() {
    let text = make_sample_decision_explanation(BackendKind::Gpu, 0.7);
    let structured = make_sample_structured_explanation(0.5);

    let narrative = build_narrative(&text, &structured);

    let lower = narrative.narrative.to_lowercase();
    assert!(lower.contains("drift") || lower.contains("instability"));
}

#[test]
fn narrative_mentions_all_factors() {
    let text = make_sample_decision_explanation(BackendKind::Cpu, 0.6);
    let structured = make_sample_structured_explanation(0.2);

    let narrative = build_narrative(&text, &structured);
    let lower = narrative.narrative.to_lowercase();

    assert!(lower.contains("historical success"));
    assert!(lower.contains("drift impact"));
    assert!(lower.contains("observation count"));
    assert!(lower.contains("memory stability"));
}

#[test]
fn narrative_is_deterministic() {
    let text = make_sample_decision_explanation(BackendKind::Cpu, 0.63);
    let structured = make_sample_structured_explanation(0.3);

    let n1 = build_narrative(&text, &structured);
    let n2 = build_narrative(&text, &structured);

    assert_eq!(n1.recommended_backend as u8, n2.recommended_backend as u8);
    assert!((n1.confidence - n2.confidence).abs() < 1e-6);
    assert_eq!(n1.narrative, n2.narrative);
}
