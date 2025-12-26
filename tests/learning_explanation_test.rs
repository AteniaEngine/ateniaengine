use atenia_engine::v13::learning_snapshot::{BackendKind, LearningContextSnapshot};
use atenia_engine::v13::self_trainer::{BackendChoice, ExecutionContext, SelfTrainer};

#[test]
fn explanation_none_when_context_unknown() {
    let trainer = SelfTrainer::new();

    let ctx = LearningContextSnapshot {
        gpu_available: true,
        vram_band: 1,
        ram_band: 1,
    };

    let explanation = trainer.explain_decision(ctx);
    assert!(explanation.is_none());
}

#[test]
fn explanation_for_gpu_context() {
    let mut trainer = SelfTrainer::new();

    let exec_ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.2,
        ram_pressure: 0.3,
    };

    let decision = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Gpu,
        promote_count: 1,
        degrade_count: 0,
        keep_count: 0,
    };

    let outcome = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: true,
        score: 10,
        had_drift: false,
    };

    let episode = atenia_engine::v13::self_trainer::TrainingEpisode {
        ctx: exec_ctx,
        decision,
        outcome,
    };

    trainer.record_episode(episode);
    let snap = trainer.snapshot();
    assert!(snap.entries.len() >= 1);
    let ctx = snap.entries[0].context;

    let explanation = trainer
        .explain_decision(ctx)
        .expect("expected explanation for known context");

    match explanation.recommended_backend {
        BackendKind::Gpu => {}
        BackendKind::Cpu => panic!("expected GPU recommendation in explanation"),
    }

    assert!(explanation.confidence > 0.5);
    assert!(explanation.explanation.contains("GPU"));
}

#[test]
fn explanation_for_cpu_context_due_to_drift() {
    let mut trainer = SelfTrainer::new();

    let exec_ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.5,
        ram_pressure: 0.4,
    };

    let decision_gpu = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Gpu,
        promote_count: 1,
        degrade_count: 0,
        keep_count: 0,
    };

    let decision_cpu = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Cpu,
        promote_count: 0,
        degrade_count: 0,
        keep_count: 1,
    };

    // Record a GPU episode with drift and low success.
    let outcome_bad_gpu = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: false,
        score: 1,
        had_drift: true,
    };

    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode {
        ctx: exec_ctx,
        decision: decision_gpu,
        outcome: outcome_bad_gpu,
    });

    // Record a CPU episode without drift and with success to tilt preference.
    let outcome_good_cpu = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: true,
        score: 5,
        had_drift: false,
    };

    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode {
        ctx: exec_ctx,
        decision: decision_cpu,
        outcome: outcome_good_cpu,
    });

    let snap = trainer.snapshot();
    assert!(snap.entries.len() >= 1);
    let ctx = snap.entries[0].context;

    let explanation = trainer
        .explain_decision(ctx)
        .expect("expected explanation for context with drift history");

    match explanation.recommended_backend {
        BackendKind::Cpu => {}
        BackendKind::Gpu => panic!("expected CPU recommendation due to drift"),
    }

    assert!(explanation.explanation.contains("CPU"));
    assert!(
        explanation
            .explanation
            .to_lowercase()
            .contains("drift")
            || explanation
                .explanation
                .to_lowercase()
                .contains("instability"),
    );
}

#[test]
fn explanation_is_deterministic() {
    let mut trainer = SelfTrainer::new();

    let exec_ctx = ExecutionContext {
        gpu_available: false,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let decision = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Cpu,
        promote_count: 0,
        degrade_count: 0,
        keep_count: 1,
    };

    let outcome = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: true,
        score: 3,
        had_drift: false,
    };

    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode {
        ctx: exec_ctx,
        decision,
        outcome,
    });

    let ctx = LearningContextSnapshot {
        gpu_available: false,
        vram_band: 0,
        ram_band: 0,
    };

    let e1 = trainer
        .explain_decision(ctx)
        .expect("expected explanation for known context");
    let e2 = trainer
        .explain_decision(ctx)
        .expect("expected explanation for known context");

    assert_eq!(e1.recommended_backend as u8, e2.recommended_backend as u8);
    assert!((e1.confidence - e2.confidence).abs() < 1e-6);
    assert_eq!(e1.explanation, e2.explanation);
}
