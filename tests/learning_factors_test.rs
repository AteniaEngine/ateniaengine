use atenia_engine::v13::learning_factors::DecisionFactorKind;
use atenia_engine::v13::learning_snapshot::LearningContextSnapshot;
use atenia_engine::v13::self_trainer::{BackendChoice, ExecutionContext, SelfTrainer};

#[test]
fn structured_none_when_context_unknown() {
    let trainer = SelfTrainer::new();

    let ctx = LearningContextSnapshot {
        gpu_available: true,
        vram_band: 0,
        ram_band: 0,
    };

    let explanation = trainer.explain_decision_structured(ctx);
    assert!(explanation.is_none());
}

#[test]
fn structured_contains_all_factors() {
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

    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: exec_ctx, decision, outcome });

    let snap = trainer.snapshot();
    assert!(snap.entries.len() >= 1);
    let ctx = snap.entries[0].context;

    let explanation = trainer
        .explain_decision_structured(ctx)
        .expect("expected structured explanation for known context");

    assert_eq!(explanation.factors.len(), 4);
    assert_eq!(explanation.factors[0].kind, DecisionFactorKind::HistoricalSuccessRate);
    assert_eq!(explanation.factors[1].kind, DecisionFactorKind::DriftPenalty);
    assert_eq!(explanation.factors[2].kind, DecisionFactorKind::ObservationCount);
    assert_eq!(explanation.factors[3].kind, DecisionFactorKind::MemoryStability);
}

#[test]
fn weights_are_clamped_and_safe() {
    let mut trainer = SelfTrainer::new();

    let exec_ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.9,
        ram_pressure: 0.9,
    };

    let decision = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Gpu,
        promote_count: 1,
        degrade_count: 0,
        keep_count: 0,
    };

    let outcome = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: false,
        score: 1,
        had_drift: true,
    };

    // Record many episodes to drive counts high.
    for _ in 0..100 {
        trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: exec_ctx, decision, outcome });
    }

    let snap = trainer.snapshot();
    assert!(snap.entries.len() >= 1);
    let ctx = snap.entries[0].context;

    let explanation = trainer
        .explain_decision_structured(ctx)
        .expect("expected structured explanation for known context");

    for f in &explanation.factors {
        assert!(f.weight >= 0.0);
        assert!(f.weight <= 1.0);
    }
}

#[test]
fn gpu_structured_explanation_has_high_success_weight() {
    let mut trainer = SelfTrainer::new();

    let exec_ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.2,
        ram_pressure: 0.2,
    };

    let decision = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Gpu,
        promote_count: 1,
        degrade_count: 0,
        keep_count: 0,
    };

    let good_outcome = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: true,
        score: 10,
        had_drift: false,
    };

    for _ in 0..40 {
        trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: exec_ctx, decision, outcome: good_outcome });
    }

    let snap = trainer.snapshot();
    assert!(snap.entries.len() >= 1);
    let ctx = snap.entries[0].context;

    let explanation = trainer
        .explain_decision_structured(ctx)
        .expect("expected structured explanation for known context");

    let success_factor = &explanation.factors[0];
    assert!(success_factor.weight > 0.6);
}

#[test]
fn drift_penalty_present_for_unstable_context() {
    let mut trainer = SelfTrainer::new();

    let exec_ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.5,
        ram_pressure: 0.4,
    };

    let decision = atenia_engine::v13::self_trainer::DecisionSummary {
        backend: BackendChoice::Gpu,
        promote_count: 1,
        degrade_count: 0,
        keep_count: 0,
    };

    let outcome_unstable = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: false,
        score: 1,
        had_drift: true,
    };

    for _ in 0..10 {
        trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: exec_ctx, decision, outcome: outcome_unstable });
    }

    let snap = trainer.snapshot();
    assert!(snap.entries.len() >= 1);
    let ctx = snap.entries[0].context;

    let explanation = trainer
        .explain_decision_structured(ctx)
        .expect("expected structured explanation for context with drift history");

    let drift_factor = &explanation.factors[1];
    assert!(drift_factor.weight > 0.0);
    let desc = drift_factor.description.to_lowercase();
    assert!(desc.contains("instability") || desc.contains("drift"));
}
