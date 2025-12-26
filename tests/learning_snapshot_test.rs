use atenia_engine::v13::learning_snapshot::BackendKind;
use atenia_engine::v13::self_trainer::{BackendChoice, ExecutionContext, SelfTrainer};

#[test]
fn snapshot_empty_trainer() {
    let trainer = SelfTrainer::new();
    let snap = trainer.snapshot();

    assert_eq!(snap.entries.len(), 0);
    assert_eq!(snap.summary.total_entries, 0);
    assert_eq!(snap.summary.total_episodes, 0);
    assert_eq!(snap.summary.gpu_preference_ratio, 0.0);
}

#[test]
fn snapshot_contains_learned_entry() {
    let mut trainer = SelfTrainer::new();

    let ctx = ExecutionContext {
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

    let ep = atenia_engine::v13::self_trainer::TrainingEpisode { ctx, decision, outcome };
    trainer.record_episode(ep);

    let snap = trainer.snapshot();

    assert_eq!(snap.entries.len(), 1);
    let entry = &snap.entries[0];

    assert_eq!(entry.context.gpu_available, true);
    assert_eq!(entry.context.vram_band, 0);
    assert_eq!(entry.context.ram_band, 0);

    match entry.recommended_backend {
        BackendKind::Gpu => {}
        BackendKind::Cpu => panic!("expected GPU recommendation in snapshot"),
    }

    assert_eq!(entry.stats.episodes, 1);
    assert_eq!(entry.stats.successes, 1);
    assert_eq!(entry.stats.drift_events, 0);
    assert!(entry.stats.average_score > 0.0);
}

#[test]
fn snapshot_is_deterministic_ordered() {
    let mut trainer = SelfTrainer::new();

    // Insert contexts in non-sorted order.
    let ctx1 = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.8,
        ram_pressure: 0.3,
    };
    let ctx2 = ExecutionContext {
        gpu_available: false,
        vram_pressure: 0.1,
        ram_pressure: 0.9,
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

    let outcome = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: true,
        score: 5,
        had_drift: false,
    };

    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: ctx1, decision: decision_gpu, outcome });
    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: ctx2, decision: decision_cpu, outcome });

    let snap1 = trainer.snapshot();
    let snap2 = trainer.snapshot();

    assert_eq!(snap1.entries.len(), 2);
    assert_eq!(snap2.entries.len(), 2);

    // Snapshots must be ordered identically across calls.
    for (a, b) in snap1.entries.iter().zip(snap2.entries.iter()) {
        assert_eq!(a.context.gpu_available, b.context.gpu_available);
        assert_eq!(a.context.vram_band, b.context.vram_band);
        assert_eq!(a.context.ram_band, b.context.ram_band);
    }
}

#[test]
fn summary_fields_are_correct() {
    let mut trainer = SelfTrainer::new();

    let ctx_gpu = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.2,
        ram_pressure: 0.3,
    };
    let ctx_cpu = ExecutionContext {
        gpu_available: false,
        vram_pressure: 0.4,
        ram_pressure: 0.5,
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

    let outcome_gpu = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: true,
        score: 10,
        had_drift: false,
    };
    let outcome_cpu = atenia_engine::v13::self_trainer::EpisodeOutcome {
        success: false,
        score: 2,
        had_drift: true,
    };

    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: ctx_gpu, decision: decision_gpu, outcome: outcome_gpu });
    trainer.record_episode(atenia_engine::v13::self_trainer::TrainingEpisode { ctx: ctx_cpu, decision: decision_cpu, outcome: outcome_cpu });

    let snap = trainer.snapshot();

    assert_eq!(snap.summary.total_entries, snap.entries.len());
    assert_eq!(snap.summary.total_episodes, 2);

    // One entry prefers GPU, one prefers CPU, so ratio should be 0.5.
    assert!((snap.summary.gpu_preference_ratio - 0.5).abs() < 1e-6);
}
