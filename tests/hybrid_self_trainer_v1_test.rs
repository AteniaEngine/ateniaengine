use atenia_engine::v13::checkpoint::{WarmStartAction, WarmStartDecision, WarmStartPlan};
use atenia_engine::v13::self_trainer::{
    summarize_warm_start_plan, BackendChoice, ExecutionContext, EpisodeOutcome, SelfTrainer,
    TrainingEpisode,
};

fn mock_plan_with_backend(backend: BackendChoice) -> WarmStartPlan {
    let action = match backend {
        BackendChoice::Cpu => WarmStartAction::Keep,
        BackendChoice::Gpu => WarmStartAction::HintPromote { to: atenia_engine::v13::memory_types::MemoryTier::Vram },
    };

    let mut reason = String::from("Base plan");
    if let BackendChoice::Gpu = backend {
        reason = String::from("GPU preferred for this context");
    }

    let decision = WarmStartDecision {
        id: "t1".to_string(),
        is_grad: false,
        current: atenia_engine::v13::memory_types::MemoryTier::Ram,
        desired: None,
        action,
        reason,
    };

    WarmStartPlan {
        decisions: vec![decision],
        summary: String::from("warm_start: keep=1 promote=0 degrade=0"),
    }
}

#[test]
fn cpu_when_gpu_unavailable() {
    let trainer = SelfTrainer::new();

    let ctx = ExecutionContext {
        gpu_available: false,
        vram_pressure: 0.2,
        ram_pressure: 0.3,
    };

    let backend = trainer.recommend_backend(ctx);
    match backend {
        BackendChoice::Cpu => {}
        BackendChoice::Gpu => panic!("expected Cpu when GPU is unavailable"),
    }
}

#[test]
fn learns_gpu_when_successful_and_low_drift() {
    let mut trainer = SelfTrainer::new();

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let gpu_plan = mock_plan_with_backend(BackendChoice::Gpu);
    let gpu_summary = summarize_warm_start_plan(&gpu_plan);

    // Record several successful GPU episodes with high scores.
    for _ in 0..5 {
        let ep = TrainingEpisode {
            ctx,
            decision: gpu_summary,
            outcome: EpisodeOutcome {
                success: true,
                score: 10,
                had_drift: false,
            },
        };
        trainer.record_episode(ep);
    }

    // Record some CPU episodes with lower scores.
    let cpu_plan = mock_plan_with_backend(BackendChoice::Cpu);
    let cpu_summary = summarize_warm_start_plan(&cpu_plan);

    for _ in 0..3 {
        let ep = TrainingEpisode {
            ctx,
            decision: cpu_summary,
            outcome: EpisodeOutcome {
                success: true,
                score: 2,
                had_drift: false,
            },
        };
        trainer.record_episode(ep);
    }

    let backend = trainer.recommend_backend(ctx);
    match backend {
        BackendChoice::Gpu => {}
        BackendChoice::Cpu => panic!("expected Gpu to be preferred when GPU has higher scores"),
    }
}

#[test]
fn penalizes_gpu_when_drift_frequent() {
    let mut trainer = SelfTrainer::new();

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let gpu_plan = mock_plan_with_backend(BackendChoice::Gpu);
    let gpu_summary = summarize_warm_start_plan(&gpu_plan);

    // GPU episodes: moderate scores but frequent drift.
    for _ in 0..10 {
        let ep = TrainingEpisode {
            ctx,
            decision: gpu_summary,
            outcome: EpisodeOutcome {
                success: true,
                score: 5,
                had_drift: true,
            },
        };
        trainer.record_episode(ep);
    }

    // CPU episodes: slightly lower scores but no drift.
    let cpu_plan = mock_plan_with_backend(BackendChoice::Cpu);
    let cpu_summary = summarize_warm_start_plan(&cpu_plan);

    for _ in 0..5 {
        let ep = TrainingEpisode {
            ctx,
            decision: cpu_summary,
            outcome: EpisodeOutcome {
                success: true,
                score: 4,
                had_drift: false,
            },
        };
        trainer.record_episode(ep);
    }

    let backend = trainer.recommend_backend(ctx);
    match backend {
        BackendChoice::Cpu => {}
        BackendChoice::Gpu => panic!("expected Cpu due to high GPU drift penalty"),
    }
}

#[test]
fn stats_update_is_correct() {
    let mut trainer = SelfTrainer::new();

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.4,
        ram_pressure: 0.6,
    };

    let plan = mock_plan_with_backend(BackendChoice::Gpu);
    let summary = summarize_warm_start_plan(&plan);

    let ep = TrainingEpisode {
        ctx,
        decision: summary,
        outcome: EpisodeOutcome {
            success: true,
            score: 7,
            had_drift: true,
        },
    };

    trainer.record_episode(ep);

    let stats_opt = trainer.stats_for(ctx, BackendChoice::Gpu);
    let stats = match stats_opt {
        Some(s) => s,
        None => panic!("expected stats for GPU backend"),
    };

    assert_eq!(stats.count, 1);
    assert_eq!(stats.success_count, 1);
    assert_eq!(stats.score_sum, 7);
    assert_eq!(stats.drift_count, 1);
}
