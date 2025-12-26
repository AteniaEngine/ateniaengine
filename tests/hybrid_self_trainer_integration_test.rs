use atenia_engine::v13::checkpoint::drift::{CheckpointDrift, DriftReport};
use atenia_engine::v13::checkpoint::{WarmStartAction, WarmStartDecision, WarmStartPlan};
use atenia_engine::v13::memory_types::{MemoryTier, TensorId};
use atenia_engine::v13::self_trainer::{BackendChoice, SelfTrainer};
use atenia_engine::v13::self_trainer_integration::{
    context_from_pressures, record_from_warm_start, recommend_for_next_tick, ExecResult,
};

fn make_gpu_promote_plan() -> WarmStartPlan {
    let decision = WarmStartDecision {
        id: "t1".to_string(),
        is_grad: false,
        current: MemoryTier::Ram,
        desired: Some(MemoryTier::Vram),
        action: WarmStartAction::HintPromote { to: MemoryTier::Vram },
        reason: "Desired VRAM and GPU available".to_string(),
    };

    WarmStartPlan {
        decisions: vec![decision],
        summary: "warm_start: keep=0 promote=1 degrade=0".to_string(),
    }
}

#[test]
fn records_episode_from_warm_start_no_drift() {
    let mut trainer = SelfTrainer::new();

    let ctx = context_from_pressures(true, 0.1, 0.2);
    let plan = make_gpu_promote_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    let res = ExecResult::Ok { score: 10 };

    record_from_warm_start(&mut trainer, ctx, &plan, &drifts, res);

    let stats_opt = trainer.stats_for(ctx, BackendChoice::Gpu);
    let stats = match stats_opt {
        Some(s) => s,
        None => panic!("expected stats for GPU backend"),
    };

    assert_eq!(stats.count, 1);
    assert_eq!(stats.success_count, 1);
    assert_eq!(stats.drift_count, 0);
}

#[test]
fn records_episode_with_drift_flag() {
    let mut trainer = SelfTrainer::new();

    let ctx = context_from_pressures(true, 0.1, 0.2);
    let plan = make_gpu_promote_plan();

    let drift = DriftReport {
        entry_id: TensorId("t1".to_string()),
        drifts: vec![CheckpointDrift::TierDowngrade {
            desired: MemoryTier::Vram,
            restored: MemoryTier::Ram,
        }],
    };

    let drifts = vec![drift];

    let res = ExecResult::Ok { score: 8 };

    record_from_warm_start(&mut trainer, ctx, &plan, &drifts, res);

    let stats_opt = trainer.stats_for(ctx, BackendChoice::Gpu);
    let stats = match stats_opt {
        Some(s) => s,
        None => panic!("expected stats for GPU backend"),
    };

    assert_eq!(stats.count, 1);
    assert_eq!(stats.success_count, 1);
    assert_eq!(stats.drift_count, 1);
}

#[test]
fn recommend_changes_after_recording() {
    let mut trainer = SelfTrainer::new();

    let ctx = context_from_pressures(true, 0.1, 0.2);
    let plan = make_gpu_promote_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    // Record multiple GPU-success episodes with higher scores.
    for _ in 0..5 {
        let res = ExecResult::Ok { score: 10 };
        record_from_warm_start(&mut trainer, ctx, &plan, &drifts, res);
    }

    // Record some CPU episodes with lower scores by constructing
    // a CPU-leaning warm start plan.
    let cpu_decision = WarmStartDecision {
        id: "t2".to_string(),
        is_grad: false,
        current: MemoryTier::Ram,
        desired: None,
        action: WarmStartAction::Keep,
        reason: "CPU preferred for this context".to_string(),
    };

    let cpu_plan = WarmStartPlan {
        decisions: vec![cpu_decision],
        summary: "warm_start: keep=1 promote=0 degrade=0".to_string(),
    };

    for _ in 0..3 {
        let res = ExecResult::Ok { score: 2 };
        record_from_warm_start(&mut trainer, ctx, &cpu_plan, &drifts, res);
    }

    let backend = recommend_for_next_tick(&trainer, ctx);

    match backend {
        BackendChoice::Gpu => {}
        BackendChoice::Cpu => panic!("expected GPU to be preferred after recording high-score GPU episodes"),
    }
}

#[test]
fn clamp_pressures_is_safe() {
    let ctx = context_from_pressures(true, 2.0, -1.0);

    assert!(ctx.vram_pressure <= 1.0 && ctx.vram_pressure >= 0.0);
    assert!(ctx.ram_pressure <= 1.0 && ctx.ram_pressure >= 0.0);
}
