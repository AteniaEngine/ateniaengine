use crate::v13::checkpoint::drift::DriftReport;
use crate::v13::checkpoint::WarmStartPlan;
use crate::v13::self_trainer::{
    summarize_warm_start_plan, BackendChoice, ExecutionContext, EpisodeOutcome, SelfTrainer,
    TrainingEpisode,
};

#[derive(Debug, Clone, Copy)]
pub enum ExecResult {
    Ok { score: i32 },
    Err { score: i32 },
}

pub fn context_from_pressures(
    gpu_available: bool,
    vram_pressure: f32,
    ram_pressure: f32,
) -> ExecutionContext {
    fn clamp01(x: f32) -> f32 {
        if x.is_nan() {
            0.0
        } else if x < 0.0 {
            0.0
        } else if x > 1.0 {
            1.0
        } else {
            x
        }
    }

    ExecutionContext {
        gpu_available,
        vram_pressure: clamp01(vram_pressure),
        ram_pressure: clamp01(ram_pressure),
    }
}

pub fn had_drift(drifts: &[DriftReport]) -> bool {
    !drifts.is_empty()
}

pub fn outcome_from_exec_result(res: ExecResult, drift: bool) -> EpisodeOutcome {
    match res {
        ExecResult::Ok { score } => EpisodeOutcome {
            success: true,
            score,
            had_drift: drift,
        },
        ExecResult::Err { score } => EpisodeOutcome {
            success: false,
            score,
            had_drift: drift,
        },
    }
}

pub fn record_from_warm_start(
    trainer: &mut SelfTrainer,
    ctx: ExecutionContext,
    plan: &WarmStartPlan,
    drifts: &[DriftReport],
    res: ExecResult,
) {
    let decision = summarize_warm_start_plan(plan);
    let outcome = outcome_from_exec_result(res, had_drift(drifts));
    let episode = TrainingEpisode { ctx, decision, outcome };
    trainer.record_episode(episode);
}

pub fn recommend_for_next_tick(trainer: &SelfTrainer, ctx: ExecutionContext) -> BackendChoice {
    trainer.recommend_backend(ctx)
}
