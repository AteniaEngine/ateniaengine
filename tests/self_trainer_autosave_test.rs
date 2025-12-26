use std::fs;
use std::path::PathBuf;

use atenia_engine::v13::auto_trainer_loop::{AutoTrainerConfig, AutoTrainerLoop};
use atenia_engine::v13::checkpoint::{WarmStartAction, WarmStartDecision, WarmStartPlan};
use atenia_engine::v13::checkpoint::drift::DriftReport;
use atenia_engine::v13::memory_types::MemoryTier;
use atenia_engine::v13::self_trainer::{BackendChoice, ExecutionContext};
use atenia_engine::v13::self_trainer_integration::ExecResult;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("atenia_autosave_test_{}", name));
    p
}

fn make_gpu_plan() -> WarmStartPlan {
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
fn autosave_not_triggered_when_disabled() {
    let cfg = AutoTrainerConfig {
        cooldown_ticks: 0,
        drift_penalty: 0,
        autosave_every_ticks: None,
        autosave_every_seconds: None,
        autosave_path: None,
    };
    let mut loop_state = AutoTrainerLoop::new(cfg);

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    let path = temp_path("disabled");
    let _ = fs::remove_file(&path);

    for _ in 0..10 {
        let _ = loop_state.on_tick(ctx, &plan, &drifts, ExecResult::Ok { score: 1 });
    }

    assert!(!path.exists());
}

#[test]
fn autosave_triggers_by_tick_interval() {
    let path = temp_path("tick_interval");
    let _ = fs::remove_file(&path);

    let cfg = AutoTrainerConfig {
        cooldown_ticks: 0,
        drift_penalty: 0,
        autosave_every_ticks: Some(2),
        autosave_every_seconds: None,
        autosave_path: Some(path.clone()),
    };
    let mut loop_state = AutoTrainerLoop::new(cfg);

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    // tick 0
    let _ = loop_state.on_tick(ctx, &plan, &drifts, ExecResult::Ok { score: 1 });
    assert!(!path.exists());

    // tick 1
    let _ = loop_state.on_tick(ctx, &plan, &drifts, ExecResult::Ok { score: 1 });
    assert!(path.exists());

    let _ = fs::remove_file(&path);
}

#[test]
fn autosave_respects_dirty_flag() {
    let path = temp_path("dirty_flag");
    let _ = fs::remove_file(&path);

    let cfg = AutoTrainerConfig {
        cooldown_ticks: 0,
        drift_penalty: 0,
        autosave_every_ticks: Some(1),
        autosave_every_seconds: None,
        autosave_path: Some(path.clone()),
    };
    let mut loop_state = AutoTrainerLoop::new(cfg);

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    // First tick should trigger autosave once.
    let _ = loop_state.on_tick(ctx, &plan, &drifts, ExecResult::Ok { score: 1 });
    assert!(path.exists());

    // Clear dirty flag by explicitly marking save as done.
    // try_autosave should now be false because nothing new was recorded.
    let result = loop_state.try_autosave();
    assert!(matches!(result, Ok(false)));

    let _ = fs::remove_file(&path);
}

#[test]
fn autosave_failure_does_not_break_tick() {
    // Use a path that is likely to be invalid or unwritable (directory as file).
    let invalid_path = PathBuf::from("./.atenia_self_trainer_persist_test");
    let _ = fs::create_dir_all(&invalid_path);

    let cfg = AutoTrainerConfig {
        cooldown_ticks: 0,
        drift_penalty: 0,
        autosave_every_ticks: Some(1),
        autosave_every_seconds: None,
        autosave_path: Some(invalid_path.clone()),
    };
    let mut loop_state = AutoTrainerLoop::new(cfg);

    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    let choice = loop_state.on_tick(ctx, &plan, &drifts, ExecResult::Ok { score: 1 });

    // Even if autosave fails, we must still get a valid backend choice.
    match choice {
        BackendChoice::Cpu | BackendChoice::Gpu => {}
    }

    assert!(loop_state.last_debug_reason().contains("autosave: failed"));
}
