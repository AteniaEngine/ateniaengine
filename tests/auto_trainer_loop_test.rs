use atenia_engine::v13::auto_trainer_loop::{AutoTrainerConfig, AutoTrainerLoop};
use atenia_engine::v13::checkpoint::{WarmStartAction, WarmStartDecision, WarmStartPlan};
use atenia_engine::v13::checkpoint::drift::DriftReport;
use atenia_engine::v13::memory_types::MemoryTier;
use atenia_engine::v13::self_trainer::{BackendChoice, ExecutionContext};
use atenia_engine::v13::self_trainer_integration::ExecResult;

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

fn make_cpu_plan() -> WarmStartPlan {
    let decision = WarmStartDecision {
        id: "t2".to_string(),
        is_grad: false,
        current: MemoryTier::Ram,
        desired: None,
        action: WarmStartAction::Keep,
        reason: "CPU preferred for this context".to_string(),
    };

    WarmStartPlan {
        decisions: vec![decision],
        summary: "warm_start: keep=1 promote=0 degrade=0".to_string(),
    }
}

#[test]
fn cooldown_prevents_thrashing() {
    let cfg = AutoTrainerConfig {
        cooldown_ticks: 3,
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

    let gpu_plan = make_gpu_plan();
    let cpu_plan = make_cpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    // tick 0: GPU strongly better
    let choice0 = loop_state.on_tick(ctx, &gpu_plan, &drifts, ExecResult::Ok { score: 10 });
    assert_eq!(choice0, BackendChoice::Gpu);

    // tick 1: CPU better, but cooldown active -> should hold GPU
    let choice1 = loop_state.on_tick(ctx, &cpu_plan, &drifts, ExecResult::Ok { score: 10 });
    assert_eq!(choice1, BackendChoice::Gpu);
    assert!(loop_state.last_debug_reason().contains("cooldown"));

    // tick 2: CPU still better, cooldown still active -> hold GPU
    let choice2 = loop_state.on_tick(ctx, &cpu_plan, &drifts, ExecResult::Ok { score: 10 });
    assert_eq!(choice2, BackendChoice::Gpu);
    assert!(loop_state.last_debug_reason().contains("cooldown"));

    // tick 3: cooldown expired, allow switch to CPU if trainer prefers it
    let choice3 = loop_state.on_tick(ctx, &cpu_plan, &drifts, ExecResult::Ok { score: 10 });
    assert_eq!(choice3, BackendChoice::Cpu);
}

#[test]
fn always_cpu_when_gpu_unavailable() {
    let cfg = AutoTrainerConfig {
        cooldown_ticks: 3,
        drift_penalty: 0,
        autosave_every_ticks: None,
        autosave_every_seconds: None,
        autosave_path: None,
    };
    let mut loop_state = AutoTrainerLoop::new(cfg);

    let ctx = ExecutionContext {
        gpu_available: false,
        vram_pressure: 0.1,
        ram_pressure: 0.2,
    };

    let gpu_plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    for _ in 0..5 {
        let choice = loop_state.on_tick(ctx, &gpu_plan, &drifts, ExecResult::Ok { score: 10 });
        assert_eq!(choice, BackendChoice::Cpu);
        assert!(loop_state.last_debug_reason().contains("gpu unavailable"));
    }
}

#[test]
fn records_episodes_every_tick() {
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
        vram_pressure: 0.3,
        ram_pressure: 0.4,
    };

    let gpu_plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    for _ in 0..5 {
        let _ = loop_state.on_tick(ctx, &gpu_plan, &drifts, ExecResult::Ok { score: 5 });
    }

    // We do not assert on a specific backend; just check that
    // some stats have at least 5 episodes recorded for this context.
    let cpu_stats = loop_state.inner_trainer().stats_for(ctx, BackendChoice::Cpu);
    let gpu_stats = loop_state.inner_trainer().stats_for(ctx, BackendChoice::Gpu);

    let total = cpu_stats.map(|s| s.count).unwrap_or(0) + gpu_stats.map(|s| s.count).unwrap_or(0);
    assert!(total >= 5);
}

#[test]
fn deterministic_given_same_inputs() {
    let ctx = ExecutionContext {
        gpu_available: true,
        vram_pressure: 0.2,
        ram_pressure: 0.3,
    };

    let gpu_plan = make_gpu_plan();
    let drifts: Vec<DriftReport> = Vec::new();

    // First run
    let cfg_a = AutoTrainerConfig {
        cooldown_ticks: 2,
        drift_penalty: 0,
        autosave_every_ticks: None,
        autosave_every_seconds: None,
        autosave_path: None,
    };
    let mut loop_a = AutoTrainerLoop::new(cfg_a);
    let mut choices_a = Vec::new();
    for i in 0..6 {
        let plan = if i % 2 == 0 { &gpu_plan } else { &gpu_plan };
        let choice = loop_a.on_tick(ctx, plan, &drifts, ExecResult::Ok { score: 5 });
        choices_a.push(choice);
    }

    // Second run with the same inputs
    let cfg_b = AutoTrainerConfig {
        cooldown_ticks: 2,
        drift_penalty: 0,
        autosave_every_ticks: None,
        autosave_every_seconds: None,
        autosave_path: None,
    };
    let mut loop_b = AutoTrainerLoop::new(cfg_b);
    let mut choices_b = Vec::new();
    for i in 0..6 {
        let plan = if i % 2 == 0 { &gpu_plan } else { &gpu_plan };
        let choice = loop_b.on_tick(ctx, plan, &drifts, ExecResult::Ok { score: 5 });
        choices_b.push(choice);
    }

    assert_eq!(choices_a.len(), choices_b.len());
    for (a, b) in choices_a.iter().zip(choices_b.iter()) {
        assert_eq!(*a, *b);
    }
}
