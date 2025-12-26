use crate::v13::checkpoint::drift::DriftReport;
use crate::v13::checkpoint::WarmStartPlan;
use crate::v13::self_trainer::{
    BackendChoice, ExecutionContext, SelfTrainer,
};
use crate::v13::self_trainer_integration::{
    record_from_warm_start, recommend_for_next_tick, ExecResult,
};
use crate::v13::self_trainer_persistence::{load_trainer_from_path, save_trainer_to_path, PersistError};

#[derive(Debug, Clone)]
pub struct AutoTrainerConfig {
    pub cooldown_ticks: u32,
    pub drift_penalty: i32,

    // Optional autosave configuration.
    pub autosave_every_ticks: Option<u32>,
    pub autosave_every_seconds: Option<u64>,
    pub autosave_path: Option<std::path::PathBuf>,
}

pub struct AutoTrainerLoop {
    trainer: SelfTrainer,
    cfg: AutoTrainerConfig,
    tick: u64,
    last_choice: BackendChoice,
    last_switch_tick: u64,
    last_reason: String,
    dirty: bool,
    last_save_tick: u64,
    last_save_instant: std::time::Instant,
}

impl AutoTrainerLoop {
    pub fn new(cfg: AutoTrainerConfig) -> Self {
        AutoTrainerLoop {
            trainer: SelfTrainer::new(),
            cfg,
            tick: 0,
            last_choice: BackendChoice::Cpu,
            last_switch_tick: 0,
            last_reason: "init".to_string(),
            dirty: false,
            last_save_tick: 0,
            last_save_instant: std::time::Instant::now(),
        }
    }

    pub fn on_tick(
        &mut self,
        ctx: ExecutionContext,
        plan: &WarmStartPlan,
        drifts: &[DriftReport],
        res: ExecResult,
    ) -> BackendChoice {
        // 1) record episode
        record_from_warm_start(&mut self.trainer, ctx, plan, drifts, res);
        self.dirty = true;

        let current_tick = self.tick;

        // 2) raw recommendation
        let raw = recommend_for_next_tick(&self.trainer, ctx);

        // 3) cooldown & stabilization
        let final_choice;
        let reason;

        if !ctx.gpu_available {
            // Always prefer CPU when GPU is unavailable.
            final_choice = BackendChoice::Cpu;
            if self.last_choice == BackendChoice::Gpu {
                reason = "hold: gpu unavailable".to_string();
                self.last_switch_tick = current_tick;
            } else {
                reason = "hold: gpu unavailable".to_string();
            }
        } else {
            // GPU is available; apply cooldown-based hysteresis.
            if raw != self.last_choice {
                let cooldown = self.cfg.cooldown_ticks as u64;
                if cooldown > 0 && current_tick > 0 {
                    let since_switch = current_tick.saturating_sub(self.last_switch_tick);
                    if since_switch < cooldown {
                        // Cooldown active: keep previous choice.
                        final_choice = self.last_choice;
                        reason = "hold: cooldown active".to_string();
                    } else {
                        // Cooldown expired: allow switch.
                        final_choice = raw;
                        self.last_switch_tick = current_tick;
                        reason = match (self.last_choice, raw) {
                            (BackendChoice::Cpu, BackendChoice::Gpu) => {
                                "switch: cpu->gpu due to learned score".to_string()
                            }
                            (BackendChoice::Gpu, BackendChoice::Cpu) => {
                                "switch: gpu->cpu due to learned score".to_string()
                            }
                            _ => "switch: backend changed due to learned score".to_string(),
                        };
                    }
                } else {
                    // No cooldown configured or first tick: allow immediate switch.
                    final_choice = raw;
                    self.last_switch_tick = current_tick;
                    reason = match (self.last_choice, raw) {
                        (BackendChoice::Cpu, BackendChoice::Gpu) => {
                            "switch: cpu->gpu due to learned score".to_string()
                        }
                        (BackendChoice::Gpu, BackendChoice::Cpu) => {
                            "switch: gpu->cpu due to learned score".to_string()
                        }
                        _ => "switch: backend changed due to learned score".to_string(),
                    };
                }
            } else {
                // No change requested; keep current backend.
                final_choice = self.last_choice;
                reason = "hold: same backend preferred".to_string();
            }
        }

        self.last_choice = final_choice;
        self.last_reason = reason;
        self.tick = current_tick.saturating_add(1);

        // Autosave is best-effort and must not affect the backend choice.
        if let Err(_e) = self.try_autosave() {
            self.last_reason = "autosave: failed (io)".to_string();
        }

        final_choice
    }

    pub fn last_debug_reason(&self) -> &str {
        &self.last_reason
    }

    pub fn inner_trainer(&self) -> &SelfTrainer {
        &self.trainer
    }

    fn should_autosave(&self) -> bool {
        if !self.dirty {
            return false;
        }
        let path_set = match self.cfg.autosave_path {
            Some(_) => true,
            None => return false,
        };
        if !path_set {
            return false;
        }

        if let Some(n) = self.cfg.autosave_every_ticks {
            let cooldown = n as u64;
            if cooldown > 0 {
                let since = self.tick.saturating_sub(self.last_save_tick);
                if since >= cooldown {
                    return true;
                }
            }
        }

        if let Some(secs) = self.cfg.autosave_every_seconds {
            if self.last_save_instant.elapsed().as_secs() >= secs {
                return true;
            }
        }

        false
    }

    pub fn save_learning(&self, path: &std::path::Path) -> Result<(), PersistError> {
        save_trainer_to_path(&self.trainer, path)
    }

    pub fn load_learning(&mut self, path: &std::path::Path) -> Result<(), PersistError> {
        match load_trainer_from_path(path) {
            Ok(trainer) => {
                self.trainer = trainer;
                self.dirty = false;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn try_autosave(&mut self) -> Result<bool, PersistError> {
        if !self.should_autosave() {
            return Ok(false);
        }

        let path = match &self.cfg.autosave_path {
            Some(p) => p,
            None => return Ok(false),
        };

        match self.save_learning(path) {
            Ok(()) => {
                self.dirty = false;
                self.last_save_tick = self.tick;
                self.last_save_instant = std::time::Instant::now();
                self.last_reason = "autosave: wrote learning table".to_string();
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }
}
