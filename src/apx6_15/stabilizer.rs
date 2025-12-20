use std::sync::{OnceLock, RwLock};

use crate::apx6_10::GlobalDecision;
use crate::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};

/// Small temperature wrapper for APX 6.15, built on top of 6.14.
#[derive(Debug, Clone, Copy)]
pub struct ApxTemperature {
    pub value: f32,
}

impl ApxTemperature {
    pub fn from_value(v: f32) -> Self {
        Self { value: v }
    }

    pub fn current(&self) -> f32 {
        self.value
    }
}

#[derive(Debug, Clone)]
pub struct ApxStabilizer {
    pub last_decision: Option<GlobalDecision>,
}

impl ApxStabilizer {
    pub fn new() -> Self {
        Self { last_decision: None }
    }

    /// Main entrypoint: decide "who is in charge" for this step.
    pub fn stabilize(
        &mut self,
        selector_decision: Option<GlobalDecision>,
        temperature: &ApxTemperature,
    ) -> GlobalDecision {
        // 1. If there is no data: baseline
        let Some(dec) = selector_decision else {
            self.last_decision = Some(GlobalDecision::NoPreference);
            set_runtime_policy(FusionRuntimePolicy::Baseline);
            return GlobalDecision::NoPreference;
        };

        // 2. Apply temperature-based probability (Softmax + sampling)
        // For APX 6.15 we rely on the discrete decision already produced by
        // selector 6.10/6.13: temperature here only controls stability against
        // changes, without touching real kernels.
        let t = temperature.current().max(0.05);
        let keep_prob = 1.0_f32.min(1.0_f32.max(1.0 - (t - 0.25).abs()));
        let r: f32 = rand::random();

        let sampled = if r < keep_prob {
            dec
        } else {
            // Small probability of returning to the last stable decision, if any.
            self.last_decision.unwrap_or(dec)
        };

        // 3. Update runtime policy without touching forward/backward
        match sampled {
            GlobalDecision::PreferFull =>
                set_runtime_policy(FusionRuntimePolicy::PreferFull),
            GlobalDecision::PreferQKV =>
                set_runtime_policy(FusionRuntimePolicy::PreferQKV),
            GlobalDecision::NoPreference =>
                set_runtime_policy(FusionRuntimePolicy::Baseline),
        }

        self.last_decision = Some(sampled);
        sampled
    }
}

static APX_GLOBAL_STABILIZER: OnceLock<RwLock<ApxStabilizer>> = OnceLock::new();

pub fn global_stabilizer() -> &'static RwLock<ApxStabilizer> {
    APX_GLOBAL_STABILIZER.get_or_init(|| RwLock::new(ApxStabilizer::new()))
}
