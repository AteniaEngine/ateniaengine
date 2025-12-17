use std::sync::{OnceLock, RwLock};

use crate::apx6_10::GlobalDecision;
use crate::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};

/// Pequeño wrapper de temperatura para APX 6.15 apoyado en 6.14.
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

    /// Entrada principal: decide "quién manda" para este paso.
    pub fn stabilize(
        &mut self,
        selector_decision: Option<GlobalDecision>,
        temperature: &ApxTemperature,
    ) -> GlobalDecision {
        // 1. Si no hay datos  baseline
        let Some(dec) = selector_decision else {
            self.last_decision = Some(GlobalDecision::NoPreference);
            set_runtime_policy(FusionRuntimePolicy::Baseline);
            return GlobalDecision::NoPreference;
        };

        // 2. Aplicar probabilidad con temperatura (Softmax + sampling)
        // Para APX 6.15 nos apoyamos en la decisión discreta ya producida por
        // el selector 6.10/6.13: la temperatura aquí sólo controla la
        // estabilidad frente a cambios, sin tocar kernels reales.
        let t = temperature.current().max(0.05);
        let keep_prob = 1.0_f32.min(1.0_f32.max(1.0 - (t - 0.25).abs()));
        let r: f32 = rand::random();

        let sampled = if r < keep_prob {
            dec
        } else {
            // Pequeña probabilidad de volver a la última decisión estable si existe.
            self.last_decision.unwrap_or(dec)
        };

        // 3. Actualizar runtime policy sin tocar forward/backward
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
