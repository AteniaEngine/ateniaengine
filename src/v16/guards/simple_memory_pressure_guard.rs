//! `SimpleMemoryPressureGuard`: reacción básica a presión de memoria.
//!
//! Produce `GuardAction::Degrade` cuando `conditions.memory_pressure`
//! supera un umbral empírico. El handler de `Degrade` en el `Graph`
//! (introducido en M3-e.1) responde migrando todos los tensores
//! `TensorStorage::Cuda` del grafo de vuelta a CPU, liberando VRAM.
//!
//! Diseño deliberadamente simple: un único umbral, sin histéresis ni
//! debounce. Si el `memory_pressure` fluctúa alrededor del umbral el
//! guard puede activarse y desactivarse en nodos consecutivos
//! ("flapping"). En la práctica no es problema porque (a) una vez que
//! `Degrade` corre, los tensores ya están en CPU y `memory_pressure`
//! baja, así que el próximo nodo ve `Continue`; (b) la decisión
//! migra tensores completos, no se puede "deshacer" trivialmente, así
//! que un flap solo produce migraciones adicionales, no inconsistencia.
//! Añadir histéresis o debounce queda como refinamiento futuro si la
//! telemetría real muestra ciclos problemáticos.

#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;

use super::execution_guard::ExecutionGuard;
use super::guard_action::GuardAction;
use super::guard_conditions::GuardConditions;

/// Umbral empírico a partir del cual se considera que la presión de
/// memoria (VRAM o RAM, el máximo de ambas según
/// `SignalBus::collect_memory_pressure`) justifica una reacción
/// `Degrade`. Punto de partida razonable; ajustar con mediciones de
/// workload real en milestones posteriores.
pub const DEGRADE_MEMORY_PRESSURE_THRESHOLD: f32 = 0.65;

/// Guard que emite `Degrade` cuando `memory_pressure` supera
/// [`DEGRADE_MEMORY_PRESSURE_THRESHOLD`]. El trigger es estrictamente
/// `>`, no `>=`: un valor exactamente igual al umbral es `Continue`.
///
/// El umbral por defecto es [`DEGRADE_MEMORY_PRESSURE_THRESHOLD`];
/// [`SimpleMemoryPressureGuard::with_threshold`] permite construcción
/// con un umbral alternativo (útil para tests y para políticas futuras
/// que quieran emitir `Degrade` más o menos agresivamente que la
/// configuración default).
pub struct SimpleMemoryPressureGuard {
    threshold: f32,
}

impl SimpleMemoryPressureGuard {
    /// Construye el guard con el umbral default
    /// [`DEGRADE_MEMORY_PRESSURE_THRESHOLD`].
    pub fn new() -> Self {
        Self {
            threshold: DEGRADE_MEMORY_PRESSURE_THRESHOLD,
        }
    }

    /// Construye el guard con un umbral alternativo. Útil para tests
    /// y para configuraciones que quieran desviar del default.
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Umbral configurado para esta instancia (para tests /
    /// diagnóstico).
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for SimpleMemoryPressureGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionGuard for SimpleMemoryPressureGuard {
    fn name(&self) -> &'static str {
        "simple_memory_pressure_guard"
    }

    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> GuardAction {
        if conditions.memory_pressure > self.threshold {
            GuardAction::Degrade
        } else {
            GuardAction::Continue
        }
    }
}
