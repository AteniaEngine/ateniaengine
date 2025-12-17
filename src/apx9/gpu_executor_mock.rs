// APX 9.8 — GPU Executor Mock (GXE v0)
// Ejecuta un GPUExecutionPlan de forma 100% simulada: no ejecuta kernels reales
// ni usa VRAM; sólo acumula tiempos y métricas simbólicas.

use crate::apx9::gpu_execution_planner::{GPUExecutionPlan, GPUPlanStep};
use crate::apx9::gpu_autotuner::GpuAutoTuner;
use std::sync::RwLock;
use once_cell::sync::Lazy;

// APX 9.9: auto-tuner global basado exclusivamente en tiempos simulados del GXE.
static AUTOTUNER: Lazy<RwLock<GpuAutoTuner>> = Lazy::new(|| RwLock::new(GpuAutoTuner::new()));

#[derive(Debug)]
pub struct GPUExecutionResult {
    pub total_time_ms: f32,
    pub per_step: Vec<f32>,
    pub spills: usize,
    pub executed_steps: usize,
}

pub fn execute_plan_mock(plan: &GPUExecutionPlan) -> GPUExecutionResult {
    if plan.steps.is_empty() {
        return GPUExecutionResult {
            total_time_ms: 0.0,
            per_step: Vec::new(),
            spills: 0,
            executed_steps: 0,
        };
    }

    let mut total_time_ms = 0.0f32;
    let mut per_step = Vec::with_capacity(plan.steps.len());
    let mut spills = 0usize;

    for (idx, step) in plan.steps.iter().enumerate() {
        let t = simulate_step_time(step, idx as u32);

        if step.spill_to_cpu {
            spills += 1;
        }

        if crate::apx_debug_enabled() {
            eprintln!(
                "[APX 9.8 GXE] step={} device={} kernel={} time={:.4}ms partitions={}",
                step.node_id,
                step.device,
                step.kernel_name,
                t,
                step.partitions.len(),
            );
        }

        total_time_ms += t;
        per_step.push(t);

        // APX 9.9: registrar muestras CPU/GPU simuladas para el auto-tuner.
        if crate::apx_mode_at_least("9.9") {
            let tensor_size: usize = step
                .partitions
                .iter()
                .map(|(s, e)| e.saturating_sub(*s))
                .sum();
            let cpu_time_us: u64 = (step.estimated_time_ms.max(0.0) * 1000.0) as u64;
            let gpu_time_us: u64 = (t.max(0.0) * 1000.0) as u64;

            if tensor_size > 0 {
                if let Ok(mut at) = AUTOTUNER.write() {
                    at.record(tensor_size, cpu_time_us, gpu_time_us);
                }
            }
        }
    }

    GPUExecutionResult {
        total_time_ms,
        per_step,
        spills,
        executed_steps: plan.steps.len(),
    }
}

/// Simulación determinista del tiempo por step.
fn simulate_step_time(step: &GPUPlanStep, seed: u32) -> f32 {
    let mut base = step.estimated_time_ms.max(0.0);

    // Pequeña variación determinista ±5% basada en un hash simple.
    let mut h: u64 = seed as u64;
    for b in step.kernel_name.as_bytes() {
        h = h.wrapping_mul(31).wrapping_add(*b as u64);
    }
    let frac = (h % 1000) as f32 / 1000.0; // [0,1)
    let factor = 0.95 + frac * 0.10;       // [0.95,1.05]
    base *= factor;

    // Overhead por particiones múltiples.
    if step.partitions.len() > 1 {
        let overhead = step.estimated_time_ms * 0.05 * (step.partitions.len() as f32);
        base += overhead;
    }

    base
}
