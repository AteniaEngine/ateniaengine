use super::placement_types::*;
use super::types::{GlobalHardwareSnapshot, GpuCaps, ReliabilityStats};

const VRAM_SAFE: f32 = 0.80;
const RAM_SAFE: f32 = 0.85;
const TENSOR_HUGE_BYTES: u64 = 512 * 1024 * 1024;
const COMPUTE_HEAVY: f32 = 100.0; // abstract units
const VRAM_CRITICAL: f32 = 0.95;
const RAM_CRITICAL: f32 = 0.95;
const RAM_VERY_HIGH: f32 = 0.90;
const SSD_LAST_RESORT_ALLOWED: bool = true;

pub struct TensorPlacementEngine;

fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn reliability_score(stats: Option<&ReliabilityStats>) -> f32 {
    match stats {
        None => 1.0,
        Some(s) => {
            let total = s.ok_count + s.fail_count;
            let mut score = if total == 0 {
                1.0
            } else {
                s.ok_count as f32 / total as f32
            };
            score = clamp01(score);

            if s.last_error_epoch_ms.is_some() {
                score *= 0.85;
            }

            clamp01(score)
        }
    }
}

fn best_gpu<'a>(hw: &'a GlobalHardwareSnapshot) -> Option<(&'a GpuCaps, f32)> {
    let mut best: Option<(&GpuCaps, f32)> = None;

    for gpu in &hw.gpus {
        let stats = hw.reliability_by_device.get(&gpu.id);
        let score = reliability_score(stats);

        best = match best {
            None => Some((gpu, score)),
            Some((best_gpu, best_score)) => {
                if score > best_score {
                    Some((gpu, score))
                } else if (score - best_score).abs() < f32::EPSILON {
                    let best_compute = best_gpu.compute_score_est.unwrap_or(0.0);
                    let cur_compute = gpu.compute_score_est.unwrap_or(0.0);

                    if cur_compute > best_compute {
                        Some((gpu, score))
                    } else if (cur_compute - best_compute).abs() < f32::EPSILON
                        && gpu.id < best_gpu.id
                    {
                        Some((gpu, score))
                    } else {
                        Some((best_gpu, best_score))
                    }
                } else {
                    Some((best_gpu, best_score))
                }
            }
        };
    }

    best
}

impl TensorPlacementEngine {
    pub fn decide(tensor: &TensorProfile, hw: &GlobalHardwareSnapshot) -> PlacementDecision {
        // Default safe fallback
        let mut decision = PlacementDecision {
            target: PlacementTarget::Cpu,
            device_id: None,
            reason: "Default CPU fallback".to_string(),
        };

        // If no GPU available at all, stay on CPU
        if hw.gpus.is_empty() {
            decision.reason = "No GPU detected".to_string();
            return decision;
        }

        let tensor_size_bytes = tensor.total_size_bytes();
        let compute_cost = tensor.estimated_compute_cost.unwrap_or(0.0);

        // Compute memory pressure signals (best-effort)
        let vram_pressure = hw.pressure.vram_pressure.unwrap_or(1.0);
        let ram_pressure = hw.pressure.ram_pressure.unwrap_or(0.0);

        let best_gpu = match best_gpu(hw) {
            Some(v) => v,
            None => {
                decision.reason = "No suitable GPU found; falling back to CPU".to_string();
                return decision;
            }
        };

        let (gpu, reliability) = best_gpu;
        let vram_safe = vram_pressure < VRAM_SAFE;
        let ram_safe = ram_pressure < RAM_SAFE;
        let is_huge = tensor_size_bytes > TENSOR_HUGE_BYTES;
        let is_compute_heavy = compute_cost > COMPUTE_HEAVY;
        let vram_critical = vram_pressure >= VRAM_CRITICAL;
        let ram_critical = ram_pressure >= RAM_CRITICAL;
        let ram_very_high = ram_pressure > RAM_VERY_HIGH;

        // Prefer GPU/VRAM for compute-heavy tensors when pressure and reliability allow it
        if vram_safe && !is_huge && (is_compute_heavy || reliability > 0.0) {
            decision.target = PlacementTarget::Vram;
            decision.device_id = Some(gpu.id.clone());
            decision.reason = format!(
                "Selected GPU {} (reliability={:.2}) and VRAM pressure is safe",
                gpu.id, reliability
            );
            return decision;
        }

        // Huge tensors prefer RAM when pressure allows it, even if VRAM is moderately free
        if is_huge && ram_safe {
            // If RAM is already very high and the tensor is not compute-heavy,
            // prefer SSD as a last resort to reduce OOM risk.
            if SSD_LAST_RESORT_ALLOWED && ram_very_high && !is_compute_heavy {
                decision.target = PlacementTarget::Ssd;
                decision.device_id = None;
                decision.reason = "RAM pressure is very high and tensor is huge but not compute-heavy; placing tensor on SSD as last resort".to_string();
            } else {
                decision.target = PlacementTarget::Ram;
                decision.reason = "Tensor is huge and RAM pressure is acceptable; placing tensor in RAM"
                    .to_string();
            }
            return decision;
        }

        // If VRAM is under pressure but RAM is fine, go to RAM
        if !vram_safe && ram_safe {
            decision.target = PlacementTarget::Ram;
            decision.reason = "VRAM pressure is high; placing tensor in RAM".to_string();
            return decision;
        }

        // SSD last-resort when both VRAM and RAM are under critical pressure and
        // the tensor is huge.
        if SSD_LAST_RESORT_ALLOWED && is_huge && vram_critical && ram_critical {
            decision.target = PlacementTarget::Ssd;
            decision.device_id = None;
            decision.reason =
                "VRAM and RAM are under critical pressure; placing huge tensor on SSD as last resort"
                    .to_string();
            return decision;
        }

        // High pressure everywhere: CPU fallback
        decision.target = PlacementTarget::Cpu;
        decision.reason = "System memory pressure is high; falling back to CPU".to_string();
        decision
    }
}
