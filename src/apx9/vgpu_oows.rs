// APX 9.22 — Out-of-Order Warp Scheduler (OOWS)
// Scheduler totalmente simulado y seguro, sin GPU real ni PTX/SASS.

use crate::apx9::vgpu_warp::VGPUWarp;
use crate::apx9::vgpu_pipeline::PipelineStage;

#[derive(Debug)]
pub struct VGPUOOWarpScheduler;

impl VGPUOOWarpScheduler {
    /// Selecciona el índice de un warp listo para ejecutar, o None si no hay.
    pub fn select_warp(warps: &Vec<VGPUWarp>) -> Option<usize> {
        let mut best: Option<usize> = None;

        for (i, w) in warps.iter().enumerate() {
            if !w.active {
                continue;
            }

            // Debe estar en un estado donde tenga trabajo pendiente
            if w.fetched_instr.is_none() && w.stage != PipelineStage::Fetch {
                continue;
            }

            // Si tiene hazards, lo saltamos
            if w.has_hazard() {
                continue;
            }

            // Selección simple: first-ready
            best = Some(i);
            break;
        }

        best
    }
}
