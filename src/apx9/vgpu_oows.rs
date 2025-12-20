// APX 9.22 — Out-of-Order Warp Scheduler (OOWS)
// Fully simulated and safe scheduler, without real GPU nor PTX/SASS.

use crate::apx9::vgpu_warp::VGPUWarp;
use crate::apx9::vgpu_pipeline::PipelineStage;

#[derive(Debug)]
pub struct VGPUOOWarpScheduler;

impl VGPUOOWarpScheduler {
    /// Select the index of a warp ready to execute, or None if there is none.
    pub fn select_warp(warps: &Vec<VGPUWarp>) -> Option<usize> {
        let mut best: Option<usize> = None;

        for (i, w) in warps.iter().enumerate() {
            if !w.active {
                continue;
            }

            // Must be in a state where it has pending work
            if w.fetched_instr.is_none() && w.stage != PipelineStage::Fetch {
                continue;
            }

            // If it has hazards, skip it
            if w.has_hazard() {
                continue;
            }

            // Simple selection: first-ready
            best = Some(i);
            break;
        }

        best
    }
}
