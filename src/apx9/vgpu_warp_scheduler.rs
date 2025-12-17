// APX 9.18 — Warp Scheduler (WSIMT)
// Scheduler SIMT simulado, 100% CPU-only, sin GPU real ni paralelismo.

use crate::apx9::vgpu_warp::VGPUWarp;

#[derive(Debug, Clone, PartialEq)]
pub enum WarpState {
    Ready,
    Running,
    Blocked,
    Finished,
}

#[derive(Debug)]
pub struct VGPUWarpCtx {
    pub warp_id: usize,
    pub warp: VGPUWarp,
    pub state: WarpState,
}

pub struct VGPUWarpScheduler {
    pub warps: Vec<VGPUWarpCtx>,
    pub next: usize,
}

impl VGPUWarpScheduler {
    pub fn new(warps: Vec<VGPUWarpCtx>) -> Self {
        Self { warps, next: 0 }
    }

    /// Selección round-robin de warps en estado Ready.
    pub fn next_warp(&mut self) -> Option<&mut VGPUWarpCtx> {
        let n = self.warps.len();
        for _ in 0..n {
            let idx = self.next % n;
            self.next += 1;

            if self.warps[idx].state == WarpState::Ready {
                return Some(&mut self.warps[idx]);
            }
        }
        None
    }
}
