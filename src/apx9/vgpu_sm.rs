// APX 9.25 — Virtual Streaming Multiprocessor (VGPU-SM)
// Virtual SM model that integrates warps, pipeline, OOW scheduler, dual-issue,
// virtual memory, and tensor core. Fully symbolic and CPU-only.

use crate::apx9::vgpu_warp::VGPUWarp;
use crate::apx9::vgpu_warp_scheduler::VGPUWarpScheduler;
use crate::apx9::vgpu_scoreboard::VGPUScoreboard;
use crate::apx9::vgpu_oows::VGPUOOWarpScheduler;
use crate::apx9::vgpu_dual_issue::VGPUDualIssue;
use crate::apx9::vgpu_pipeline::VGPUPipeline;
use crate::apx9::vgpu_memory::VGpuMemory;
use crate::apx9::vgpu_tensor_core::VGPUTensorCore;
use crate::apx9::vgpu_instr::VGPUInstr;
use crate::apx9::vgpu_warp_scheduler::{VGPUWarpCtx, WarpState};
use crate::tensor::{Tensor, Device, DType};

/// Symbolic aliases so the SM API reads close to the paper design.
pub type VirtualWarp = VGPUWarp;
pub type Scoreboard = VGPUScoreboard;
pub type VirtualWarpScheduler = VGPUWarpScheduler;
pub type OOWScheduler = VGPUOOWarpScheduler;
pub type DualIssueUnit = VGPUDualIssue;
pub type TensorCoreUnit = VGPUTensorCore;

pub struct VirtualSM {
    /// Virtual SIMT warps managed by this SM.
    pub warps: Vec<VirtualWarp>,
    /// Global symbolic scoreboard (in addition to per-warp scoreboard).
    pub scoreboard: Scoreboard,
    /// Logical round-robin scheduler (APX 9.18).
    pub scheduler: VirtualWarpScheduler,
    /// Symbolic out-of-order scheduler (APX 9.22).
    pub oows: OOWScheduler,
    /// Symbolic dual-issue unit (APX 9.23).
    pub dual_issue: DualIssueUnit,
    /// F/D/E pipeline (APX 9.20).
    pub pipeline: VGPUPipeline,
    /// SM global virtual memory (APX 9.13).
    pub memory: VGpuMemory,
    /// Virtual tensor core (APX 9.24).
    pub tensor_core: TensorCoreUnit,
    /// Global symbolic program counter (for now, warp 0 PC).
    pub pc: usize,
    /// Symbolic program executed by the warps.
    pub program: Vec<VGPUInstr>,
}

impl VirtualSM {
    /// Build a simple virtual SM with `num_warps` warps and a symbolic program.
    /// Does not interpret real PTX: only uses VGPUInstr for integration tests.
    pub fn new(program: Vec<VGPUInstr>, num_warps: usize, warp_size: usize, global_mem_size: usize) -> Self {
        let mut warps = Vec::new();
        for w in 0..num_warps {
            let base_tid = w * warp_size;
            warps.push(VGPUWarp::new(warp_size, base_tid));
        }

        // Round-robin scheduler with separate contexts (cloned warps for simulation).
        let rr_warps: Vec<VGPUWarpCtx> = warps
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, w)| VGPUWarpCtx { warp_id: i, warp: w, state: WarpState::Ready })
            .collect();

        Self {
            warps,
            scoreboard: VGPUScoreboard::new(64),
            scheduler: VGPUWarpScheduler::new(rr_warps),
            oows: VGPUOOWarpScheduler,
            dual_issue: VGPUDualIssue,
            pipeline: VGPUPipeline,
            memory: VGpuMemory::new(global_mem_size, 0, 1, 1),
            tensor_core: VGPUTensorCore,
            pc: 0,
            program,
        }
    }

    /// One SM simulation step: select a ready warp and advance its pipeline.
    pub fn step(&mut self) {
        // Select a ready warp according to the OOW scheduler.
        if let Some(wid) = VGPUOOWarpScheduler::select_warp(&self.warps) {
            let program_len = self.program.len();
            if program_len == 0 {
                return;
            }

            let warp = &mut self.warps[wid];

            // Advance F/D/E pipeline for this warp.
            VGPUPipeline::step(warp, &self.program);

            // Issue up to 2 instructions to the warp's symbolic queue.
            let i1 = if warp.pc < program_len {
                Some(self.program[warp.pc].clone())
            } else {
                None
            };
            let i2 = if warp.pc + 1 < program_len {
                Some(self.program[warp.pc + 1].clone())
            } else {
                None
            };
            let _ = VGPUDualIssue::issue(warp, i1, i2);

            // Global symbolic PC = warp 0 PC (for inspection in tests).
            self.pc = self.warps.get(0).map(|w| w.pc).unwrap_or(0);
        }
    }

    /// Run a bounded number of simulation steps.
    pub fn run_steps(&mut self, max_steps: usize) {
        for _ in 0..max_steps {
            self.step();
        }
    }

    /// Convenience helper for tests: simulate an SM-controlled vec_add,
    /// but all arithmetic happens on CPU over normal tensors.
    /// This guarantees APX 9.25 does not change math compared to the CPU model.
    pub fn simulate_vec_add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.data.len(), b.data.len());
        let len = a.data.len();

        let mut out = Tensor::zeros(vec![len], Device::CPU, DType::F32);
        for i in 0..len {
            out.data[i] = a.data[i] + b.data[i];
        }
        out
    }
}
