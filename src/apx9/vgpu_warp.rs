// APX 9.17 / 9.19 / 9.20 — SIMT Warp Model (Lanes, Warp Masks, Divergence, Pipeline)
// Simulated SIMT model, 100% CPU-only, without real GPU nor parallelism.

use crate::apx9::vgpu_divergence::{WarpMask, DivergenceStack};
use crate::apx9::vgpu_scoreboard::VGPUScoreboard;
use crate::apx9::vgpu_pipeline::PipelineStage;
use crate::apx9::vgpu_instr::VGPUInstr;

#[derive(Debug, Clone)]
pub struct VGPULane {
    pub tid: usize,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct VGPUWarp {
    pub lanes: Vec<VGPULane>,
    /// Logical mask of active lanes (formal SIMT model APX 9.19).
    pub mask: WarpMask,
    /// Reconvergence stack for control divergence.
    pub div_stack: DivergenceStack,
    /// Symbolic program counter within the virtual program.
    pub pc: usize,
    /// Current pipeline F/D/E stage.
    pub stage: PipelineStage,
    /// Instruction currently fetched by the pipeline.
    pub fetched_instr: Option<VGPUInstr>,
    /// SIMT scoreboard to manage register hazards.
    pub scoreboard: VGPUScoreboard,
    /// Warp activity flag (when finished it can be marked inactive).
    pub active: bool,
    /// Symbolic queue of instructions issued by the pipeline (APX 9.23 dual-issue).
    pub pipeline: Vec<VGPUInstr>,
}

impl VGPUWarp {
    pub fn new(warp_size: usize, base_tid: usize) -> Self {
        let mut lanes = Vec::new();
        for i in 0..warp_size {
            lanes.push(VGPULane {
                tid: base_tid + i,
                active: true,
            });
        }
        Self {
            lanes,
            mask: WarpMask::full(),
            div_stack: DivergenceStack::new(),
            pc: 0,
            stage: PipelineStage::Fetch,
            fetched_instr: None,
            scoreboard: VGPUScoreboard::new(64),
            active: true,
            pipeline: Vec::new(),
        }
    }

    /// Return true if the fetched instruction has any symbolic RAW/WAW hazard.
    pub fn has_hazard(&self) -> bool {
        if let Some(instr) = &self.fetched_instr {
            for r in instr.read_regs() {
                if !self.scoreboard.can_read(r) {
                    return true;
                }
            }
            for r in instr.write_regs() {
                if !self.scoreboard.can_write(r) {
                    return true;
                }
            }
        }
        false
    }

    /// Apply a predicate to each lane, updating the SIMT bitmask.
    pub fn apply_predicate<F>(&mut self, pred: F)
    where
        F: Fn(usize) -> bool,
    {
        let mut pred_bits = [false; 32];
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            lane.active = pred(lane.tid);
            if lane.active {
                if i < 32 {
                    pred_bits[i] = true;
                }
            }
        }
        self.mask = WarpMask::from_predicate(&pred_bits);
    }

    /// Reconverge: all lanes become active again.
    pub fn reconverge(&mut self) {
        for lane in self.lanes.iter_mut() {
            lane.active = true;
        }
        self.mask = WarpMask::full();
    }
}
