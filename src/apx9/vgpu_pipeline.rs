// APX 9.20 — SIMT Pipeline (Fetch → Decode → Execute)
// Per-warp simulated pipeline, 100% CPU-only and without real GPU.

use crate::apx9::vgpu_instr::VGPUInstr;
use crate::apx9::vgpu_warp::VGPUWarp;
use crate::apx9::vgpu_divergence::WarpMask;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Fetch,
    Decode,
    Execute,
}

pub struct VGPUPipeline;

impl VGPUPipeline {
    /// Advance one pipeline stage for a given warp over a virtual "program".
    /// Does not modify tensors nor real memory; only the warp's symbolic state.
    pub fn step(warp: &mut VGPUWarp, program: &Vec<VGPUInstr>) {
        match warp.stage {
            PipelineStage::Fetch => {
                if warp.pc < program.len() {
                    warp.fetched_instr = Some(program[warp.pc].clone());
                } else {
                    warp.fetched_instr = None;
                }
                warp.stage = PipelineStage::Decode;
            }

            PipelineStage::Decode => {
                if let Some(instr) = warp.fetched_instr.clone() {
                    // Check read hazards: if any src is busy, do not advance.
                    for src in instr.read_regs() {
                        if !warp.scoreboard.can_read(src) {
                            return;
                        }
                    }

                    // Check write hazards: if any dst is busy, do not advance.
                    for dst in instr.write_regs() {
                        if !warp.scoreboard.can_write(dst) {
                            return;
                        }
                    }
                }

                warp.stage = PipelineStage::Execute;
            }

            PipelineStage::Execute => {
                if let Some(instr) = warp.fetched_instr.clone() {
                    // Mark pending writes before executing.
                    for dst in instr.write_regs() {
                        warp.scoreboard.mark_write(dst);
                    }

                    Self::exec_instr(warp, &instr, program.len());

                    // Complete writes symbolically after "execution".
                    for dst in instr.write_regs() {
                        warp.scoreboard.mark_write_complete(dst);
                    }
                }
                warp.stage = PipelineStage::Fetch;
                warp.fetched_instr = None;
            }
        }
    }

    fn exec_instr(warp: &mut VGPUWarp, instr: &VGPUInstr, program_len: usize) {
        match instr {
            VGPUInstr::Noop => {
                if warp.pc + 1 < program_len {
                    warp.pc += 1;
                }
            }
            VGPUInstr::Add { .. } => {
                // In this phase we do not have real registers; only advance PC.
                if warp.pc + 1 < program_len {
                    warp.pc += 1;
                }
            }
            VGPUInstr::If { pred, then_pc, else_pc: _, join_pc } => {
                // Apply symbolic predication and push reconvergence state.
                warp.div_stack.push(warp.mask.clone(), *join_pc);
                warp.mask = WarpMask::from_predicate(pred);
                warp.pc = *then_pc;
            }
            VGPUInstr::Reconverge => {
                if let Some(frame) = warp.div_stack.pop() {
                    warp.mask = frame.mask;
                    warp.pc = frame.join_pc;
                }
            }

            // APX 9.24 — Symbolic HMMA: real matmul happens via VGPUTensorCore;
            // this exec_instr only maintains PC without altering additional state.
            VGPUInstr::HMMA { .. } => {
                if warp.pc + 1 < program_len {
                    warp.pc += 1;
                }
            }
        }
    }
}
