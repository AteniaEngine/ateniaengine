// APX 9.20 — SIMT Pipeline (Fetch → Decode → Execute)
// Pipeline simulado por warp, 100% CPU-only y sin GPU real.

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
    /// Avanza una etapa del pipeline para un warp dado sobre un "programa" virtual.
    /// No modifica tensores ni memoria real, sólo estado simbólico del warp.
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
                    // Chequear hazards de lectura: si algún src está ocupado, no avanzamos.
                    for src in instr.read_regs() {
                        if !warp.scoreboard.can_read(src) {
                            return;
                        }
                    }

                    // Chequear hazards de escritura: si algún dst está ocupado, no avanzamos.
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
                    // Marcar writes pendientes antes de ejecutar.
                    for dst in instr.write_regs() {
                        warp.scoreboard.mark_write(dst);
                    }

                    Self::exec_instr(warp, &instr, program.len());

                    // Completar writes simbólicamente tras la "ejecución".
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
                // En esta fase no tenemos registros reales; solo avanzamos PC.
                if warp.pc + 1 < program_len {
                    warp.pc += 1;
                }
            }
            VGPUInstr::If { pred, then_pc, else_pc: _, join_pc } => {
                // Aplicar predicación simbólica y apilar estado de reconvergencia.
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

            // APX 9.24 — HMMA simbólica: el matmul real se hace a través de VGPUTensorCore,
            // este exec_instr sólo mantiene el PC sin alterar estado adicional.
            VGPUInstr::HMMA { .. } => {
                if warp.pc + 1 < program_len {
                    warp.pc += 1;
                }
            }
        }
    }
}
