// APX 9.17 / 9.19 / 9.20 — SIMT Warp Model (Lanes, Warp Masks, Divergencia, Pipeline)
// Modelo SIMT simulado, 100% CPU-only, sin GPU real ni paralelismo.

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
    /// Máscara lógica de lanes activos (modelo SIMT formal APX 9.19).
    pub mask: WarpMask,
    /// Pila de reconvergencia de divergencia de control.
    pub div_stack: DivergenceStack,
    /// Program counter simbólico dentro del programa virtual.
    pub pc: usize,
    /// Etapa actual del pipeline F/D/E.
    pub stage: PipelineStage,
    /// Instrucción actualmente "fetcheada" por el pipeline.
    pub fetched_instr: Option<VGPUInstr>,
    /// Scoreboard SIMT para gestionar hazards de registros.
    pub scoreboard: VGPUScoreboard,
    /// Flag lógico de actividad del warp (cuando termina puede marcarse como inactivo).
    pub active: bool,
    /// Cola simbólica de instrucciones emitidas por el pipeline (APX 9.23 dual-issue).
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

    /// Devuelve true si la instrucción fetcheada tiene algún hazard RAW/WAW simbólico.
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

    /// Aplica un predicado a cada lane, actualizando el bitmask SIMT.
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

    /// Reconverge: todos los lanes vuelven a estar activos.
    pub fn reconverge(&mut self) {
        for lane in self.lanes.iter_mut() {
            lane.active = true;
        }
        self.mask = WarpMask::full();
    }
}
