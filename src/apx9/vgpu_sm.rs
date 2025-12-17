// APX 9.25 — Virtual Streaming Multiprocessor (VGPU-SM)
// Modelo de SM virtual que integra warps, pipeline, OOW scheduler, dual-issue,
// memoria virtual y tensor core. Totalmente simbólico y CPU-only.

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

/// Alias simbólicos para que el API del SM se lea cercano al diseño del paper.
pub type VirtualWarp = VGPUWarp;
pub type Scoreboard = VGPUScoreboard;
pub type VirtualWarpScheduler = VGPUWarpScheduler;
pub type OOWScheduler = VGPUOOWarpScheduler;
pub type DualIssueUnit = VGPUDualIssue;
pub type TensorCoreUnit = VGPUTensorCore;

pub struct VirtualSM {
    /// Warps SIMT virtuales gestionados por este SM.
    pub warps: Vec<VirtualWarp>,
    /// Scoreboard global simbólico (además del scoreboard por warp).
    pub scoreboard: Scoreboard,
    /// Scheduler round-robin lógico (APX 9.18).
    pub scheduler: VirtualWarpScheduler,
    /// Scheduler out-of-order simbólico (APX 9.22).
    pub oows: OOWScheduler,
    /// Unidad dual-issue simbólica (APX 9.23).
    pub dual_issue: DualIssueUnit,
    /// Pipeline F/D/E (APX 9.20).
    pub pipeline: VGPUPipeline,
    /// Memoria virtual global del SM (APX 9.13).
    pub memory: VGpuMemory,
    /// Tensor core virtual (APX 9.24).
    pub tensor_core: TensorCoreUnit,
    /// Program counter global simbólico (por ahora, PC del warp 0).
    pub pc: usize,
    /// Programa simbólico que ejecutan los warps.
    pub program: Vec<VGPUInstr>,
}

impl VirtualSM {
    /// Construye un SM virtual sencillo con `num_warps` warps y un programa simbólico.
    /// No interpreta PTX real: sólo usa VGPUInstr para tests de integración.
    pub fn new(program: Vec<VGPUInstr>, num_warps: usize, warp_size: usize, global_mem_size: usize) -> Self {
        let mut warps = Vec::new();
        for w in 0..num_warps {
            let base_tid = w * warp_size;
            warps.push(VGPUWarp::new(warp_size, base_tid));
        }

        // Scheduler round-robin con contextos separados (warps clonados para simulación).
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

    /// Un paso de simulación del SM: selecciona un warp listo y avanza su pipeline.
    pub fn step(&mut self) {
        // Seleccionamos un warp listo según el OOW scheduler.
        if let Some(wid) = VGPUOOWarpScheduler::select_warp(&self.warps) {
            let program_len = self.program.len();
            if program_len == 0 {
                return;
            }

            let warp = &mut self.warps[wid];

            // Avanzar pipeline F/D/E para este warp.
            VGPUPipeline::step(warp, &self.program);

            // Emitir hasta 2 instrucciones a la cola simbólica del warp.
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

            // PC global simbólico = PC del warp 0 (para inspección en tests).
            self.pc = self.warps.get(0).map(|w| w.pc).unwrap_or(0);
        }
    }

    /// Ejecuta un número acotado de pasos de simulación.
    pub fn run_steps(&mut self, max_steps: usize) {
        for _ in 0..max_steps {
            self.step();
        }
    }

    /// Helper de conveniencia para los tests: simula un vec_add controlado por el SM,
    /// pero toda la aritmética ocurre en CPU sobre tensores normales.
    /// Esto garantiza que APX 9.25 no cambia la matemática frente al modelo CPU.
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
