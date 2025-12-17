use crate::apx9::vgpu_runner::VGpuRunner;
use crate::apx9::vgpu_memory::VGpuMemory;
use crate::apx9::gpu_ir::GpuKernelIR;

// APX 9.15 — Virtual GPU Block Launcher
// Coordina grid, bloques y threads, ejecutando IR usando el runner.
// DISCLAIMER: ejecución totalmente virtual y CPU-only, sin GPU real ni VRAM.

pub struct VGpuBlockLauncher;

impl VGpuBlockLauncher {
    /// Ejecuta un kernel IR en el grid virtual especificado.
    pub fn launch(
        ir: &GpuKernelIR,
        mem: &mut VGpuMemory,
        grid_dim: usize,
        block_dim: usize,
    ) {
        for block_id in 0..grid_dim {
            for thread_id in 0..block_dim {
                VGpuRunner::run_kernel(ir, mem, block_id, thread_id);
            }
        }
    }
}
