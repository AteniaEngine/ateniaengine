use crate::apx9::vgpu_runner::VGpuRunner;
use crate::apx9::vgpu_memory::VGpuMemory;
use crate::apx9::gpu_ir::GpuKernelIR;

// APX 9.15 — Virtual GPU Block Launcher
// Coordinates grid, blocks, and threads, running IR using the runner.
// DISCLAIMER: fully virtual and CPU-only execution, without real GPU nor VRAM.

pub struct VGpuBlockLauncher;

impl VGpuBlockLauncher {
    /// Execute a kernel IR in the specified virtual grid.
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
