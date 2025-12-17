// APX 9.14 — Virtual GPU Kernel Runner
// Ejecutor seguro de IR sobre VGpuMemory. No usa GPU real.

use crate::apx9::gpu_ir::*;
use crate::apx9::vgpu_memory::VGpuMemory;

pub struct VGpuRunner;

impl VGpuRunner {
    /// Ejecuta un kernel IR sobre memoria virtual.
    pub fn run_kernel(
        ir: &GpuKernelIR,
        mem: &mut VGpuMemory,
        _block_id: usize,
        thread_id: usize,
    ) {
        for op in &ir.ops {
            match op {
                // Interpretamos `Load { dst, src }` como:
                // local[dst_hash] = global[src_hash]
                GpuOp::Load { dst, src } => {
                    let dst_idx = Self::hash_slot(dst);
                    let src_idx = Self::hash_slot(src);
                    let v = mem.load_global(src_idx);
                    mem.store_local(thread_id, dst_idx, v);
                }

                // `Add { dst, a, b }` = local[dst] = local[a] + local[b]
                GpuOp::Add { dst, a, b } => {
                    let dst_idx = Self::hash_slot(dst);
                    let a_idx = Self::hash_slot(a);
                    let b_idx = Self::hash_slot(b);
                    let va = mem.load_local(thread_id, a_idx);
                    let vb = mem.load_local(thread_id, b_idx);
                    mem.store_local(thread_id, dst_idx, va + vb);
                }

                // `Store { dst, src }` = global[dst] = local[src]
                GpuOp::Store { dst, src } => {
                    let dst_idx = Self::hash_slot(dst);
                    let src_idx = Self::hash_slot(src);
                    let v = mem.load_local(thread_id, src_idx);
                    mem.store_global(dst_idx, v);
                }

                // APX 9.16 — Sync: en la simulación secuencial actual no hace nada,
                // pero preserva el punto de sincronización en el IR.
                GpuOp::Sync => {
                    // no-op simbólico
                }

                // APX 9.17 — Predicate: marca un punto de divergencia SIMT.
                // En esta fase seguimos ejecutando un único thread, así que
                // lo interpretamos como no-op para mantener la matemática.
                GpuOp::Predicate { .. } => {
                    // no-op simbólico
                }
            }
        }
    }

    /// Mapear nombres simbólicos a slots de local/global de forma determinista.
    pub fn hash_slot(name: &str) -> usize {
        let mut h: u64 = 0;
        for b in name.as_bytes() {
            h = h.wrapping_mul(31).wrapping_add(*b as u64);
        }
        (h as usize) % 32
    }
}
