// APX 9.14 — Virtual GPU Kernel Runner
// Safe IR executor over VGpuMemory. Does not use real GPU.

use crate::apx9::gpu_ir::*;
use crate::apx9::vgpu_memory::VGpuMemory;

pub struct VGpuRunner;

impl VGpuRunner {
    /// Execute a kernel IR over virtual memory.
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

                // APX 9.16 — Sync: in the current sequential simulation it does nothing,
                // but preserves the synchronization point in the IR.
                GpuOp::Sync => {
                    // symbolic no-op
                }

                // APX 9.17 — Predicate: marks a SIMT divergence point.
                // At this stage we still execute a single thread, so we
                // interpret it as a no-op to preserve math.
                GpuOp::Predicate { .. } => {
                    // symbolic no-op
                }
            }
        }
    }

    /// Map symbolic names to local/global slots deterministically.
    pub fn hash_slot(name: &str) -> usize {
        let mut h: u64 = 0;
        for b in name.as_bytes() {
            h = h.wrapping_mul(31).wrapping_add(*b as u64);
        }
        (h as usize) % 32
    }
}
