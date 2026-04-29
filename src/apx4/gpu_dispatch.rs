use crate::apx4::gpu_context::gpu_available;
use crate::apx4::gpu_kernels::gpu_matmul;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApxExecTarget {
    CPU,
    GPU,
    Auto,
}

/// M4.7.6.d — counter increment helper. The legacy
/// `gpu_matmul` (apx4) path is what the Llama 2 13B hot path
/// actually exercises: M4.7.3's residency-aware
/// `try_gpu_matmul` rejects 13B-scale MatMul because every
/// weight tensor (5120 × 5120 = 100 MB; 5120 × 13824 = 270 MB)
/// exceeds the 64 MiB `DEFAULT_BLOCK_SIZE` pool block (the
/// M4.7.6.c capacity check returns false for those shapes).
/// So the whole 13B forward routes through `dispatch_matmul_gpu`
/// → `gpu_matmul` and the M4.7.3 counters stay at zero — even
/// though the GPU is doing the work.
///
/// The fix is to count on this path too. The
/// `gpu_matmul_*_count` semantics are unified: they count "any
/// MatMul that the engine handed off to a CUDA kernel",
/// regardless of which dispatch primitive routed it. The two
/// existing `try_gpu_matmul` counters
/// (`GPU_MATMUL_RESIDENT_COUNT`, `GPU_MATMUL_ROUNDTRIP_COUNT`)
/// stay as M4.7.3-specific instrumentation; this new
/// `GPU_MATMUL_LEGACY_COUNT` covers the apx4 path. The public
/// API in `gpu/dispatch/hooks.rs` exposes the union via
/// `gpu_matmul_total_count()` so the demo and tests can ask
/// "did GPU MatMul fire at all?" with one call.
fn record_legacy_gpu_matmul() {
    crate::gpu::dispatch::hooks::increment_legacy_gpu_matmul_counter();
}

pub fn dispatch_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
    target: ApxExecTarget,
) {
    match target {
        ApxExecTarget::GPU => {
            if gpu_available() {
                gpu_matmul(a, b, m, k, n, out);
                record_legacy_gpu_matmul();
            } else {
                crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
            }
        }
        ApxExecTarget::CPU => {
            crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
        }
        ApxExecTarget::Auto => {
            if gpu_available() {
                gpu_matmul(a, b, m, k, n, out);
                record_legacy_gpu_matmul();
            } else {
                crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
            }
        }
    }
}
