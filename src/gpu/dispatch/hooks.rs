//! Vendor-neutral home of the per-op GPU dispatch hooks.
//!
//! Moved from `crate::apx4_11::gpu_hooks`; the original file is kept
//! as a pass-through re-export so existing callers in `amg::graph` and
//! elsewhere keep compiling. Imports of `crate::cuda::*` are contained
//! inside this module (the `src/gpu/` layer is the designated home for
//! the vendor-specific glue); upstream modules only see the
//! `&Tensor`-based public API.
//!
//! Keeps the pool coupling with `crate::apx4_12::pool_dispatcher`: APX
//! 4.11 (these hooks) and APX 4.12 (memory pool) are coevolutions and
//! the hook consults the pool before spending VRAM.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::{Tensor, TensorStorage};
use crate::cuda::{
    self, matmul::cuda_matmul, matmul::cuda_matmul_inplace,
    fused_linear_silu::cuda_fused_linear_silu,
};
use crate::amg::graph::Graph;
use crate::apx4_12::pool_dispatcher::try_gpu_with_pool;

fn apx_trace_enabled() -> bool {
    matches!(std::env::var("APX_TRACE").as_deref(), Ok("1")) && !crate::apx_is_silent()
}

/// M4.7.3.e — process-wide counters for `try_gpu_matmul` execution
/// path. Exposed for smoke / regression tests that need concrete
/// evidence the GPU branch actually ran instead of falling back to
/// CPU. Counts are monotonic across the process lifetime; tests that
/// care about a delta should snapshot before, run, and snapshot
/// after. No production behaviour depends on these counters — they
/// are observability-only.
static GPU_MATMUL_RESIDENT_COUNT: AtomicUsize = AtomicUsize::new(0);
static GPU_MATMUL_ROUNDTRIP_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the count of MatMul invocations that took the residency
/// (all-Cuda → device-pointer) path inside `try_gpu_matmul`.
pub fn gpu_matmul_resident_count() -> usize {
    GPU_MATMUL_RESIDENT_COUNT.load(Ordering::Relaxed)
}

/// Returns the count of MatMul invocations that took the CPU-roundtrip
/// (host alloc → upload → kernel → download) path inside
/// `try_gpu_matmul`.
pub fn gpu_matmul_roundtrip_count() -> usize {
    GPU_MATMUL_ROUNDTRIP_COUNT.load(Ordering::Relaxed)
}

/// Simple heuristic to decide whether it is worth using the CUDA MatMul kernel.
///
/// **M4.7.6.c — pool capacity check added.** The CPU-roundtrip path
/// inside `try_gpu_matmul` allocates 3 device buffers via the
/// fixed-size APX 4.12 pool (`DEFAULT_BLOCK_SIZE = 64 MiB`, see
/// `crate::apx4_12::DEFAULT_BLOCK_SIZE`). A single allocation request
/// larger than the block size cannot be served — the buffer is
/// undersized, the subsequent `cudaMemcpy` writes past the end, and
/// the driver surfaces `TransferFailed`. Llama 2 13B's LM head
/// (5120 × 32000 × 4 = 655 MB), Qwen 2.5's LM head (151936 × 1536
/// × 4 = 933 MB), and similar large GEMMs are all above the 64 MiB
/// limit. Pre-M4.7.6.c the `!in_gpu_segment` gate kept these out of
/// `try_gpu_matmul`, masking the issue. With the gate gone in
/// M4.7.6.c, the shape check has to take over: if any of the three
/// buffers (A, B, output) would exceed one pool block, return false
/// and let the caller's CPU / legacy `dispatch_matmul_gpu` path run.
pub fn gpu_can_run_matmul(m: usize, k: usize, n: usize) -> bool {
    // Require a minimum amount of work to amortize host<->device overhead.
    let ops = m.saturating_mul(k).saturating_mul(n);
    if ops <= 256 {
        return false;
    }

    // The remaining validations (device, dtype, shapes) are performed in `try_gpu_matmul`.
    if !cuda::cuda_available() {
        return false;
    }

    // M4.7.6.c — bail out when any of the three buffers would
    // exceed the pool block size. The pool serves one block per
    // allocation request, so an oversize request is unservable.
    let f32_size = std::mem::size_of::<f32>();
    let a_bytes = m.saturating_mul(k).saturating_mul(f32_size);
    let b_bytes = k.saturating_mul(n).saturating_mul(f32_size);
    let out_bytes = m.saturating_mul(n).saturating_mul(f32_size);
    let max_per_alloc = a_bytes.max(b_bytes).max(out_bytes);
    if max_per_alloc > crate::apx4_12::DEFAULT_BLOCK_SIZE {
        return false;
    }

    true
}

/// Attempt to execute MatMul on GPU (M4.7.3.a residency-aware).
///
/// Returns `true` if it ran on GPU and `out` contains the result.
///
/// Per-storage gating (option (b) of the M4.7.3 investigation):
///
/// - **Both operands `Cuda` and `out` is `Cuda`**: residency path.
///   Calls [`cuda_matmul_inplace`] directly on the device pointers.
///   The shape gate (`gpu_can_run_matmul`) is bypassed because
///   uploading the operands has already been paid; running CPU on
///   already-VRAM-resident data would force a download.
/// - **All operands `Cpu`**: existing CPU-roundtrip path. Shape gate
///   applies; output materialised back to Cpu.
/// - **Mixed (one operand on each)**: returns `false`. The caller
///   normalises the storage at the executor arm via `ensure_decoded`
///   before re-dispatching, so this branch is unreachable on the
///   Llama hot path; it stays as a defensive fallback.
pub fn try_gpu_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> bool {
    if a.dtype != b.dtype || a.dtype != out.dtype {
        return false;
    }
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return false;
    }

    let m = a.shape[0];
    let k = a.shape[1];
    if b.shape[0] != k {
        return false;
    }
    let n = b.shape[1];

    if out.shape != [m, n] {
        return false;
    }

    if !cuda::cuda_available() {
        return false;
    }

    // ---- Residency path: every operand and the output already on VRAM ----
    let all_cuda = matches!(
        (&a.storage, &b.storage, &out.storage),
        (
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
        )
    );
    if all_cuda {
        cuda_matmul_inplace(a, b, out, m, k, n);
        GPU_MATMUL_RESIDENT_COUNT.fetch_add(1, Ordering::Relaxed);
        if apx_trace_enabled() {
            println!("[APX M4.7.3] GPU MatMul executed (residency-aware)");
        }
        return true;
    }

    // ---- CPU-roundtrip path: shape gate + pool budget ----
    let all_cpu = matches!(
        (&a.storage, &b.storage, &out.storage),
        (
            TensorStorage::Cpu(_),
            TensorStorage::Cpu(_),
            TensorStorage::Cpu(_),
        )
    );
    if !all_cpu {
        // Mixed storage. The executor arm normalises before reaching
        // this hook (see `Tensor::ensure_decoded`); a mixed call
        // signals a routing mistake. Bail out silently and let the
        // caller's CPU path handle the operation.
        return false;
    }

    if !gpu_can_run_matmul(m, k, n) {
        return false;
    }

    let mut ran_gpu = false;
    let bytes_needed = m
        .saturating_mul(n)
        .saturating_mul(std::mem::size_of::<f32>());

    try_gpu_with_pool(
        bytes_needed,
        || {
            let gpu_out = cuda_matmul(a, b, m, k, n);
            if gpu_out.shape == out.shape {
                out.as_cpu_slice_mut().clone_from_slice(gpu_out.as_cpu_slice());
                ran_gpu = true;
                GPU_MATMUL_ROUNDTRIP_COUNT.fetch_add(1, Ordering::Relaxed);
                if apx_trace_enabled() {
                    println!("[APX 4.11] GPU MatMul executed (CPU-roundtrip)");
                }
            }
        },
        || {
            // CPU fallback: do nothing here; the caller will see `false`
            // and execute the standard CPU path.
        },
    );

    ran_gpu
}

/// Attempt to execute Linear on GPU: y = x·w + b (optional).
/// Returns `true` if it ran on GPU and `out` contains the result.
pub fn try_gpu_linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>, out: &mut Tensor) -> bool {
    // APX 4.11 MiniFlux: fully disable the GPU path for Linear (with or without
    // bias). Always execute on CPU to correctly record backward and avoid
    // divergences between kernels.
    let _ = (x, w, b, out); // avoid warnings for unused parameters
    return false;

    // Original code intentionally disabled.
}

/// APX 4.13: fused Linear+SiLU execution hook from node IDs in the graph.
/// For simplicity, we only support the case where bias is present; if there is
/// no bias, we fall back to the standard CPU path (Linear + SiLU).
pub unsafe fn fused_linear_silu_gpu(
    x_id: usize,
    w_id: usize,
    b_id: Option<usize>,
    out_id: usize,
    graph: &mut Graph,
    _record_tape: bool,
) {
    let x = graph.nodes[x_id]
        .output
        .as_ref()
        .expect("fused_linear_silu: missing x output")
        .clone();
    let w = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("fused_linear_silu: missing w output")
        .clone();

    let (m, k) = (x.shape[0], x.shape[1]);
    let n = w.shape[1];

    let mut out = Tensor::with_layout(
        vec![m, n],
        0.0,
        x.device,
        crate::tensor::Layout::Contiguous,
        x.dtype,
    );

    if let Some(bid) = b_id {
        let b = graph.nodes[bid]
            .output
            .as_ref()
            .expect("fused_linear_silu: missing bias output")
            .clone();

        cuda_fused_linear_silu(
            &x,
            &w,
            &b,
            &mut out,
            m,
            k,
            n,
        );
        graph.nodes[out_id].set_output(out);
    } else {
        // No bias: use the standard CPU Linear + SiLU path.
        let mut tmp = crate::nn::linear::linear(&x, &w, None);
        tmp = crate::nn::activations::silu(&tmp);
        graph.nodes[out_id].set_output(tmp);
    }
}
