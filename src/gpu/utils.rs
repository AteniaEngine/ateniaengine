//! Vendor-neutral GPU utilities.
//!
//! This module hosts helpers that the APX layer used to import directly
//! from `crate::cuda::*`. Placing them under `crate::gpu` encapsulates
//! the vendor-specific detail inside the gpu abstraction layer; callers
//! upstream (APX modules, graph, tests) import from `crate::gpu::utils`
//! or via the pass-through re-export left at the original location for
//! backward compatibility.

use crate::cuda::cuda_available;

/// **M6.c.7 fix 5** — disabled by default.
///
/// `gpu_enabled()` gates the legacy [`crate::amg::graph::Graph::exec_gpu_segment`]
/// path (`src/gpu/dispatch/executor.rs`). The executor's
/// MatMul arm reaches that path for every node that is the
/// start of a `GpuSegment` in `Graph::gpu_plan` — and on
/// the Llama hot path **every MatMul becomes its own
/// 1-node `GpuSegment`** (see `amg/graph.rs:3300` comment).
/// Pre-M6.c.7 fix 5 this routed every 13B matmul through
/// `cuda_matmul_inplace` → fallback to `cuda_matmul`
/// (pool-routed, 64 MiB block ceiling). The 270 MB FFN-class
/// weight forced the pool to grow per-call via
/// `cudaMalloc(64 MiB)`, paying ~100 ms of driver overhead
/// per matmul × 360 matmuls/token = ~36 s of pure
/// orchestration overhead per generated token. The smoke
/// test on Llama 2 13B Chat with `ATENIA_GPU=0` (kill-switch
/// for `try_gpu_matmul` only) measured 285 s/token vs the
/// M5.f.a baseline of 14 s/token — a regression that fix
/// 1 (cache nvidia-smi probe) addressed but did not close,
/// because the per-matmul `cudaMalloc` cost survives the
/// cache.
///
/// M5 baseline avoided this because `cuda_available()`
/// itself was uncached (per call spawned nvidia-smi),
/// AND the executor short-circuited some segments
/// elsewhere. Either way: the legacy `exec_gpu_segment`
/// path was **never load-bearing** in any v20 milestone —
/// M4.7's residency-aware `try_gpu_matmul` is the
/// authoritative GPU dispatch.
///
/// Disabling this flag:
///   - Disables `exec_gpu_segment` for every node (the
///     `if !gpu_enabled() { return; }` guard at the top
///     of the function fires).
///   - Does NOT affect `try_gpu_matmul` — that path uses
///     `cuda::cuda_available()` directly, independent of
///     this flag. M4.7.3 residency dispatch + M6.c.4
///     mixed-storage path remain fully active.
///   - Does NOT affect `WeightStore::upload_resident_layers`
///     — that consumes `CudaBackend::available_vram_bytes()`
///     directly via the M6.c.1 Backend trait.
///   - Does NOT affect `gpu_kernels` registration in
///     other APX modes (those have their own gates).
///
/// Future re-activation requires `exec_gpu_segment` and
/// `cuda_matmul_inplace`'s fallback path to grow:
///   - A shape-size gate (reject when max_per_alloc
///     exceeds pool block).
///   - The `ATENIA_GPU=0` kill-switch.
///   - Decision: do we even want it? `try_gpu_matmul`
///     covers every dispatch surface that matters. Track
///     in a future M6+ sub-phase if anyone needs it.
pub fn gpu_enabled() -> bool {
    // Suppress unused-import without changing the
    // public surface — `cuda_available` stays available
    // for direct callers (`try_gpu_matmul` etc.).
    let _ = cuda_available;
    false
}

pub fn log_gpu(msg: &str) {
    if !crate::apx_is_silent() && std::env::var("APX_TRACE").is_ok() {
        eprintln!("[APX 4.3 GPU] {msg}");
    }
}
