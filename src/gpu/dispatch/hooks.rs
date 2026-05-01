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
    matmul::cuda_matmul_non_pooled,
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
/// M4.7.6.d — legacy `dispatch_matmul_gpu` (apx4) path counter.
/// See `apx4::gpu_dispatch::record_legacy_gpu_matmul` for the
/// rationale. Incremented every time `gpu_matmul` (apx4) is
/// called via the legacy dispatch path — the path Llama 2 13B's
/// hot path actually uses because the 64 MiB pool block size
/// excludes its 100-270 MB weight tensors from the M4.7.3
/// residency-aware code path.
static GPU_MATMUL_LEGACY_COUNT: AtomicUsize = AtomicUsize::new(0);
/// **M6 step 2b** — non-pooled CPU-roundtrip path counter. Fires
/// when `try_gpu_matmul`'s all-Cpu branch dispatches an oversize
/// MatMul (any single buffer > `DEFAULT_BLOCK_SIZE = 64 MiB`)
/// directly through [`crate::cuda::matmul::cuda_matmul_non_pooled`],
/// bypassing the pool. This is the path the Llama 2 13B forward is
/// expected to take after step 2b — every Q/K/V/O proj
/// (5120×5120 = 100 MB F32) and FFN gate/up/down
/// (5120×13824 = 270 MB F32) lives above the 64 MiB ceiling.
static GPU_MATMUL_NON_POOLED_COUNT: AtomicUsize = AtomicUsize::new(0);

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

/// M4.7.6.d — count of MatMul invocations that took the legacy
/// `dispatch_matmul_gpu` → `gpu_matmul` path (apx4). This path
/// handles MatMul shapes whose weight tensor exceeds the
/// `DEFAULT_BLOCK_SIZE = 64 MiB` pool block — the M4.7.3
/// residency-aware path is unreachable for those shapes today,
/// so the legacy path is the only GPU route for 13B-class
/// models.
pub fn gpu_matmul_legacy_count() -> usize {
    GPU_MATMUL_LEGACY_COUNT.load(Ordering::Relaxed)
}

/// **M6 step 2b** — count of MatMul invocations that took the
/// non-pooled CPU-roundtrip path inside `try_gpu_matmul`. Use this
/// counter to verify the Llama 2 13B hot path actually reached the
/// new dispatch surface (expected `> 0` after a forward pass).
pub fn gpu_matmul_non_pooled_count() -> usize {
    GPU_MATMUL_NON_POOLED_COUNT.load(Ordering::Relaxed)
}

/// M4.7.6.d — union of the GPU MatMul counters.
/// `total > 0` answers "did GPU MatMul fire at all on this
/// model" without forcing the caller to know which dispatch
/// path was taken. Use this in tests / demos that should be
/// agnostic to whether the M4.7.3 residency path, the legacy
/// apx4 path, or the M6.2b non-pooled path served the work.
pub fn gpu_matmul_total_count() -> usize {
    GPU_MATMUL_RESIDENT_COUNT.load(Ordering::Relaxed)
        + GPU_MATMUL_ROUNDTRIP_COUNT.load(Ordering::Relaxed)
        + GPU_MATMUL_LEGACY_COUNT.load(Ordering::Relaxed)
        + GPU_MATMUL_NON_POOLED_COUNT.load(Ordering::Relaxed)
}

/// M4.7.6.d — internal increment used by
/// `apx4::gpu_dispatch::dispatch_matmul`. `pub` so the apx4
/// module can call it without a circular dep on hooks.rs's
/// private statics; the function name is more specific than
/// the public reads to discourage accidental external bumps.
pub fn increment_legacy_gpu_matmul_counter() {
    GPU_MATMUL_LEGACY_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Simple heuristic to decide whether it is worth using the CUDA MatMul kernel.
///
/// **M6 step 2b — pool size gate (G5) lifted.** Up to step 2a, this
/// function rejected any MatMul whose largest operand exceeded
/// `DEFAULT_BLOCK_SIZE = 64 MiB`. That gate kept every Llama 2 13B
/// weight tensor (Q/K/V/O proj 100 MB F32, FFN gate/up/down 270 MB
/// F32, LM head 655 MB F32) on the CPU dispatcher. Step 2b lifts
/// the gate at this layer: oversize matmuls now reach
/// [`try_gpu_matmul`], which routes them through the non-pooled
/// path ([`crate::cuda::matmul::cuda_matmul_non_pooled`]) instead
/// of the pool. Sub-block-size matmuls keep the historical pool
/// path bit-exact.
///
/// `cuda_available()` is consulted before any size math, so on
/// hosts without a CUDA driver the function bails before anything
/// else and the caller falls through to CPU. The cache from M6
/// step 1 (`cuda::cuda_available`) means repeated calls cost a
/// single atomic read.
pub fn gpu_can_run_matmul(m: usize, k: usize, n: usize) -> bool {
    // Require a minimum amount of work to amortize host<->device overhead.
    let ops = m.saturating_mul(k).saturating_mul(n);
    if ops <= 256 {
        return false;
    }

    // Driver gate. Cached behind `OnceLock<bool>` since M6 step 1
    // (`cuda/mod.rs:cuda_available`), so this is one atomic load
    // after the first call. Must remain the last bail before
    // `try_gpu_matmul`'s shape-class routing — see S3 in
    // `INVESTIGATION_M6_DEEP.md`.
    if !cuda::cuda_available() {
        return false;
    }

    // M6 step 2b: the historical `max_per_alloc > DEFAULT_BLOCK_SIZE`
    // bail used to live here. It now lives inside `try_gpu_matmul`
    // as a shape-class router (pool path vs non-pooled path), not as
    // a refusal. Oversize MatMuls return `true` from this function
    // so the executor calls `try_gpu_matmul`, which then picks the
    // right sub-path.

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

    // **M6 step 2b** — shape-class router for the all-Cpu branch.
    //
    // Decision: does any single buffer (A, B, out) exceed the
    // 64 MiB pool block size?
    //   - **Yes** → non-pooled path. Direct `cuda_malloc`/
    //     `cudaMemcpy` via [`cuda_matmul_non_pooled`]; bypasses the
    //     pool and the `cuda_matmul_inplace` mixed fallback (whose
    //     clone-and-recurse behaviour, S5 in
    //     `INVESTIGATION_M6_DEEP.md`, would silently double host
    //     memory). On `None` we return `false` so the caller falls
    //     through to CPU AVX2.
    //   - **No** → existing pool path through `try_gpu_with_pool`,
    //     bit-exact with the M5.f.a behaviour for sub-block-size
    //     MatMuls.
    let f32_size = std::mem::size_of::<f32>();
    let a_bytes = m.saturating_mul(k).saturating_mul(f32_size);
    let b_bytes = k.saturating_mul(n).saturating_mul(f32_size);
    let out_bytes = m.saturating_mul(n).saturating_mul(f32_size);
    let max_per_alloc = a_bytes.max(b_bytes).max(out_bytes);

    if max_per_alloc > crate::apx4_12::DEFAULT_BLOCK_SIZE {
        // Non-pooled path. The all_cpu branch above guarantees both
        // operand storages are `Cpu`, so `as_cpu_slice` is safe.
        let a_slice = a.as_cpu_slice();
        let b_slice = b.as_cpu_slice();
        match cuda_matmul_non_pooled(a_slice, b_slice, m, k, n) {
            Some(gpu_out) => {
                if gpu_out.shape == out.shape {
                    out.as_cpu_slice_mut()
                        .clone_from_slice(gpu_out.as_cpu_slice());
                    GPU_MATMUL_NON_POOLED_COUNT.fetch_add(1, Ordering::Relaxed);
                    if apx_trace_enabled() {
                        println!("[APX M6.2b] GPU MatMul executed (non-pooled)");
                    }
                    return true;
                }
                false
            }
            None => false,
        }
    } else {
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

#[cfg(test)]
mod m6_step_2b_routing_tests {
    //! Routing-only tests for the M6 step 2b shape-class router in
    //! `try_gpu_matmul`. These tests assert that the right counter
    //! fires for a given shape class — they do not validate
    //! numerical correctness (that is covered by the per-path
    //! tests: `cuda_matmul_residency_test`,
    //! `cuda_matmul_non_pooled_tests` in `cuda::matmul`, and the
    //! existing pool-path coverage).
    //!
    //! Both tests require a working CUDA driver (residency-aware
    //! kernels and the non-pooled `cuda_malloc` path both panic on
    //! a stub driver). The `cuda_available()` skip mirrors the
    //! convention used in `tests/cuda_matmul_residency_test.rs`.
    //!
    //! # Why a module-level Mutex
    //!
    //! Both tests measure deltas on the process-wide counters
    //! (`GPU_MATMUL_NON_POOLED_COUNT`, `GPU_MATMUL_ROUNDTRIP_COUNT`).
    //! Rust's default test harness runs unit tests in parallel; if
    //! both tests run concurrently, each one's "after" snapshot
    //! observes increments from the other and the assertions fire
    //! on contamination, not on a real routing bug. Serialising
    //! through a single `Mutex` is the contained fix — no extra
    //! dev-dependency, no refactor of the production counters, and
    //! the lock scope is bounded to the test functions in this
    //! module (other counter consumers are unaffected). Locking is
    //! acquire-on-entry / release-on-drop; if a test panics the
    //! `PoisonError` is unwrapped and ignored so subsequent tests
    //! still run.
    use std::sync::Mutex;

    use super::{
        gpu_matmul_non_pooled_count, gpu_matmul_roundtrip_count, try_gpu_matmul,
    };
    use crate::cuda::cuda_available;
    use crate::tensor::tensor::Tensor;

    static COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Oversize shape (single buffer > 64 MiB) must take the
    /// non-pooled path, not the pool path. Uses the smallest
    /// shape that exceeds `DEFAULT_BLOCK_SIZE = 64 MiB` to keep
    /// the test fast while still triggering the router.
    ///
    /// 4097² × 4 bytes = 67.14 MB → all three buffers (A, B, out)
    /// land just above the 64 MiB ceiling, so `max_per_alloc` is
    /// strictly greater and the router has to pick the non-pooled
    /// path.
    #[test]
    fn oversize_shape_routes_to_non_pooled() {
        let _guard = COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        let dim = 4097_usize;
        let a = Tensor::new_cpu(vec![dim, dim], vec![0.0_f32; dim * dim]);
        let b = Tensor::new_cpu(vec![dim, dim], vec![0.0_f32; dim * dim]);
        let mut out = Tensor::new_cpu(vec![dim, dim], vec![0.0_f32; dim * dim]);

        let np_before = gpu_matmul_non_pooled_count();
        let rt_before = gpu_matmul_roundtrip_count();

        let ran = try_gpu_matmul(&a, &b, &mut out);
        assert!(ran, "try_gpu_matmul must succeed on a 4097×4097 f32 shape");

        let np_after = gpu_matmul_non_pooled_count();
        let rt_after = gpu_matmul_roundtrip_count();

        assert_eq!(
            np_after - np_before,
            1,
            "oversize shape must increment non-pooled counter by exactly 1"
        );
        assert_eq!(
            rt_after - rt_before,
            0,
            "oversize shape must NOT touch the pool roundtrip counter"
        );
    }

    /// Sub-block-size shape (every buffer ≤ 64 MiB) must take the
    /// existing pool path. The non-pooled counter must NOT move.
    /// 64×64 × 64×64 = 16 KB per buffer, well under the ceiling.
    #[test]
    fn small_shape_stays_on_pool_path() {
        let _guard = COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        let dim = 64_usize;
        // Deterministic small values so the kernel runs without
        // numeric warnings; we are not asserting numerical results
        // here, only counter movement.
        let a_data: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.002).collect();
        let a = Tensor::new_cpu(vec![dim, dim], a_data);
        let b = Tensor::new_cpu(vec![dim, dim], b_data);
        let mut out = Tensor::new_cpu(vec![dim, dim], vec![0.0_f32; dim * dim]);

        let np_before = gpu_matmul_non_pooled_count();
        let rt_before = gpu_matmul_roundtrip_count();

        let ran = try_gpu_matmul(&a, &b, &mut out);
        assert!(ran, "try_gpu_matmul must succeed on a 64×64 f32 shape");

        let np_after = gpu_matmul_non_pooled_count();
        let rt_after = gpu_matmul_roundtrip_count();

        assert_eq!(
            np_after - np_before,
            0,
            "small shape must NOT touch the non-pooled counter"
        );
        assert_eq!(
            rt_after - rt_before,
            1,
            "small shape must increment the pool roundtrip counter by exactly 1"
        );
    }
}
