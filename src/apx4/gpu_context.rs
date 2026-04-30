use once_cell::sync::Lazy;
use std::sync::Mutex;

#[derive(Debug)]
pub struct ApxGpuContext {
    pub available: bool,
    pub device_count: usize,
}

impl ApxGpuContext {
    pub fn new() -> Self {
        // **M6.c.7 fix 4** — REVERTED to hardcoded false.
        //
        // M6.b flipped this to runtime detect (calling
        // `cuda::cuda_available()`). That was the wrong
        // surface to flip. This `gpu_available` flag gates
        // the **legacy apx4 dispatch path**
        // (`apx4::gpu_dispatch::dispatch_matmul_gpu` /
        //  `apx4::gpu_kernels::gpu_matmul`), which the
        // `Graph::execute_single` MatMul arm reaches when
        // `try_gpu_matmul` returns false (line 3425 in
        // `amg/graph.rs`).
        //
        // The legacy `gpu_matmul` ABI has its own implicit
        // host↔device transfer per call, with no
        // residency awareness. Activating it for 13B
        // matmuls (270 MB FFN-down weight) costs
        // ~750 ms/call — 20× slower than the M4.8
        // matrixmultiply CPU path. The M6.c live smoke
        // measured 211 s/token under `ATENIA_GPU=0`
        // because the kill-switch only protected
        // `try_gpu_matmul` — the legacy apx4 path stayed
        // active.
        //
        // M6's actual GPU activation lives in
        // `cuda::cuda_available()` (cached, M6.c.7 fix 1)
        // consumed by [`crate::gpu::backend::CudaBackend`]
        // and [`crate::gpu::dispatch::hooks::try_gpu_matmul`].
        // The legacy apx4 path stays disabled — it was
        // never the load-bearing surface; M4.7's
        // `try_gpu_matmul` (M4.7.3.a) is the authoritative
        // GPU dispatch.
        //
        // Future re-activation of the legacy path requires
        // a separate sub-phase that adds shape gating, a
        // pool / non-pooled router, and the same kill-
        // switch protection `try_gpu_matmul` got in M6.b.
        // Not on the M6 critical path; track in M6.f or v21
        // if anyone ever needs it.
        Self {
            available: false,
            device_count: 0,
        }
    }
}

pub static GPU_CONTEXT: Lazy<Mutex<ApxGpuContext>> =
    Lazy::new(|| Mutex::new(ApxGpuContext::new()));

pub fn gpu_available() -> bool {
    GPU_CONTEXT.lock().unwrap().available
}
