use once_cell::sync::Lazy;
use std::sync::Mutex;

#[derive(Debug)]
pub struct ApxGpuContext {
    pub available: bool,
    pub device_count: usize,
}

impl ApxGpuContext {
    pub fn new() -> Self {
        // **M6.b.1** — runtime CUDA detection.
        //
        // Pre-M6.b this was hardcoded `available: false`
        // (M4.7 decision 35 carryover, "placeholder until
        // real CUDA integration"). M6.b lifts the gate by
        // delegating to [`crate::cuda::cuda_available`],
        // which probes the driver via `nvidia-smi`. On boxes
        // without NVIDIA hardware / driver, the probe fails
        // and the legacy CPU dispatch path
        // (`apx4::gpu_dispatch::dispatch_matmul_gpu`) keeps
        // the existing fallback semantics — bit-exact
        // identical to the pre-M6.b build.
        //
        // The kill-switch `ATENIA_GPU=0` lives one level
        // higher (in `gpu/dispatch/hooks.rs::try_gpu_matmul`)
        // so operators can disable GPU dispatch without
        // recompiling.
        let available = crate::cuda::cuda_available();
        Self {
            available,
            device_count: if available { 1 } else { 0 },
        }
    }
}

pub static GPU_CONTEXT: Lazy<Mutex<ApxGpuContext>> =
    Lazy::new(|| Mutex::new(ApxGpuContext::new()));

pub fn gpu_available() -> bool {
    GPU_CONTEXT.lock().unwrap().available
}
