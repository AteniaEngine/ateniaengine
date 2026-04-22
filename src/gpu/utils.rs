//! Vendor-neutral GPU utilities.
//!
//! This module hosts helpers that the APX layer used to import directly
//! from `crate::cuda::*`. Placing them under `crate::gpu` encapsulates
//! the vendor-specific detail inside the gpu abstraction layer; callers
//! upstream (APX modules, graph, tests) import from `crate::gpu::utils`
//! or via the pass-through re-export left at the original location for
//! backward compatibility.

use crate::cuda::cuda_available;

pub fn gpu_enabled() -> bool {
    cuda_available()
}

pub fn log_gpu(msg: &str) {
    if !crate::apx_is_silent() && std::env::var("APX_TRACE").is_ok() {
        eprintln!("[APX 4.3 GPU] {msg}");
    }
}
