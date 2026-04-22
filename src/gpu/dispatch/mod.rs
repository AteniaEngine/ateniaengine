//! Vendor-neutral GPU dispatch hooks.
//!
//! Hosts the per-op "should this run on GPU?" decision logic and the
//! graph-level fused execution hooks. Originally scattered under
//! `crate::apx*_*::gpu_*` module paths; consolidated here so the
//! `crate::cuda::*` dependency stays inside the `src/gpu/` layer.

pub mod hooks;
