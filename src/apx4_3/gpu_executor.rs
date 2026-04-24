//! GPU executor dispatch — moved to `src/gpu/dispatch/executor.rs`.
//!
//! This module previously contained the `impl Graph` methods for GPU
//! segment execution (`exec_gpu_segment`, `exec_gpu_matmul`,
//! `exec_gpu_add`, `exec_gpu_mul`, `exec_gpu_linear`,
//! `exec_gpu_fused_linear_silu`). As part of the vendor-neutrality
//! invariant cleanup (last M3 debt in the cuda-leak cleanup arc,
//! following the M3-d precedent with `gpu_hooks.rs`), the
//! `crate::cuda::*` imports were isolated to `src/gpu/`.
//!
//! The methods are still accessible via the `Graph` impl — moving the
//! code file does not change the public API, because `impl Graph`
//! blocks are resolved by path on the `Graph` type rather than by the
//! module containing the `impl`. Callers (`amg::graph::execute`,
//! `apx4_7::fused_pairs`) are unaffected.
//!
//! See `src/gpu/dispatch/executor.rs` for the actual code.
