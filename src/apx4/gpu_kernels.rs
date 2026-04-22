//! Pass-through to the vendor-neutral home of the slice-based CUDA
//! MatMul wrapper.
//!
//! The implementation lives in [`crate::gpu::ops::matmul_wrapper`].
//! This file is kept as a re-export so existing
//! `use crate::apx4::gpu_kernels::gpu_matmul` imports continue to
//! compile without changes. New code should import from
//! `crate::gpu::ops::matmul_wrapper` directly.

pub use crate::gpu::ops::matmul_wrapper::*;
