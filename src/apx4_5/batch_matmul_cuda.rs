//! Pass-through to the vendor-neutral home of the batch MatMul
//! dispatch wrapper.
//!
//! The implementation lives in [`crate::gpu::ops::batch_matmul_dispatch`].
//! This file is kept as a re-export so existing
//! `use crate::apx4_5::batch_matmul_cuda::batch_matmul_cuda` imports
//! continue to compile without changes. New code should import from
//! `crate::gpu::ops::batch_matmul_dispatch` directly.

pub use crate::gpu::ops::batch_matmul_dispatch::*;
