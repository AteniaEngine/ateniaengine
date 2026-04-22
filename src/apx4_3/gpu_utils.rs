//! Pass-through to the vendor-neutral home of these helpers.
//!
//! The implementation lives in [`crate::gpu::utils`]. This file is
//! kept as a re-export so existing `crate::apx4_3::{gpu_enabled,
//! log_gpu}` / `use crate::apx4_3::gpu_utils::*` imports continue to
//! compile without changes. New code should import from
//! `crate::gpu::utils` directly.

pub use crate::gpu::utils::*;
