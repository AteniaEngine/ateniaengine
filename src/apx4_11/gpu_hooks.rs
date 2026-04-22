//! Pass-through re-export. The actual implementation lives in
//! [`crate::gpu::dispatch::hooks`]. This file preserves the
//! `apx4_11::gpu_hooks` module path for backward compatibility with
//! existing callers in `amg::graph` and elsewhere. New code should
//! import from `crate::gpu::dispatch::hooks` directly.

pub use crate::gpu::dispatch::hooks::*;
