// M6.c.1 — vendor-neutral Backend trait + CudaBackend
// impl. M6's load-bearing abstraction; v22 adds a wgpu peer.
pub mod backend;
// M6.c.2 — pure-function planner that picks resident vs
// streamed layers given a VRAM budget and per-layer cost.
pub mod residency_planner;

pub mod nvrtc;
pub mod loader;
pub mod memory;
pub mod runtime;
pub mod launcher;
pub mod translator;
pub mod ops;
pub mod tensor;
pub mod safety;
pub mod arch;
pub mod autodiff;
pub mod linker;
pub mod kernel;
pub mod planning;
pub mod profiler;
pub mod autotuner;
pub mod device;
pub mod fingerprint;
pub mod tags;
pub mod utils;
pub mod dispatch;

pub use tags::*;

use std::sync::OnceLock;

use crate::gpu::memory::GpuMemoryEngine;

static GPU_ENGINE: OnceLock<Option<GpuMemoryEngine>> = OnceLock::new();

/// Returns the process-wide shared [`GpuMemoryEngine`], or `None` if CUDA
/// is unavailable in this environment.
///
/// The engine is initialized lazily on the first call. If initialization
/// fails (no driver, no device), the result is cached as `None` for the
/// remainder of the process and subsequent calls return `None` without
/// retrying.
///
/// This is the recommended accessor for VRAM-owning types (`TensorGPU`,
/// and from M3-d.2 onward `TensorStorage::Cuda`). Prefer it over
/// [`GpuMemoryEngine::new`], which constructs an independent CUDA context
/// and is retained only as an escape hatch for isolated tests.
pub fn gpu_engine() -> Option<&'static GpuMemoryEngine> {
    GPU_ENGINE
        .get_or_init(|| GpuMemoryEngine::new().ok())
        .as_ref()
}
