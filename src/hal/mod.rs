//! Hardware Abstraction Layer for Atenia Engine.
//! Targets CUDA, ROCm, Metal, Vulkan, and oneAPI backends.

pub mod cuda;
pub mod traits;

/// Attempts to detect whether CUDA is available on the host.
pub fn detect_cuda() -> bool {
    cuda::is_available()
}
