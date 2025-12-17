use crate::gpu::safety::error_codes::{describe, CUDA_SUCCESS};
use crate::gpu::runtime::logging::log;

pub struct GpuSafety;

impl GpuSafety {
    pub fn check(code: i32, context: &str) -> bool {
        if code == CUDA_SUCCESS {
            return true;
        }

        log(&format!(
            "[SAFETY] CUDA ERROR in {} -> {} ({})",
            context,
            describe(code),
            code
        ));

        false
    }

    /// Returns true if fallback to CPU is required
    pub fn should_fallback(code: i32) -> bool {
        match code {
            700 | 703 => true, // launch failed, illegal access, invalid pitch
            _ => false,
        }
    }
}
