use once_cell::sync::Lazy;
use std::sync::Mutex;

#[derive(Debug)]
pub struct ApxGpuContext {
    pub available: bool,
    pub device_count: usize,
}

impl ApxGpuContext {
    pub fn new() -> Self {
        // Placeholder until real CUDA integration (APX 4.0).
        Self {
            available: false,
            device_count: 0,
        }
    }
}

pub static GPU_CONTEXT: Lazy<Mutex<ApxGpuContext>> =
    Lazy::new(|| Mutex::new(ApxGpuContext::new()));

pub fn gpu_available() -> bool {
    GPU_CONTEXT.lock().unwrap().available
}
