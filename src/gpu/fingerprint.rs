use std::time::{SystemTime, UNIX_EPOCH};
use crate::gpu::tags::KernelTags;

#[derive(Debug, Clone)]
pub struct KernelFingerprint {
    pub ptx_hash: u64,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub param_bytes: usize,
    pub device_sm_count: u32,
    pub device_warp_size: u32,
    pub timestamp_ms: u128,
    pub exec_path: String, // "GPU", "CPU-FALLBACK", "SAFE-CUBIN", etc.
    pub tags: KernelTags,
}

impl KernelFingerprint {
    pub fn new(
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        param_bytes: usize,
        device_sm_count: u32,
        device_warp_size: u32,
        exec_path: impl Into<String>,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        Self {
            ptx_hash,
            grid,
            block,
            shared_mem,
            param_bytes,
            device_sm_count,
            device_warp_size,
            timestamp_ms: ts,
            exec_path: exec_path.into(),
            tags: KernelTags::new(),
        }
    }
}
