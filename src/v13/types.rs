use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    OneApi,
    Vulkan,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct CpuCaps {
    pub physical_cores: Option<u32>,
    pub logical_cores: Option<u32>,
    pub simd: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GpuCaps {
    pub id: String,
    pub backend: BackendKind,
    pub vram_total_bytes: Option<u64>,
    pub vram_free_bytes: Option<u64>,
    pub bandwidth_gbps_est: Option<f32>,
    pub compute_score_est: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct RamCaps {
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SsdCaps {
    pub cache_dir: String,
    pub read_mb_s_est: Option<f32>,
    pub write_mb_s_est: Option<f32>,
    pub latency_ms_est: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct ReliabilityStats {
    pub ok_count: u64,
    pub fail_count: u64,
    pub last_error: Option<String>,
    pub last_error_epoch_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct PressureSnapshot {
    pub ram_pressure: Option<f32>,
    pub vram_pressure: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct GlobalHardwareSnapshot {
    pub timestamp_epoch_ms: u64,
    pub cpu: CpuCaps,
    pub gpus: Vec<GpuCaps>,
    pub ram: RamCaps,
    pub ssd: SsdCaps,
    pub reliability_by_device: HashMap<String, ReliabilityStats>,
    pub pressure: PressureSnapshot,
}
