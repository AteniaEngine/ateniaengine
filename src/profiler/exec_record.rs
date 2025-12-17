use crate::gpu::fingerprint::KernelFingerprint;
use crate::profiler::heatmap::GpuHeatmap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub id: u64,
    pub kernel_name: String,
    pub duration_ms: f32,
    pub fingerprint: KernelFingerprint,
    pub timestamp_ms: u128,
    pub tags: Vec<&'static str>,
}

impl ExecutionRecord {
    pub fn new(
        id: u64,
        kernel_name: impl Into<String>,
        duration_ms: f32,
        fingerprint: KernelFingerprint,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let tags = fingerprint.tags.all();

        Self {
            id,
            kernel_name: kernel_name.into(),
            duration_ms,
            fingerprint,
            timestamp_ms: ts,
            tags,
        }
    }
}

#[derive(Default)]
pub struct ExecutionRecorder {
    pub records: Vec<ExecutionRecord>,
    pub heatmap: GpuHeatmap,
    next_id: u64,
}

impl ExecutionRecorder {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            heatmap: GpuHeatmap::new(),
            next_id: 0,
        }
    }

    pub fn record(
        &mut self,
        kernel_name: impl Into<String>,
        duration_ms: f32,
        fp: KernelFingerprint,
    ) {
        let name: String = kernel_name.into();
        let rec = ExecutionRecord::new(self.next_id, name.clone(), duration_ms, fp);
        self.records.push(rec);
        self.heatmap.record(&name, duration_ms);
        self.next_id += 1;
    }
}
