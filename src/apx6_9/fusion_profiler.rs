use std::collections::HashMap;

pub struct FusionRecord {
    pub op_name: String,
    pub unfused_time_us: u64,
    pub fused_time_us: u64,
}

pub struct FusionProfiler {
    pub records: HashMap<String, FusionRecord>,
}

impl FusionProfiler {
    pub fn new() -> Self {
        Self { records: HashMap::new() }
    }

    pub fn record(&mut self, op: &str, unfused: u64, fused: u64) {
        let entry = FusionRecord {
            op_name: op.to_string(),
            unfused_time_us: unfused,
            fused_time_us: fused,
        };
        self.records.insert(op.to_string(), entry);
    }

    pub fn should_use_fused(&self, op: &str) -> Option<bool> {
        let rec = self.records.get(op)?;
        let unf = rec.unfused_time_us as f64;
        let fus = rec.fused_time_us as f64;

        if fus < unf * 0.90 {
            Some(true)
        } else if fus > unf * 1.10 {
            Some(false)
        } else {
            None
        }
    }
}

use std::sync::Mutex;
use std::sync::OnceLock;

static GLOBAL_FUSION_PROFILER: OnceLock<Mutex<FusionProfiler>> = OnceLock::new();

pub fn fusion_profiler() -> &'static Mutex<FusionProfiler> {
    GLOBAL_FUSION_PROFILER.get_or_init(|| Mutex::new(FusionProfiler::new()))
}
