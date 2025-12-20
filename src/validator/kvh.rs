use crate::profiler::exec_record::ExecutionRecord;

#[derive(Debug)]
pub struct ValidationResult {
    pub ok: bool,
    pub max_diff: f32,
    pub avg_diff: f32,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self { ok: true, max_diff: 0.0, avg_diff: 0.0 }
    }
}

pub struct KernelValidationHarness {
    pub enabled: bool,
    pub atol: f32,
    pub rtol: f32,
}

impl KernelValidationHarness {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            atol: 1e-5,
            rtol: 1e-5,
        }
    }

    pub fn validate<CT, GT>(
        &self,
        _rec: &ExecutionRecord,
        cpu_out: &CT,
        gpu_out: &GT,
    ) -> ValidationResult
    where
        CT: AsRef<[f32]>,
        GT: AsRef<[f32]>,
    {
        if !self.enabled {
            return ValidationResult::success();
        }

        let cpu = cpu_out.as_ref();
        let gpu = gpu_out.as_ref();

        let mut max_diff = 0.0;
        let mut sum_diff = 0.0;

        for (a, b) in cpu.iter().zip(gpu.iter()) {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff += diff;
        }

        let avg_diff = if cpu.is_empty() { 0.0 } else { sum_diff / cpu.len() as f32 };

        // For this first version, consider ok if max_diff and avg_diff are small.
        let ok = max_diff <= self.atol * 10.0 && avg_diff <= self.atol * 10.0;

        ValidationResult { ok, max_diff, avg_diff }
    }
}
