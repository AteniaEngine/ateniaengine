#[derive(Debug, Clone, Copy)]
pub struct GpuTunerSample {
    pub size: usize,
    pub cpu_time_us: u64,
    pub gpu_time_us: u64,
}

pub struct GpuAutoTuner {
    pub samples: Vec<GpuTunerSample>,
}

impl GpuAutoTuner {
    pub fn new() -> Self {
        Self { samples: vec![] }
    }

    pub fn record(&mut self, size: usize, cpu_t: u64, gpu_t: u64) {
        self.samples.push(GpuTunerSample { size, cpu_time_us: cpu_t, gpu_time_us: gpu_t });
    }

    pub fn decide(&self, size: usize) -> GpuDecision {
        if self.samples.len() < 3 {
            return GpuDecision::Cpu; // fallback seguro
        }

        let mut cpu_avg = 0.0;
        let mut gpu_avg = 0.0;
        let mut count = 0.0;

        for s in &self.samples {
            if (s.size as i32 - size as i32).abs() < (size/2) as i32 {
                cpu_avg += s.cpu_time_us as f64;
                gpu_avg += s.gpu_time_us as f64;
                count += 1.0;
            }
        }

        if count < 1.0 { return GpuDecision::Cpu; }

        cpu_avg /= count;
        gpu_avg /= count;

        if gpu_avg < cpu_avg {
            GpuDecision::Gpu
        } else {
            GpuDecision::Cpu
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuDecision {
    Cpu,
    Gpu,
}
