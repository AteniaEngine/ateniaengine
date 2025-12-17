use std::time::Instant;

use crate::profiler::gpu_info;

pub struct StepMetrics {
    pub step_time_ms: f32,
    pub forward_ms: f32,
    pub backward_ms: f32,
    pub optim_ms: f32,
    pub gpu_used_mb: u32,
    pub gpu_total_mb: u32,
    pub gpu_util: u32,
}

pub struct Profiler {
    start_step: Instant,
    start_forward: Instant,
    start_backward: Instant,
    start_optim: Instant,

    forward_ms: f32,
    backward_ms: f32,
    optim_ms: f32,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            start_step: Instant::now(),
            start_forward: Instant::now(),
            start_backward: Instant::now(),
            start_optim: Instant::now(),
            forward_ms: 0.0,
            backward_ms: 0.0,
            optim_ms: 0.0,
        }
    }

    pub fn begin_step(&mut self) {
        self.start_step = Instant::now();
    }

    pub fn begin_forward(&mut self) {
        self.start_forward = Instant::now();
    }

    pub fn end_forward(&mut self) {
        self.forward_ms = self.start_forward.elapsed().as_micros() as f32 / 1000.0;
    }

    pub fn begin_backward(&mut self) {
        self.start_backward = Instant::now();
    }

    pub fn end_backward(&mut self) {
        self.backward_ms = self.start_backward.elapsed().as_micros() as f32 / 1000.0;
    }

    pub fn begin_optim(&mut self) {
        self.start_optim = Instant::now();
    }

    pub fn end_optim(&mut self) {
        self.optim_ms = self.start_optim.elapsed().as_micros() as f32 / 1000.0;
    }

    pub fn finalize_step(&mut self) -> StepMetrics {
        let step_time_ms = self.start_step.elapsed().as_micros() as f32 / 1000.0;
        let (gpu_used_mb, gpu_total_mb) = gpu_info::gpu_memory_mb();
        let gpu_util = gpu_info::gpu_utilization();

        StepMetrics {
            step_time_ms,
            forward_ms: self.forward_ms,
            backward_ms: self.backward_ms,
            optim_ms: self.optim_ms,
            gpu_used_mb,
            gpu_total_mb,
            gpu_util,
        }
    }

    pub fn print(&self, step: usize, m: &StepMetrics) {
        println!(
            "[Profiler] Step {:4} | step {:.2} ms | fwd {:.2} ms | bwd {:.2} ms | opt {:.2} ms | VRAM {} / {} MB | GPU {}%",
            step,
            m.step_time_ms,
            m.forward_ms,
            m.backward_ms,
            m.optim_ms,
            m.gpu_used_mb,
            m.gpu_total_mb,
            m.gpu_util
        );
    }
}
