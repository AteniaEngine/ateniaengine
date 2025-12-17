use std::time::Instant;

use crate::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use crate::apx5::kernel_planner::KernelTarget;
use crate::apx6_4::matmul_4x8_avx2;
use crate::tensor::Device;

use super::runtime_profile::{KernelPerf, RuntimeProfile};

pub fn run_initial_bench(profile: &mut RuntimeProfile) {
    let sizes = [128usize, 256, 512, 1024];

    for &s in &sizes {
        let m = s;
        let k = s;
        let n = s;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.151).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.313).cos()).collect();

        // Baseline APX 3.8
        let ctx = DeviceContext::new(Device::CPU);
        let mut out_base = vec![0.0f32; m * n];
        let t0 = Instant::now();
        dispatch_matmul_apx3_8(&a, &b, &mut out_base, m, k, n, &ctx);
        let baseline_us = t0.elapsed().as_micros();

        // Microkernel 6.4 (4x8 AVX2)
        let mut out_micro = vec![0.0f32; m * n];
        let t1 = Instant::now();
        matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out_micro.as_mut_ptr(), m, k, n);
        let micro64_us = t1.elapsed().as_micros();

        // Regla de decisión: 5% más rápido que baseline para elegir micro64.
        let threshold = baseline_us * 95 / 100;
        let selected = if micro64_us < threshold {
            "micro64".to_string()
        } else {
            "baseline".to_string()
        };

        let perf = KernelPerf {
            size: s,
            baseline_us,
            micro64_us,
            selected,
        };

        profile.record(perf);
    }
}

pub fn estimate_best_kernel(size: usize, profile: &RuntimeProfile) -> Option<KernelTarget> {
    let selected = profile.best_for(size)?;
    let target = if selected == "micro64" {
        // Usar target optimizado CPU (AVX2) ya existente.
        KernelTarget::CpuFastAvx2
    } else {
        KernelTarget::Cpu
    };
    Some(target)
}
