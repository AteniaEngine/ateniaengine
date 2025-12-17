use std::time::Instant;

use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::apx6_4::matmul_4x8_avx2;
use atenia_engine::apx6_7::runtime_profile::RuntimeProfile;
use atenia_engine::apx6_7::auto_bench::run_initial_bench;
use atenia_engine::tensor::Device;

fn baseline_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let ctx = DeviceContext::new(Device::CPU);
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn micro_6_4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

#[test]
fn apx_6_7_benchmark_profile_logging() {
    let sizes = [128usize, 256, 512, 1024];

    let mut profile = RuntimeProfile::new();
    run_initial_bench(&mut profile);

    for &s in &sizes {
        let m = s;
        let k = s;
        let n = s;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.173).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.293).cos()).collect();

        let (_, base_us) = baseline_matmul(&a, &b, m, k, n);
        let (_, micro_us) = micro_6_4(&a, &b, m, k, n);

        println!(
            "[APX 6.7 ABL] size={} baseline={}us micro64={}us selected={}",
            s,
            base_us,
            micro_us,
            profile.best_for(s).unwrap_or_else(|| "baseline".to_string()),
        );
    }
}
