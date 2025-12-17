use std::time::Instant;

use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::kernels::matmul_tiled_cpu; // APX 6.1
use atenia_engine::apx6_3::tiled_avx2::matmul_tiled_avx2; // APX 6.3
use atenia_engine::tensor::Device;

fn baseline_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let ctx = DeviceContext::new(Device::CPU);
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn matmul_6_1(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    matmul_tiled_cpu::matmul_tiled_cpu(a, b, &mut out, m, k, n);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn matmul_6_3(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    matmul_tiled_avx2(a, b, &mut out, m, k, n);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn bench_size(m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.123).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.321).cos()).collect();

    let (out_base, t_base) = baseline_matmul(&a, &b, m, k, n);
    let (out_61, t_61) = matmul_6_1(&a, &b, m, k, n);
    let (out_63, t_63) = matmul_6_3(&a, &b, m, k, n);

    for i in 0..out_base.len() {
        assert!((out_base[i] - out_61[i]).abs() < 1e-3);
        assert!((out_base[i] - out_63[i]).abs() < 1e-3);
    }

    let speedup_61 = t_base as f64 / t_61 as f64;
    let speedup_63 = t_base as f64 / t_63 as f64;

    println!(
        "[APX 6.3 BENCH] size {} -> baseline={}us | tiled6.1={}us | tiled6.3={}us | speedup6.1={:.2} | speedup6.3={:.2}",
        m,
        t_base,
        t_61,
        t_63,
        speedup_61,
        speedup_63,
    );

    // En algunos entornos el baseline AVX2 ya está muy optimizado; el objetivo
    // de este benchmark es observacional, no forzar un speedup mínimo.
}

#[test]
fn apx_6_3_matmul_tiled_benchmark() {
    let sizes = [128usize, 256, 512, 1024];
    for &s in &sizes {
        bench_size(s, s, s);
    }
}
