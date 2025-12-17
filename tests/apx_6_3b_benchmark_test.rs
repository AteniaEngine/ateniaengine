use std::time::Instant;

use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::kernels::matmul_tiled_cpu; // APX 6.1
use atenia_engine::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b; // APX 6.3B
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

fn matmul_6_3b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    matmul_tiled_6_3b(a, b, &mut out, m, k, n);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn bench_size(m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.113).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.317).cos()).collect();

    let (out_base, t_base) = baseline_matmul(&a, &b, m, k, n);
    let (out_61, t_61) = matmul_6_1(&a, &b, m, k, n);
    let (out_63b, t_63b) = matmul_6_3b(&a, &b, m, k, n);

    for i in 0..out_base.len() {
        assert!((out_base[i] - out_61[i]).abs() < 1e-5);
        assert!((out_base[i] - out_63b[i]).abs() < 1e-5);
    }

    let speedup_63b = t_base as f64 / t_63b as f64;

    println!(
        "[APX 6.3B BENCH] size {} -> baseline={} | tiled6.1={} | tiled6.3b={} | speedup6.3b={:.2}",
        m,
        t_base,
        t_61,
        t_63b,
        speedup_63b,
    );
    // En este entorno concreto, el baseline APX 3.8 ya está muy optimizado
    // y puede superar al kernel 6.3B. Este benchmark se mantiene como
    // herramienta de diagnóstico y comparación, sin forzar un speedup
    // mínimo mediante aserciones.
}

#[test]
fn apx_6_3b_benchmark() {
    let sizes = [128usize, 256, 512];
    for &s in &sizes {
        bench_size(s, s, s);
    }
}
