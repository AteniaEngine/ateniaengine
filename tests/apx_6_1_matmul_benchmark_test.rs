use std::time::Instant;

use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::kernels::matmul_tiled_cpu::matmul_tiled_cpu;
use atenia_engine::tensor::Device;

fn run_baseline(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let ctx = DeviceContext::new(Device::CPU);
    let mut out = vec![0.0f32; m * n];
    let start = Instant::now();
    dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
    let elapsed = start.elapsed().as_micros();
    (out, elapsed)
}

fn run_tiled(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let start = Instant::now();
    matmul_tiled_cpu(a, b, &mut out, m, k, n);
    let elapsed = start.elapsed().as_micros();
    (out, elapsed)
}

fn benchmark_size(m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32).sin() * 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32).cos() * 0.5).collect();

    let (out_base, t_base) = run_baseline(&a, &b, m, k, n);
    let (out_tiled, t_tiled) = run_tiled(&a, &b, m, k, n);

    for (i, (x, y)) in out_base.iter().zip(out_tiled.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff < 1e-3, "mismatch at {}: base={} tiled={} diff={}", i, x, y, diff);
    }

    let speedup = if t_tiled > 0 { t_base as f64 / t_tiled as f64 } else { 0.0 };
    println!(
        "[APX 6.1 BENCH] size {}x{}x{} -> baseline={} us | tiled={} us | speedup={:.2}x",
        m, k, n, t_base, t_tiled, speedup
    );
}

#[test]
fn apx_6_1_matmul_benchmark() {
    // Este benchmark no modifica el runtime ni las rutas de ejecución; solo
    // mide los kernels existentes para distintos tamaños.
    benchmark_size(64, 64, 64);
    benchmark_size(128, 128, 128);
    benchmark_size(256, 256, 256);
    benchmark_size(512, 512, 512);
}
