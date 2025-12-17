use std::time::Instant;

use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b;
use atenia_engine::apx6_4::matmul_4x8_avx2;
use atenia_engine::tensor::Device;

fn baseline_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let ctx = DeviceContext::new(Device::CPU);
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
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

fn matmul_6_4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn bench_size(m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.151).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.311).cos()).collect();

    let (out_base, t_base) = baseline_matmul(&a, &b, m, k, n);
    let (out_63b, t_63b) = matmul_6_3b(&a, &b, m, k, n);
    let (out_64, t_64) = matmul_6_4(&a, &b, m, k, n);

    for i in 0..out_base.len() {
        assert!((out_base[i] - out_63b[i]).abs() < 1e-4);
        assert!((out_base[i] - out_64[i]).abs() < 1e-4);
    }

    let speedup_64 = t_base as f64 / t_64 as f64;

    println!(
        "[APX 6.4 BENCH] size={} -> baseline={}us | tiled6.3B={}us | microkernel6.4={}us | speedup={:.2}%",
        m,
        t_base,
        t_63b,
        t_64,
        speedup_64 * 100.0,
    );
}

#[test]
fn apx_6_4_benchmark() {
    let sizes = [128usize, 256, 512, 1024];
    for &s in &sizes {
        bench_size(s, s, s);
    }
}
