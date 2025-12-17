use atenia_engine::matmul_dispatcher::matmul_dispatch;
use atenia_engine::kernels::matmul_tiled_cpu::matmul_tiled_cpu;
use atenia_engine::apx4::gpu_dispatch::{dispatch_matmul, ApxExecTarget};
use std::time::Instant;

fn random_matrix(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.12345).sin()).collect()
}

fn bench_case(m: usize, k: usize, n: usize) {
    let a = random_matrix(m * k);
    let b = random_matrix(k * n);

    let mut out_cpu = vec![0.0f32; m * n];
    let mut out_tiled = vec![0.0f32; m * n];
    let mut out_gpu = vec![0.0f32; m * n];

    // --- CPU BASELINE (APX 3.8 dispatcher)
    let t0 = Instant::now();
    matmul_dispatch(&a, &b, &mut out_cpu, m, k, n);
    let t_cpu = t0.elapsed().as_micros();

    // --- CPU TILED (APX 6.1)
    let t1 = Instant::now();
    matmul_tiled_cpu(&a, &b, &mut out_tiled, m, k, n);
    let t_tiled = t1.elapsed().as_micros();

    // --- GPU (CUDA) usando el dispatcher existente de APX 4
    let t2 = Instant::now();
    dispatch_matmul(&a, &b, m, k, n, &mut out_gpu, ApxExecTarget::GPU);
    let t_gpu = t2.elapsed().as_micros();

    // --- NUMERIC CHECK
    for ((c, t), g) in out_cpu.iter().zip(out_tiled.iter()).zip(out_gpu.iter()) {
        assert!((c - t).abs() < 1e-2);
        assert!((c - g).abs() < 1e-2);
    }

    println!(
        "[APX 6.1 GPU BENCH] size {}x{}x{} -> cpu={} us | tiled={} us | gpu={} us | cpu/gpu={:.2}x | tiled/gpu={:.2}x",
        m,
        k,
        n,
        t_cpu,
        t_tiled,
        t_gpu,
        t_cpu as f64 / t_gpu as f64,
        t_tiled as f64 / t_gpu as f64,
    );
}

#[test]
fn benchmark_cpu_vs_gpu() {
    println!("\n========== APX 6.1 CPU vs GPU Benchmark ==========\n");

    let sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    for &(m, k, n) in &sizes {
        bench_case(m, k, n);
    }
}
