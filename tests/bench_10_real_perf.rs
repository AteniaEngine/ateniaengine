use std::time::Instant;

use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::gpu::tensor::{GpuTensorManager, TensorGPU};
use atenia_engine::gpu::ops::matmul::MatMulOp;
use atenia_engine::gpu::ops::linear::LinearOp;
use atenia_engine::gpu::autodiff::{matmul_backward::MatMulBackwardGPU, linear_backward::LinearBackwardGPU};

/// Helper: measure N iterations and return avg in ms
fn benchmark<F: Fn()>(name: &str, iters: usize, f: F) {
    // Warmup
    for _ in 0..5 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dur = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("[BENCH] {:30} {:10.4} ms avg", name, dur);
}

fn matmul_cpu(a: &Tensor, b: &Tensor) -> Tensor {
    let n = a.shape[0];
    let k = a.shape[1];
    let m = b.shape[1];
    let mut out = Tensor::zeros(vec![n, m], Device::CPU, DType::F32);
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a.data[i * k + kk] * b.data[kk * m + j];
            }
            out.data[i * m + j] = acc;
        }
    }
    out
}

fn linear_cpu(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    // x: [M,K], w: [N,K], b: [N]
    let m = x.shape[0];
    let k = x.shape[1];
    let n = w.shape[0];
    let mut out = Tensor::zeros(vec![m, n], Device::CPU, DType::F32);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x.data[i * k + kk] * w.data[j * k + kk];
            }
            acc += b.data[j];
            out.data[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn bench_real_perf() {
    // Try to create GPU manager
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => {
            println!("[BENCH] No GPU available → skipping benchmarks.");
            return;
        }
    };

    println!("=== APX 10.B — REAL CPU vs GPU BENCHMARK ===");

    // Sizes
    let sizes = [512usize, 1024usize];

    for &n in &sizes {
        println!("\n--- Matrix size: {} x {} ---", n, n);

        // CPU tensors
        let a_cpu = Tensor::random(vec![n, n], Device::CPU, DType::F32);
        let b_cpu = Tensor::random(vec![n, n], Device::CPU, DType::F32);

        // GPU tensors (from raw data)
        let a_gpu = match mgr.from_cpu_vec(&a_cpu.data, n, n) {
            Ok(t) => t,
            Err(_) => {
                println!("[BENCH] GPU alloc failed at size {} (a_gpu)", n);
                continue;
            }
        };
        let b_gpu = match mgr.from_cpu_vec(&b_cpu.data, n, n) {
            Ok(t) => t,
            Err(_) => {
                println!("[BENCH] GPU alloc failed at size {} (b_gpu)", n);
                continue;
            }
        };

        // MATMUL CPU
        let name = format!("CPU MatMul {}x{}", n, n);
        println!("[BENCH] Running {} ...", name);
        benchmark(&name, 5, || {
            let _ = matmul_cpu(&a_cpu, &b_cpu);
        });

        // MATMUL GPU
        let name = format!("GPU MatMul {}x{}", n, n);
        println!("[BENCH] Running {} ...", name);
        benchmark(&name, 5, || {
            let c_gpu = match TensorGPU::empty(&mgr.mem, n, n) {
                Ok(t) => t,
                Err(_) => return,
            };
            MatMulOp::run(&a_gpu.ptr, &b_gpu.ptr, &c_gpu.ptr, n, n, n);
        });

        // LINEAR CPU
        let w_cpu = Tensor::random(vec![n, n], Device::CPU, DType::F32);
        let b_cpu_lin = Tensor::random(vec![n], Device::CPU, DType::F32);

        let name = format!("CPU Linear {}x{}", n, n);
        println!("[BENCH] Running {} ...", name);
        benchmark(&name, 5, || {
            let _ = linear_cpu(&a_cpu, &w_cpu, &b_cpu_lin);
        });

        // LINEAR GPU
        let w_gpu = match mgr.from_cpu_vec(&w_cpu.data, n, n) {
            Ok(t) => t,
            Err(_) => {
                println!("[BENCH] GPU alloc failed at size {} (w_gpu)", n);
                continue;
            }
        };
        let b_gpu = match mgr.from_cpu_vec(&b_cpu_lin.data, 1, n) {
            Ok(t) => t,
            Err(_) => {
                println!("[BENCH] GPU alloc failed at size {} (b_gpu)", n);
                continue;
            }
        };

        let name = format!("GPU Linear {}x{}", n, n);
        println!("[BENCH] Running {} ...", name);
        benchmark(&name, 5, || {
            let out_gpu = match TensorGPU::empty(&mgr.mem, n, n) {
                Ok(t) => t,
                Err(_) => return,
            };
            LinearOp::run(&a_gpu.ptr, &w_gpu.ptr, &b_gpu.ptr, &out_gpu.ptr, n, n, n);
        });

        // BACKWARD MATMUL GPU (real)
        let name = format!("GPU Backward MatMul {}x{}", n, n);
        println!("[BENCH] Running {} ...", name);
        benchmark(&name, 3, || {
            let dout_gpu = match TensorGPU::empty(&mgr.mem, n, n) {
                Ok(t) => t,
                Err(_) => return,
            };
            let _ = MatMulBackwardGPU::run(&mgr, &a_gpu, &b_gpu, &dout_gpu);
        });

        // BACKWARD LINEAR GPU (real)
        let name = format!("GPU Backward Linear {}x{}", n, n);
        println!("[BENCH] Running {} ...", name);
        benchmark(&name, 3, || {
            let dout_gpu = match TensorGPU::empty(&mgr.mem, n, n) {
                Ok(t) => t,
                Err(_) => return,
            };
            let _ = LinearBackwardGPU::run(&mgr, &a_gpu, &w_gpu, &dout_gpu);
        });
    }

    println!("\n=== END BENCH ===");
}
