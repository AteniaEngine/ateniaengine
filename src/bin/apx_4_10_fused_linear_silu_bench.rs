use std::time::Instant;

use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::cuda::fused_linear_silu::cuda_fused_linear_silu;

fn main() {
    let m = 64usize;
    let k = 128usize;
    let n = 64usize;

    // Datos en CPU; el kernel CUDA opera sobre slices host igual que cuda_linear.
    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b0 = Tensor::randn(&[n], Device::CPU);
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    // Calentamiento
    for _ in 0..10 {
        cuda_fused_linear_silu(&x.data, &w.data, &b0.data, &mut out.data, m, k, n);
    }

    let iters = 1000u32;
    let start = Instant::now();
    for _ in 0..iters {
        cuda_fused_linear_silu(&x.data, &w.data, &b0.data, &mut out.data, m, k, n);
    }
    let elapsed = start.elapsed();

    let avg = elapsed.as_secs_f64() / iters as f64;
    println!("[APX 4.10] fused_linear_silu: iters={iters}, total={:?}, avg={:.3} us", elapsed, avg * 1e6);
}
