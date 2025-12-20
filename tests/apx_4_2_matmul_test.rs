use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::cuda::matmul::cuda_matmul;

#[test]
fn gpu_matmul_matches_cpu_small() {
    // Deterministic 2x2 case to debug indexing and data.
    let m = 2; let k = 2; let n = 2;

    let mut a = Tensor::zeros_new(&[m, k], Device::CPU);
    a.data.copy_from_slice(&[1.0, 2.0,
                             3.0, 4.0]);

    let mut b = Tensor::zeros_new(&[k, n], Device::CPU);
    b.data.copy_from_slice(&[5.0, 6.0,
                             7.0, 8.0]);

    // CPU reference using Tensor::matmul (which goes through the scalar matmul_dispatch).
    let cpu = a.matmul(&b);

    // CUDA path with the same host buffers.
    let gpu = cuda_matmul(&a, &b, m, k, n);

    for i in 0..cpu.data.len() {
        let d = (cpu.data[i] - gpu.data[i]).abs();
        if d >= 1e-4 {
            eprintln!(
                "idx {}: cpu={}, gpu={}, diff={}",
                i, cpu.data[i], gpu.data[i], d
            );
        }
        assert!(d < 1e-4, "diff too large at {}: {}", i, d);
    }
}
