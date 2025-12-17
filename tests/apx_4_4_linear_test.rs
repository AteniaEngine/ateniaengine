use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::nn::linear::linear;

#[test]
fn gpu_linear_matches_cpu() {
    let m = 4;
    let k = 8;
    let n = 6;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    // Bias debe ser 1D [n]
    let b = Tensor::randn(&[n], Device::CPU);

    let cpu = linear(&x, &w, Some(&b));

    let mut gpu_out = Tensor::zeros_new(&[m, n], Device::CPU);

    atenia_engine::cuda::linear::cuda_linear(
        &x.data,
        &w.data,
        &b.data,
        &mut gpu_out.data,
        m,
        k,
        n,
    );

    for i in 0..(m * n) {
        let d = (cpu.data[i] - gpu_out.data[i]).abs();
        assert!(d < 1e-3, "Mismatch at {} → {}", i, d);
    }
}
