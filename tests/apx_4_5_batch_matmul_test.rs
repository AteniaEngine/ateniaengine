use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::tensor::ops::batch_matmul::batch_matmul_parallel;
use atenia_engine::cuda::batch_matmul::cuda_batch_matmul;

#[test]
fn apx_4_5_batch_matmul_matches_cpu() {
    let batch = 3usize;
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let a = Tensor::randn(&[batch, m, k], Device::CPU);
    let b = Tensor::randn(&[batch, k, n], Device::CPU);

    let cpu = batch_matmul_parallel(&a, &b);

    let mut out = Tensor::zeros_new(&[batch, m, n], Device::CPU);

    cuda_batch_matmul(&a.data, &b.data, &mut out.data, batch, m, k, n);

    assert_eq!(cpu.shape, out.shape);

    for i in 0..cpu.data.len() {
        let d = (cpu.data[i] - out.data[i]).abs();
        assert!(d < 1e-3, "diff too large at {}: {}", i, d);
    }
}
