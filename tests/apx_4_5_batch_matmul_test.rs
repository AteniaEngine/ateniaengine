use atenia_engine::cuda::batch_matmul::cuda_batch_matmul;
use atenia_engine::tensor::ops::batch_matmul::batch_matmul_parallel;
use atenia_engine::tensor::{Device, Tensor};

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

    cuda_batch_matmul(&a, &b, &mut out, batch, m, k, n);

    assert_eq!(cpu.shape, out.shape);

    for i in 0..cpu.numel() {
        let d = (cpu.as_cpu_slice()[i] - out.as_cpu_slice()[i]).abs();
        assert!(d < 1e-3, "diff too large at {}: {}", i, d);
    }
}

#[test]
fn cuda_batch_matmul_panics_on_cuda_input() {
    // See apx_4_4_linear_test::cuda_linear_panics_on_cuda_input for rationale.
    if atenia_engine::gpu::gpu_engine().is_none() {
        println!("[TEST:cuda_batch_matmul_panics_on_cuda_input] no GPU available -> graceful skip");
        return;
    }

    let batch = 2usize;
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let mut a = Tensor::randn(&[batch, m, k], Device::CPU);
    a.ensure_gpu().expect("test setup: ensure_gpu must succeed");
    let b = Tensor::randn(&[batch, k, n], Device::CPU);
    let mut out = Tensor::zeros_new(&[batch, m, n], Device::CPU);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cuda_batch_matmul(&a, &b, &mut out, batch, m, k, n);
    }));

    let err = result.expect_err("cuda_batch_matmul must panic on GPU-resident input");
    let msg = err
        .downcast_ref::<String>()
        .map(|s| s.as_str())
        .or_else(|| err.downcast_ref::<&'static str>().copied())
        .unwrap_or("");
    assert!(
        msg.contains("GPU-resident"),
        "panic message must mention 'GPU-resident'; got: {}",
        msg
    );
    assert!(
        msg.contains("ensure_cpu"),
        "panic message must mention 'ensure_cpu'; got: {}",
        msg
    );
}
