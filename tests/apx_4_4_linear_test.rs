use atenia_engine::nn::linear::linear;
use atenia_engine::tensor::{Device, Tensor};

#[test]
fn gpu_linear_matches_cpu() {
    let m = 4;
    let k = 8;
    let n = 6;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    // Bias must be 1D [n]
    let b = Tensor::randn(&[n], Device::CPU);

    let cpu = linear(&x, &w, Some(&b));

    let mut gpu_out = Tensor::zeros_new(&[m, n], Device::CPU);

    atenia_engine::cuda::linear::cuda_linear(&x, &w, &b, &mut gpu_out, m, k, n);

    for i in 0..(m * n) {
        let d = (cpu.as_cpu_slice()[i] - gpu_out.as_cpu_slice()[i]).abs();
        assert!(d < 1e-3, "Mismatch at {} → {}", i, d);
    }
}

#[test]
fn cuda_linear_panics_on_cuda_input() {
    // After M3-d.4.C the op receives `&Tensor`. Passing a GPU-resident
    // input triggers the same panic that `Tensor::as_cpu_slice` has
    // emitted since M3-d.2: the caller is expected to call
    // `ensure_cpu()` before invoking the op. The panic message must
    // mention "GPU-resident" and "ensure_cpu" so callers can act.
    if atenia_engine::gpu::gpu_engine().is_none() {
        println!("[TEST:cuda_linear_panics_on_cuda_input] no GPU available -> graceful skip");
        return;
    }

    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let mut x = Tensor::randn(&[m, k], Device::CPU);
    x.ensure_gpu().expect("test setup: ensure_gpu must succeed");
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        atenia_engine::cuda::linear::cuda_linear(&x, &w, &b, &mut out, m, k, n);
    }));

    let err = result.expect_err("cuda_linear must panic on GPU-resident input");
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
