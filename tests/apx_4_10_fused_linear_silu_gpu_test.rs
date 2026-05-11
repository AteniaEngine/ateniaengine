use atenia_engine::apx4_8::fused_linear_activation::exec_fused_linear_silu;
use atenia_engine::cuda::fused_linear_silu::cuda_fused_linear_silu;
use atenia_engine::tensor::{Device, Tensor};

#[test]
fn test_fused_linear_silu_gpu_matches_cpu() {
    let m = 4;
    let k = 8;
    let n = 16;

    // For this APX we use Device::CPU as the logical backend (same as cuda_matmul).
    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);

    let mut out_gpu = Tensor::zeros_new(&[m, n], Device::CPU);

    cuda_fused_linear_silu(&x, &w, &b, &mut out_gpu, m, k, n);

    // CPU reference using the existing fused implementation.
    let out_cpu = exec_fused_linear_silu(&x, &w, Some(&b));

    let mut max_diff = 0.0f32;
    for i in 0..(m * n) {
        let d = (out_cpu.as_cpu_slice()[i] - out_gpu.as_cpu_slice()[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("[APX 4.10] max_diff = {}", max_diff);
    assert!(max_diff < 1e-3);
}

#[test]
fn cuda_fused_linear_silu_panics_on_cuda_input() {
    // See apx_4_4_linear_test::cuda_linear_panics_on_cuda_input for rationale.
    if atenia_engine::gpu::gpu_engine().is_none() {
        println!(
            "[TEST:cuda_fused_linear_silu_panics_on_cuda_input] no GPU available -> graceful skip"
        );
        return;
    }

    let m = 4usize;
    let k = 8usize;
    let n = 16usize;

    let mut x = Tensor::randn(&[m, k], Device::CPU);
    x.ensure_gpu().expect("test setup: ensure_gpu must succeed");
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cuda_fused_linear_silu(&x, &w, &b, &mut out, m, k, n);
    }));

    let err = result.expect_err("cuda_fused_linear_silu must panic on GPU-resident input");
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
