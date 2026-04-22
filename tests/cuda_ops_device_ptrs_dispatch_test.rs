//! APX v20 M3-d.4.E — dispatch tests for the 3 CUDA ops.
//!
//! These tests cover the post-M3-d.4.E invariant: each of
//! `cuda_linear`, `cuda_batch_matmul`, `cuda_fused_linear_silu`
//! inspects the storage of its `&Tensor` arguments and routes to one
//! of two code paths:
//!
//! - All four operands are `TensorStorage::Cuda` → device-pointer
//!   variant (`launch_*_f32_device_ptrs`), which skips the H<->D
//!   roundtrip entirely.
//! - Anything else (all-Cpu or mixed) → host path (`*_raw`), which
//!   panics via `Tensor::as_cpu_slice` if any operand is `Cuda`.
//!
//! Coverage per op:
//! 1. `_all_cuda_dispatch` — migrate every operand with
//!    `ensure_gpu`, invoke the op, verify numeric correctness
//!    against a CPU reference.
//! 2. `_all_cpu_dispatch` — every operand stays Cpu, verify the
//!    host path runs and returns the same result.
//! 3. `_mixed_storage_panics` — one operand on GPU, the rest on
//!    CPU, verify the call panics with the M3-d.2 message.
//!
//! Every test graceful-skips when no GPU is available, following the
//! pattern of prior M3-d tests.

use atenia_engine::cuda::batch_matmul::cuda_batch_matmul;
use atenia_engine::cuda::fused_linear_silu::cuda_fused_linear_silu;
use atenia_engine::cuda::linear::cuda_linear;
use atenia_engine::gpu::gpu_engine;
use atenia_engine::nn::linear::linear as cpu_linear;
use atenia_engine::tensor::ops::batch_matmul::batch_matmul_parallel;
use atenia_engine::apx4_8::fused_linear_activation::exec_fused_linear_silu;
use atenia_engine::tensor::{Device, Tensor};

fn require_gpu(test_name: &str) -> bool {
    if gpu_engine().is_some() {
        true
    } else {
        println!(
            "[TEST:{}] no GPU available (gpu_engine() = None) -> graceful skip",
            test_name
        );
        false
    }
}

fn assert_panic_message_has_gpu_resident_and_ensure_cpu(
    err: Box<dyn std::any::Any + Send>,
    op_name: &str,
) {
    let msg = err
        .downcast_ref::<String>()
        .map(|s| s.as_str())
        .or_else(|| err.downcast_ref::<&'static str>().copied())
        .unwrap_or("");
    assert!(
        msg.contains("GPU-resident"),
        "{}: panic message must mention 'GPU-resident'; got: {}",
        op_name,
        msg
    );
    assert!(
        msg.contains("ensure_cpu"),
        "{}: panic message must mention 'ensure_cpu'; got: {}",
        op_name,
        msg
    );
}

// =========================================================================
// cuda_linear
// =========================================================================

#[test]
fn test_cuda_linear_all_cuda_dispatch() {
    if !require_gpu("test_cuda_linear_all_cuda_dispatch") {
        return;
    }

    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);

    // CPU reference (before any GPU migration).
    let cpu_ref = cpu_linear(&x, &w, Some(&b));

    // Migrate every operand + output to VRAM so cuda_linear routes
    // through the device-pointer dispatch.
    let mut x_gpu = x;
    let mut w_gpu = w;
    let mut b_gpu = b;
    let mut out_gpu = Tensor::zeros_new(&[m, n], Device::CPU);

    x_gpu.ensure_gpu().expect("x ensure_gpu");
    w_gpu.ensure_gpu().expect("w ensure_gpu");
    b_gpu.ensure_gpu().expect("b ensure_gpu");
    out_gpu.ensure_gpu().expect("out ensure_gpu");

    cuda_linear(&x_gpu, &w_gpu, &b_gpu, &mut out_gpu, m, k, n);

    // Bring result back to host to check correctness.
    out_gpu.ensure_cpu().expect("out ensure_cpu");

    for i in 0..(m * n) {
        let d = (cpu_ref.as_cpu_slice()[i] - out_gpu.as_cpu_slice()[i]).abs();
        assert!(d < 1e-3, "mismatch at {}: diff = {}", i, d);
    }
}

#[test]
fn test_cuda_linear_all_cpu_dispatch() {
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let cpu_ref = cpu_linear(&x, &w, Some(&b));

    // No migration: everything stays on host; cuda_linear should
    // route to the _raw path and still produce the correct answer.
    cuda_linear(&x, &w, &b, &mut out, m, k, n);

    for i in 0..(m * n) {
        let d = (cpu_ref.as_cpu_slice()[i] - out.as_cpu_slice()[i]).abs();
        assert!(d < 1e-3, "mismatch at {}: diff = {}", i, d);
    }
}

#[test]
fn test_cuda_linear_mixed_storage_panics() {
    if !require_gpu("test_cuda_linear_mixed_storage_panics") {
        return;
    }

    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    // One operand on GPU, the rest on CPU: not "all Cuda", so the
    // host path runs and panics when `as_cpu_slice` sees the Cuda
    // operand.
    let mut x = Tensor::randn(&[m, k], Device::CPU);
    x.ensure_gpu().expect("x ensure_gpu");
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cuda_linear(&x, &w, &b, &mut out, m, k, n);
    }));
    let err = result.expect_err("cuda_linear must panic on mixed storage");
    assert_panic_message_has_gpu_resident_and_ensure_cpu(err, "cuda_linear");
}

// =========================================================================
// cuda_batch_matmul
// =========================================================================

#[test]
fn test_cuda_batch_matmul_all_cuda_dispatch() {
    if !require_gpu("test_cuda_batch_matmul_all_cuda_dispatch") {
        return;
    }

    let batch = 2usize;
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let a = Tensor::randn(&[batch, m, k], Device::CPU);
    let b = Tensor::randn(&[batch, k, n], Device::CPU);
    let cpu_ref = batch_matmul_parallel(&a, &b);

    let mut a_gpu = a;
    let mut b_gpu = b;
    let mut out_gpu = Tensor::zeros_new(&[batch, m, n], Device::CPU);

    a_gpu.ensure_gpu().expect("a ensure_gpu");
    b_gpu.ensure_gpu().expect("b ensure_gpu");
    out_gpu.ensure_gpu().expect("out ensure_gpu");

    cuda_batch_matmul(&a_gpu, &b_gpu, &mut out_gpu, batch, m, k, n);

    out_gpu.ensure_cpu().expect("out ensure_cpu");

    for i in 0..cpu_ref.numel() {
        let d = (cpu_ref.as_cpu_slice()[i] - out_gpu.as_cpu_slice()[i]).abs();
        assert!(d < 1e-3, "mismatch at {}: diff = {}", i, d);
    }
}

#[test]
fn test_cuda_batch_matmul_all_cpu_dispatch() {
    let batch = 2usize;
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let a = Tensor::randn(&[batch, m, k], Device::CPU);
    let b = Tensor::randn(&[batch, k, n], Device::CPU);
    let cpu_ref = batch_matmul_parallel(&a, &b);
    let mut out = Tensor::zeros_new(&[batch, m, n], Device::CPU);

    cuda_batch_matmul(&a, &b, &mut out, batch, m, k, n);

    for i in 0..cpu_ref.numel() {
        let d = (cpu_ref.as_cpu_slice()[i] - out.as_cpu_slice()[i]).abs();
        assert!(d < 1e-3, "mismatch at {}: diff = {}", i, d);
    }
}

#[test]
fn test_cuda_batch_matmul_mixed_storage_panics() {
    if !require_gpu("test_cuda_batch_matmul_mixed_storage_panics") {
        return;
    }

    let batch = 2usize;
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;

    let mut a = Tensor::randn(&[batch, m, k], Device::CPU);
    a.ensure_gpu().expect("a ensure_gpu");
    let b = Tensor::randn(&[batch, k, n], Device::CPU);
    let mut out = Tensor::zeros_new(&[batch, m, n], Device::CPU);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cuda_batch_matmul(&a, &b, &mut out, batch, m, k, n);
    }));
    let err = result.expect_err("cuda_batch_matmul must panic on mixed storage");
    assert_panic_message_has_gpu_resident_and_ensure_cpu(err, "cuda_batch_matmul");
}

// =========================================================================
// cuda_fused_linear_silu
// =========================================================================

#[test]
fn test_cuda_fused_linear_silu_all_cuda_dispatch() {
    if !require_gpu("test_cuda_fused_linear_silu_all_cuda_dispatch") {
        return;
    }

    let m = 4usize;
    let k = 8usize;
    let n = 16usize;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let cpu_ref = exec_fused_linear_silu(&x, &w, Some(&b));

    let mut x_gpu = x;
    let mut w_gpu = w;
    let mut b_gpu = b;
    let mut out_gpu = Tensor::zeros_new(&[m, n], Device::CPU);

    x_gpu.ensure_gpu().expect("x ensure_gpu");
    w_gpu.ensure_gpu().expect("w ensure_gpu");
    b_gpu.ensure_gpu().expect("b ensure_gpu");
    out_gpu.ensure_gpu().expect("out ensure_gpu");

    cuda_fused_linear_silu(&x_gpu, &w_gpu, &b_gpu, &mut out_gpu, m, k, n);

    out_gpu.ensure_cpu().expect("out ensure_cpu");

    let mut max_diff = 0.0f32;
    for i in 0..(m * n) {
        let d = (cpu_ref.as_cpu_slice()[i] - out_gpu.as_cpu_slice()[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(max_diff < 1e-3, "max_diff = {}", max_diff);
}

#[test]
fn test_cuda_fused_linear_silu_all_cpu_dispatch() {
    let m = 4usize;
    let k = 8usize;
    let n = 16usize;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let cpu_ref = exec_fused_linear_silu(&x, &w, Some(&b));
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    cuda_fused_linear_silu(&x, &w, &b, &mut out, m, k, n);

    let mut max_diff = 0.0f32;
    for i in 0..(m * n) {
        let d = (cpu_ref.as_cpu_slice()[i] - out.as_cpu_slice()[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(max_diff < 1e-3, "max_diff = {}", max_diff);
}

#[test]
fn test_cuda_fused_linear_silu_mixed_storage_panics() {
    if !require_gpu("test_cuda_fused_linear_silu_mixed_storage_panics") {
        return;
    }

    let m = 4usize;
    let k = 8usize;
    let n = 16usize;

    let mut x = Tensor::randn(&[m, k], Device::CPU);
    x.ensure_gpu().expect("x ensure_gpu");
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cuda_fused_linear_silu(&x, &w, &b, &mut out, m, k, n);
    }));
    let err = result.expect_err("cuda_fused_linear_silu must panic on mixed storage");
    assert_panic_message_has_gpu_resident_and_ensure_cpu(err, "cuda_fused_linear_silu");
}
