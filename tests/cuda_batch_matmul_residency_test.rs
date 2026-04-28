//! M4.7.3.b — `cuda_batch_matmul` residency path unit test.
//!
//! Mirrors `cuda_matmul_residency_test`: allocates two batched
//! tensors directly on VRAM via `Tensor::zeros_new_cuda`, fills them
//! through host upload, runs `cuda_batch_matmul` with all three
//! tensors Cuda-resident (the `all_cuda` device-pointer dispatch),
//! and asserts:
//!   - the GPU-resident output stays `Cuda` after the op (residency
//!     preserved — no CPU-roundtrip), and
//!   - the GPU result matches a CPU reference within 1e-4 absolute
//!     (per the M4.7.3 envelope).
//!
//! The companion executor-arm change (`graph.rs:BatchMatMul`) is a
//! pure routing patch: when both operands and the output land in
//! `TensorStorage::Cuda`, the executor delegates straight to
//! `cuda_batch_matmul`'s `all_cuda` branch — so verifying that
//! branch end-to-end is sufficient coverage for the .b sub-step.
//!
//! Marked `#[ignore]` because it requires a working CUDA driver. Run
//! locally with:
//!
//! ```powershell
//! cargo test --test cuda_batch_matmul_residency_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::cuda::batch_matmul::cuda_batch_matmul;
use atenia_engine::cuda::cuda_available;
use atenia_engine::tensor::tensor::{Tensor, TensorStorage};

fn cpu_batch_matmul_ref(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * m * n];
    for o in 0..batch {
        let a_off = o * m * k;
        let b_off = o * k * n;
        let out_off = o * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += a[a_off + i * k + kk] * b[b_off + kk * n + j];
                }
                out[out_off + i * n + j] = acc;
            }
        }
    }
    out
}

fn make_cuda_tensor_from_host(data: &[f32], shape: &[usize]) -> Tensor {
    let mut t =
        Tensor::zeros_new_cuda(shape).expect("VRAM allocation failed (no CUDA driver?)");
    t.ensure_cpu()
        .expect("ensure_cpu failed on freshly-allocated Cuda");
    t.set_cpu_data(data.to_vec());
    t.ensure_gpu().expect("ensure_gpu failed");
    assert!(matches!(t.storage(), TensorStorage::Cuda(_)));
    t
}

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn cuda_batch_matmul_residency_path_matches_cpu() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    let batch = 2_usize;
    let m = 3;
    let k = 5;
    let n = 4;

    let a_host: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let b_host: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.05 + 0.25).collect();

    let cpu_ref = cpu_batch_matmul_ref(&a_host, &b_host, batch, m, k, n);

    let a_gpu = make_cuda_tensor_from_host(&a_host, &[batch, m, k]);
    let b_gpu = make_cuda_tensor_from_host(&b_host, &[batch, k, n]);
    let mut out_gpu = Tensor::zeros_new_cuda(&[batch, m, n])
        .expect("VRAM allocation for output failed");

    cuda_batch_matmul(&a_gpu, &b_gpu, &mut out_gpu, batch, m, k, n);

    // Output must still be Cuda after the op (residency preserved).
    assert!(
        matches!(out_gpu.storage(), TensorStorage::Cuda(_)),
        "expected Cuda storage on output post-op (residency path); got {:?}",
        out_gpu.storage()
    );

    out_gpu.ensure_cpu().expect("device→host transfer failed");
    let gpu_values = out_gpu.as_cpu_slice();
    assert_eq!(gpu_values.len(), cpu_ref.len());

    let mut max_abs_diff = 0.0_f32;
    for (g, c) in gpu_values.iter().zip(cpu_ref.iter()) {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }

    println!(
        "GPU residency BatchMatMul vs CPU reference: max_abs_diff = {:.2e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 1e-4,
        "GPU residency BatchMatMul drift {:.2e} exceeds 1e-4 envelope",
        max_abs_diff
    );
}
