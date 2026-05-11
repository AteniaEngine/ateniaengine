//! M4.7.3.a — `cuda_matmul_inplace` residency path unit test.
//!
//! Allocates two tensors directly on VRAM via `Tensor::zeros_new_cuda`,
//! fills them through host upload (`Tensor::ensure_cpu` then back to
//! Cuda, exercising both directions of the M3 transfer machinery),
//! runs `cuda_matmul_inplace`, and asserts the GPU-resident output
//! matches a reference CPU matmul to within 1e-4 absolute (per the
//! M4.7.3 investigation envelope).
//!
//! Marked `#[ignore]` because it requires a working CUDA driver. Run
//! locally with:
//!
//! ```powershell
//! cargo test --test cuda_matmul_residency_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::cuda::cuda_available;
use atenia_engine::cuda::matmul::cuda_matmul_inplace;
use atenia_engine::tensor::tensor::{Tensor, TensorStorage};

fn cpu_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    a.matmul(b)
}

fn make_cuda_tensor_from_host(data: &[f32], shape: &[usize]) -> Tensor {
    // Allocate on VRAM, then upload.
    let mut t = Tensor::zeros_new_cuda(shape).expect("VRAM allocation failed (no CUDA driver?)");
    // Promote to Cpu, write data, promote back to Cuda — exercises
    // both directions of the storage transfer machinery.
    t.ensure_cpu()
        .expect("ensure_cpu failed on freshly-allocated Cuda");
    t.set_cpu_data(data.to_vec());
    t.ensure_gpu().expect("ensure_gpu failed");
    assert!(matches!(t.storage(), TensorStorage::Cuda(_)));
    t
}

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn cuda_matmul_inplace_residency_path_matches_cpu() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    let m = 4_usize;
    let k = 8;
    let n = 6;

    // Hand-crafted operand patterns: small, deterministic values so
    // any GPU-vs-CPU drift is well below the 1e-4 envelope claimed
    // by the M4.7.3 investigation.
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.5).collect();

    // ---- Reference CPU result ----
    let a_cpu = Tensor::new_cpu(vec![m, k], a_host.clone());
    let b_cpu = Tensor::new_cpu(vec![k, n], b_host.clone());
    let cpu_out = cpu_matmul(&a_cpu, &b_cpu);
    assert_eq!(cpu_out.shape, vec![m, n]);
    let cpu_values = cpu_out.as_cpu_slice().to_vec();

    // ---- GPU residency path ----
    let a_gpu = make_cuda_tensor_from_host(&a_host, &[m, k]);
    let b_gpu = make_cuda_tensor_from_host(&b_host, &[k, n]);
    let mut out_gpu = Tensor::zeros_new_cuda(&[m, n]).expect("VRAM allocation for output failed");

    cuda_matmul_inplace(&a_gpu, &b_gpu, &mut out_gpu, m, k, n);

    // Output must still be Cuda after the op (residency preserved).
    assert!(
        matches!(out_gpu.storage(), TensorStorage::Cuda(_)),
        "expected Cuda storage on output post-op (residency path); got {:?}",
        out_gpu.storage()
    );

    // Materialise to host for comparison.
    out_gpu.ensure_cpu().expect("device→host transfer failed");
    let gpu_values = out_gpu.as_cpu_slice();
    assert_eq!(gpu_values.len(), cpu_values.len());

    let mut max_abs_diff = 0.0_f32;
    for (g, c) in gpu_values.iter().zip(cpu_values.iter()) {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }

    println!(
        "GPU residency MatMul vs CPU reference: max_abs_diff = {:.2e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 1e-4,
        "GPU residency MatMul drift {:.2e} exceeds 1e-4 envelope; \
         this would indicate a kernel reduction-order regression or \
         a residency-path correctness bug",
        max_abs_diff
    );
}

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn cuda_matmul_inplace_falls_back_to_host_on_mixed_storage() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    let m = 3_usize;
    let k = 4;
    let n = 2;

    let a_host: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();

    // a on Cpu, b on Cuda → mixed, must still produce correct output
    // via the host-path fallback inside `cuda_matmul_inplace`.
    let a_cpu = Tensor::new_cpu(vec![m, k], a_host.clone());
    let b_gpu = make_cuda_tensor_from_host(&b_host, &[k, n]);
    let mut out = Tensor::new_cpu(vec![m, n], vec![0.0; m * n]);

    cuda_matmul_inplace(&a_cpu, &b_gpu, &mut out, m, k, n);

    // Reference.
    let b_cpu = Tensor::new_cpu(vec![k, n], b_host);
    let cpu_out = cpu_matmul(&a_cpu, &b_cpu);

    let max_abs_diff = out
        .as_cpu_slice()
        .iter()
        .zip(cpu_out.as_cpu_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "GPU mixed-storage fallback vs CPU: max_abs_diff = {:.2e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 1e-4,
        "mixed-storage fallback drift {:.2e} exceeds 1e-4",
        max_abs_diff
    );
}
