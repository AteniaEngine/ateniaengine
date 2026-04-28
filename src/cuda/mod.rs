use std::os::raw::c_int;

pub mod matmul;
pub mod linear;
pub mod batch_matmul;
pub mod fused_linear_silu;
pub(crate) mod pool_helpers;

#[link(name = "atenia_kernels", kind = "static")]
unsafe extern "C" {
    pub fn vec_add_cuda(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: c_int,
    );
}

pub fn cuda_available() -> bool {
    std::process::Command::new("nvidia-smi").output().is_ok()
}

pub fn vec_add_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());

    let n = a.len() as c_int;
    let mut out = vec![0.0f32; a.len()];

    unsafe {
        vec_add_cuda(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            n,
        );
    }

    out
}

/// Returns the raw device pointer (read-only) backing a
/// `TensorStorage::Cuda`. Panics with `unreachable!` on `Cpu` storage:
/// callers must guard the invocation with a prior check that the
/// storage variant is `Cuda` (typically an `all_cuda` match at the op
/// entry point).
pub(crate) fn cuda_device_ptr(
    storage: &crate::tensor::TensorStorage,
) -> *const f32 {
    match storage {
        crate::tensor::TensorStorage::Cuda(g) => g.device_ptr() as *const f32,
        crate::tensor::TensorStorage::Cpu(_) => {
            unreachable!("cuda_device_ptr called on Cpu storage")
        }
        crate::tensor::TensorStorage::Disk(_) => {
            unreachable!(
                "cuda_device_ptr called on Disk storage — the caller must \
                 verify via matches!(_, TensorStorage::Cuda(_)) before \
                 extracting the device pointer"
            )
        }
        crate::tensor::TensorStorage::CpuBf16(_) => {
            // M4.7.2 panic-stub: the GPU path will land in M4.7.3
            // (residency-aware kernels, BF16 host→device transfers).
            // Until then, callers that want to run a CpuBf16 param
            // through CUDA must transition the variant via
            // `Tensor::ensure_cpu` first (eager F32 upcast) and then
            // `ensure_gpu`. The Llama hot path in `graph.rs` does
            // not reach this function for CpuBf16 because the
            // executor decode-on-access pattern materialises a
            // transient F32 vec before any CUDA call.
            unreachable!(
                "cuda_device_ptr called on CpuBf16 storage — M4.7.3 \
                 (residency-aware GPU kernels with BF16 host transfers) \
                 has not landed yet; transition the variant via \
                 ensure_cpu then ensure_gpu first."
            )
        }
    }
}

/// Mutable counterpart of [`cuda_device_ptr`]. Same precondition.
pub(crate) fn cuda_device_ptr_mut(
    storage: &crate::tensor::TensorStorage,
) -> *mut f32 {
    match storage {
        crate::tensor::TensorStorage::Cuda(g) => g.device_ptr() as *mut f32,
        crate::tensor::TensorStorage::Cpu(_) => {
            unreachable!("cuda_device_ptr_mut called on Cpu storage")
        }
        crate::tensor::TensorStorage::Disk(_) => {
            unreachable!(
                "cuda_device_ptr_mut called on Disk storage — the caller must \
                 verify via matches!(_, TensorStorage::Cuda(_)) before \
                 extracting the device pointer"
            )
        }
        crate::tensor::TensorStorage::CpuBf16(_) => {
            // M4.7.2 panic-stub: see `cuda_device_ptr` for the
            // milestone breakdown. Mutating CpuBf16 in place would
            // also lose the BF16 precision contract; this call site
            // is unreachable on the Llama forward path.
            unreachable!(
                "cuda_device_ptr_mut called on CpuBf16 storage — M4.7.3 \
                 (residency-aware GPU kernels with BF16 host transfers) \
                 has not landed yet."
            )
        }
    }
}
