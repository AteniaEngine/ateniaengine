use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::{Tensor, TensorStorage};

#[link(name = "linear_cuda", kind = "static")]
unsafe extern "C" {
    fn launch_linear_f32(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );

    // Device-pointer variant: assumes the caller owns the VRAM backing
    // every pointer; does not alloc or free. Returns 0 on success,
    // 2 on kernel launch error, 1 on sync error. Wired into the public
    // `cuda_linear` wrapper via an `all_cuda` dispatch.
    pub(crate) fn launch_linear_f32_device_ptrs(
        d_a: *const f32,
        d_b: *const f32,
        d_bias: *const f32,
        d_out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> i32;
}

/// CUDA Linear op: computes `out = a @ b + bias`.
///
/// Dispatches on storage: when every input and the output are
/// `TensorStorage::Cuda`, calls the device-pointer variant that skips
/// the H<->D roundtrip. In any other configuration (all-Cpu or mixed)
/// falls through to the host path via [`Tensor::as_cpu_slice`], which
/// panics on a Cuda operand with a message pointing the caller to
/// `ensure_cpu()`. Mixed storage is therefore treated as a setup bug,
/// not a silent transfer.
pub fn cuda_linear(
    a: &Tensor,
    b: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let all_cuda = matches!(
        (&a.storage, &b.storage, &bias.storage, &out.storage),
        (
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
        )
    );

    if all_cuda {
        let d_a = cuda_device_ptr(&a.storage);
        let d_b = cuda_device_ptr(&b.storage);
        let d_bias = cuda_device_ptr(&bias.storage);
        let d_out = cuda_device_ptr_mut(&out.storage);

        let rc = unsafe {
            launch_linear_f32_device_ptrs(
                d_a,
                d_b,
                d_bias,
                d_out,
                m as c_int,
                k as c_int,
                n as c_int,
            )
        };

        if rc != 0 {
            panic!(
                "cuda_linear device_ptrs path failed with code {}: \
                 1 = sync failure, 2 = launch failure. This indicates \
                 a CUDA driver issue or an invalid kernel invocation.",
                rc
            );
        }
    } else {
        cuda_linear_raw(
            a.as_cpu_slice(),
            b.as_cpu_slice(),
            bias.as_cpu_slice(),
            out.as_cpu_slice_mut(),
            m,
            k,
            n,
        );
    }
}

fn cuda_linear_raw(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(bias.len(), n);
    assert_eq!(out.len(), m * n);

    unsafe {
        launch_linear_f32(
            a.as_ptr(),
            b.as_ptr(),
            bias.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            k as c_int,
            n as c_int,
        );
    }
}

/// For APX 4.3 planning: return true only for ops with a real linear CUDA kernel.
pub fn is_cuda_available_for_linear(t: &NodeType) -> bool {
    matches!(
        t,
        NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear
    )
}
