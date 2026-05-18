use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
#[cfg(atenia_cuda)]
use crate::cuda::pool_helpers::with_pooled_device_buffers;
#[cfg(atenia_cuda)]
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::Tensor;
#[cfg(atenia_cuda)]
use crate::tensor::TensorStorage;

#[cfg(atenia_cuda)]
#[link(name = "linear_cuda", kind = "static")]
unsafe extern "C" {
    // Device-pointer variant: assumes the caller owns the VRAM backing
    // every pointer; does not alloc or free. Returns 0 on success,
    // 2 on kernel launch error, 1 on sync error. Wired into the public
    // `cuda_linear` wrapper via an `all_cuda` dispatch AND reused by
    // the CPU-path through `with_pooled_device_buffers`.
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

// **CPU-2 C2a** — CUDA-less build: no `linear_cuda` static lib.
// Identical-signature stub; unreachable because the only callers
// (`cuda_linear` / `cuda_linear_raw`) are themselves CUDA-gated.
#[cfg(not(atenia_cuda))]
#[allow(dead_code, unused_variables)]
pub(crate) unsafe fn launch_linear_f32_device_ptrs(
    d_a: *const f32,
    d_b: *const f32,
    d_bias: *const f32,
    d_out: *mut f32,
    m: c_int,
    k: c_int,
    n: c_int,
) -> i32 {
    unreachable!(
        "CUDA symbol launch_linear_f32_device_ptrs called in CPU-only build (atenia_cuda not enabled)"
    )
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
#[cfg(atenia_cuda)]
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
                d_a, d_b, d_bias, d_out, m as c_int, k as c_int, n as c_int,
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

// **CPU-2 C2a** — CUDA-less build. `cuda_linear` has no pure-CPU
// path: even its non-`all_cuda` branch routes through
// `with_pooled_device_buffers`, which is VRAM-backed. It is only
// reached via the executor's GPU dispatch (guarded by
// `cuda_available()`), so an unreachable stub with the identical
// signature is the correct CPU-only contract.
#[cfg(not(atenia_cuda))]
#[allow(unused_variables)]
pub fn cuda_linear(
    a: &Tensor,
    b: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    unreachable!(
        "cuda_linear called in CPU-only build (atenia_cuda not enabled)"
    )
}

#[cfg(atenia_cuda)]
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

    let m_ci = m as c_int;
    let k_ci = k as c_int;
    let n_ci = n as c_int;

    let result = unsafe {
        with_pooled_device_buffers(&[a, b, bias], out, |d_in, d_out| {
            launch_linear_f32_device_ptrs(d_in[0], d_in[1], d_in[2], d_out, m_ci, k_ci, n_ci)
        })
    };

    if let Err(e) = result {
        panic!("cuda_linear_raw failed: {:?}", e);
    }
}

/// For APX 4.3 planning: return true only for ops with a real linear CUDA kernel.
pub fn is_cuda_available_for_linear(t: &NodeType) -> bool {
    matches!(
        t,
        NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear
    )
}
