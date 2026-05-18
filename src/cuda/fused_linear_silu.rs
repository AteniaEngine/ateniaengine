use std::os::raw::c_int;

#[cfg(atenia_cuda)]
use crate::cuda::pool_helpers::with_pooled_device_buffers;
#[cfg(atenia_cuda)]
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::Tensor;
#[cfg(atenia_cuda)]
use crate::tensor::TensorStorage;

#[cfg(atenia_cuda)]
#[link(name = "fused_linear_silu", kind = "static")]
unsafe extern "C" {
    // Device-pointer variant. See the doc comment on
    // `launch_linear_f32_device_ptrs` in `linear.rs` for the ownership
    // contract. Returns 0 on success, 2 on kernel launch error, 1 on
    // sync error. Wired via the `all_cuda` dispatch in
    // `cuda_fused_linear_silu` AND reused by the CPU-path through
    // `with_pooled_device_buffers`.
    pub(crate) fn launch_fused_linear_silu_f32_device_ptrs(
        d_x: *const f32,
        d_w: *const f32,
        d_b: *const f32,
        d_out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> i32;
}

// **CPU-2 C2a** — CUDA-less build: no `fused_linear_silu` static
// lib. Identical-signature stub; unreachable because the only
// callers (`cuda_fused_linear_silu` / `..._raw`) are CUDA-gated.
#[cfg(not(atenia_cuda))]
#[allow(dead_code, unused_variables)]
pub(crate) unsafe fn launch_fused_linear_silu_f32_device_ptrs(
    d_x: *const f32,
    d_w: *const f32,
    d_b: *const f32,
    d_out: *mut f32,
    m: c_int,
    k: c_int,
    n: c_int,
) -> i32 {
    unreachable!(
        "CUDA symbol launch_fused_linear_silu_f32_device_ptrs called in CPU-only build (atenia_cuda not enabled)"
    )
}

/// CUDA fused Linear + SiLU op: computes `out = silu(x @ w + b)`.
///
/// Dispatches on storage just like [`super::linear::cuda_linear`]: all
/// inputs Cuda → device-pointer variant; otherwise → host path, which
/// panics on mixed/partial Cuda storage via `as_cpu_slice`.
#[cfg(atenia_cuda)]
pub fn cuda_fused_linear_silu(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let all_cuda = matches!(
        (&x.storage, &w.storage, &b.storage, &out.storage),
        (
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
        )
    );

    if all_cuda {
        let d_x = cuda_device_ptr(&x.storage);
        let d_w = cuda_device_ptr(&w.storage);
        let d_b = cuda_device_ptr(&b.storage);
        let d_out = cuda_device_ptr_mut(&out.storage);

        let rc = unsafe {
            launch_fused_linear_silu_f32_device_ptrs(
                d_x, d_w, d_b, d_out, m as c_int, k as c_int, n as c_int,
            )
        };

        if rc != 0 {
            panic!(
                "cuda_fused_linear_silu device_ptrs path failed with code {}: \
                 1 = sync failure, 2 = launch failure. This indicates \
                 a CUDA driver issue or an invalid kernel invocation.",
                rc
            );
        }
    } else {
        cuda_fused_linear_silu_raw(
            x.as_cpu_slice(),
            w.as_cpu_slice(),
            b.as_cpu_slice(),
            out.as_cpu_slice_mut(),
            m,
            k,
            n,
        );
    }
}

// **CPU-2 C2a** — CUDA-less build. No pure-CPU path (the
// non-`all_cuda` branch also routes through
// `with_pooled_device_buffers`). Only reached via the executor's
// GPU dispatch (guarded by `cuda_available()`), so an unreachable
// stub with the identical signature is the CPU-only contract.
#[cfg(not(atenia_cuda))]
#[allow(unused_variables)]
pub fn cuda_fused_linear_silu(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    unreachable!(
        "cuda_fused_linear_silu called in CPU-only build (atenia_cuda not enabled)"
    )
}

#[cfg(atenia_cuda)]
fn cuda_fused_linear_silu_raw(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(x.len(), m * k, "cuda_fused_linear_silu: bad X size");
    assert_eq!(w.len(), k * n, "cuda_fused_linear_silu: bad W size");
    assert_eq!(b.len(), n, "cuda_fused_linear_silu: bad B size");
    assert_eq!(out.len(), m * n, "cuda_fused_linear_silu: bad OUT size");

    let m_ci = m as c_int;
    let k_ci = k as c_int;
    let n_ci = n as c_int;

    let result = unsafe {
        with_pooled_device_buffers(&[x, w, b], out, |d_in, d_out| {
            launch_fused_linear_silu_f32_device_ptrs(
                d_in[0], d_in[1], d_in[2], d_out, m_ci, k_ci, n_ci,
            )
        })
    };

    if let Err(e) = result {
        panic!("cuda_fused_linear_silu_raw failed: {:?}", e);
    }
}
