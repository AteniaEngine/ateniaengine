use std::os::raw::c_int;

use crate::cuda::pool_helpers::with_pooled_device_buffers;
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::{Tensor, TensorStorage};

#[link(name = "batch_matmul", kind = "static")]
unsafe extern "C" {
    // Device-pointer variant. See the doc comment on
    // `launch_linear_f32_device_ptrs` in `linear.rs` for the ownership
    // contract. Returns 0 on success, 2 on kernel launch error, 1 on
    // sync error. Wired via the `all_cuda` dispatch in `cuda_batch_matmul`
    // AND reused by the CPU-path through `with_pooled_device_buffers`.
    pub(crate) fn launch_batch_matmul_f32_device_ptrs(
        d_a: *const f32,
        d_b: *const f32,
        d_out: *mut f32,
        batch: c_int,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> i32;
}

/// CUDA batched matmul: computes `out[i] = a[i] @ b[i]` for each batch `i`.
///
/// Dispatches on storage just like [`super::linear::cuda_linear`]: all
/// inputs Cuda → device-pointer variant; otherwise → host path, which
/// panics on mixed/partial Cuda storage via `as_cpu_slice`.
pub fn cuda_batch_matmul(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let all_cuda = matches!(
        (&a.storage, &b.storage, &out.storage),
        (
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
        )
    );

    if all_cuda {
        let d_a = cuda_device_ptr(&a.storage);
        let d_b = cuda_device_ptr(&b.storage);
        let d_out = cuda_device_ptr_mut(&out.storage);

        let rc = unsafe {
            launch_batch_matmul_f32_device_ptrs(
                d_a,
                d_b,
                d_out,
                batch as c_int,
                m as c_int,
                k as c_int,
                n as c_int,
            )
        };

        if rc != 0 {
            panic!(
                "cuda_batch_matmul device_ptrs path failed with code {}: \
                 1 = sync failure, 2 = launch failure. This indicates \
                 a CUDA driver issue or an invalid kernel invocation.",
                rc
            );
        }
    } else {
        cuda_batch_matmul_raw(
            a.as_cpu_slice(),
            b.as_cpu_slice(),
            out.as_cpu_slice_mut(),
            batch,
            m,
            k,
            n,
        );
    }
}

fn cuda_batch_matmul_raw(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let stride_a = m * k;
    let stride_b = k * n;
    let stride_out = m * n;

    assert_eq!(a.len(), batch * stride_a);
    assert_eq!(b.len(), batch * stride_b);
    assert_eq!(out.len(), batch * stride_out);

    let b_ci = batch as c_int;
    let m_ci = m as c_int;
    let k_ci = k as c_int;
    let n_ci = n as c_int;

    let result = unsafe {
        with_pooled_device_buffers(&[a, b], out, |d_in, d_out| {
            launch_batch_matmul_f32_device_ptrs(d_in[0], d_in[1], d_out, b_ci, m_ci, k_ci, n_ci)
        })
    };

    if let Err(e) = result {
        panic!("cuda_batch_matmul_raw failed: {:?}", e);
    }
}
