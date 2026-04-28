use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
use crate::cuda::pool_helpers::with_pooled_device_buffers;
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::{Device, Tensor, TensorStorage};

#[link(name = "matmul_kernel")]
unsafe extern "C" {
    fn matmul_f32_launch_device(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );
}

/// CPU-path CUDA matmul: copies host operands to VRAM, launches
/// `matmul_f32_launch_device`, and copies the output back. All
/// alloc / H↔D / free bookkeeping lives in
/// [`crate::cuda::pool_helpers::with_pooled_device_buffers`] so this
/// function only has to describe the kernel-specific invocation.
///
/// Panics on pool exhaustion or CUDA-driver failure with a
/// [`crate::tensor::StorageTransferError`] diagnostic in the message.
/// The panic is consistent with the rest of the CPU-path CUDA ops,
/// which do not propagate `Result` up because their callers assign
/// the return tensor directly into the graph node's `output` slot.
pub fn cuda_matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let m_ci = m as c_int;
    let k_ci = k as c_int;
    let n_ci = n as c_int;

    let result = unsafe {
        with_pooled_device_buffers(
            &[a.as_cpu_slice(), b.as_cpu_slice()],
            out.as_cpu_slice_mut(),
            |d_in, d_out| {
                // `matmul_f32_launch_device` returns void; it prints
                // to stderr and returns without signalling on kernel
                // launch / sync failure (pre-existing limitation from
                // before the `_device_ptrs` return-code convention).
                // Report success unconditionally.
                matmul_f32_launch_device(d_in[0], d_in[1], d_out, m_ci, k_ci, n_ci);
                0
            },
        )
    };

    if let Err(e) = result {
        panic!("cuda_matmul failed: {:?}", e);
    }

    out
}

/// Residency-aware CUDA matmul (M4.7.3.a).
///
/// Mirrors the `all_cuda` dispatch pattern from
/// [`crate::cuda::linear::cuda_linear`]: when every operand and the
/// output buffer live on VRAM, the device-pointer launcher is called
/// directly with no host↔device traffic. Falls through to
/// [`cuda_matmul`] (CPU-roundtrip) when any operand is host-resident,
/// preserving the existing CPU-path behaviour byte-for-byte for
/// callers that have not adopted residency yet.
///
/// `out` is mutated in place. Caller is responsible for constructing
/// `out` with `TensorStorage::Cuda` before calling this — see
/// [`Tensor::zeros_new_cuda`].
///
/// The underlying `matmul_f32_launch_device` returns `void`; on
/// kernel launch / sync failure it prints to stderr but does not
/// signal back. This is a pre-existing ABI limitation versus the
/// `_device_ptrs` return-code convention used by `cuda_linear` and
/// friends. M4.7.3 trusts the launcher and assumes success; promoting
/// the kernel to a `_device_ptrs` ABI variant returning `i32` is a
/// known follow-up.
pub fn cuda_matmul_inplace(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
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

        unsafe {
            matmul_f32_launch_device(
                d_a,
                d_b,
                d_out,
                m as c_int,
                k as c_int,
                n as c_int,
            );
        }
    } else {
        // Mixed or all-Cpu storage: delegate to the CPU-roundtrip
        // path. `cuda_matmul` reads operands via `as_cpu_slice`,
        // which would panic on a Cuda operand, so we materialise
        // local Cpu clones first. This branch is unreachable on the
        // Llama hot path (the executor's `ensure_decoded` keeps
        // operand storage uniform — both Cuda or both Cpu), but
        // the fallback exists so `cuda_matmul_inplace` is correct
        // for any legal `(a, b, out)` triple a future caller may
        // produce. The output likewise gets normalised to Cpu
        // before the `clone_from_slice` write-back.
        let mut a_local = a.clone();
        let mut b_local = b.clone();
        a_local
            .ensure_cpu()
            .expect("cuda_matmul_inplace fallback: ensure_cpu on operand A failed");
        b_local
            .ensure_cpu()
            .expect("cuda_matmul_inplace fallback: ensure_cpu on operand B failed");
        let computed = cuda_matmul(&a_local, &b_local, m, k, n);
        out.ensure_cpu()
            .expect("cuda_matmul_inplace fallback: ensure_cpu on output failed");
        out.as_cpu_slice_mut()
            .clone_from_slice(computed.as_cpu_slice());
    }
}

pub fn is_cuda_available_for(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear
    )
}
