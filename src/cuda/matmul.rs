use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
use crate::cuda::pool_helpers::with_pooled_device_buffers;
use crate::tensor::{Device, Tensor};

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

pub fn is_cuda_available_for(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear
    )
}
