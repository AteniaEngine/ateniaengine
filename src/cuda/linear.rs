use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
use crate::tensor::Tensor;

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

    // Device-pointer variant added in M3-d.4.D. Assumes the caller owns
    // the VRAM backing every pointer; does not alloc or free. Returns
    // 0 on success, 2 on kernel launch error, 1 on sync error.
    // `#[allow(dead_code)]` until M3-d.4.E wires it into the public op
    // dispatch; the symbol must still be declared so the linker pulls
    // it in from the static library.
    #[allow(dead_code)]
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
/// Inputs and output are [`Tensor`]s on the CPU side. The call will
/// panic via [`Tensor::as_cpu_slice`] if any of them is GPU-resident
/// (`TensorStorage::Cuda`); callers that hold GPU-resident tensors
/// must invoke `ensure_cpu()` first. A future milestone (M3-d.4.D/E)
/// will add a device-pointer path that consumes `Cuda` storage
/// directly without the host roundtrip.
pub fn cuda_linear(
    a: &Tensor,
    b: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
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
