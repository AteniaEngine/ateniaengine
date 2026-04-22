use std::os::raw::c_int;

use crate::tensor::Tensor;

#[link(name = "fused_linear_silu", kind = "static")]
unsafe extern "C" {
    fn launch_fused_linear_silu_f32(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );
}

/// CUDA fused Linear + SiLU op: computes `out = silu(x @ w + b)`.
///
/// Inputs and output are [`Tensor`]s on the CPU side. Panics via
/// [`Tensor::as_cpu_slice`] if any is GPU-resident; see the note on
/// [`cuda_linear`](super::linear::cuda_linear).
pub fn cuda_fused_linear_silu(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
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

    unsafe {
        launch_fused_linear_silu_f32(
            x.as_ptr(),
            w.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            k as c_int,
            n as c_int,
        );
    }
}
