use std::os::raw::c_int;

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

/// CUDA fused Linear + SiLU kernel wrapper.
///
/// For now this operates on host slices (same modelo que cuda_linear),
/// usando Device::CPU de forma lógica.
pub fn cuda_fused_linear_silu(
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
