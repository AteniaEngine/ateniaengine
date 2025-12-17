use std::os::raw::c_int;

use crate::amg::nodes::NodeType;

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
}

pub fn cuda_linear(
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
