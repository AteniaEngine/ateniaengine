use std::os::raw::c_int;

#[link(name = "batch_matmul", kind = "static")]
unsafe extern "C" {
    fn launch_batch_matmul_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        batch: c_int,
        m: c_int,
        k: c_int,
        n: c_int,
    );
}

pub fn cuda_batch_matmul(
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

    unsafe {
        launch_batch_matmul_f32(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            batch as c_int,
            m as c_int,
            k as c_int,
            n as c_int,
        );
    }
}
