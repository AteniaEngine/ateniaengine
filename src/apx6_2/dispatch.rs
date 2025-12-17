pub fn dispatch_matmul_avx2(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        crate::apx6_2::avx2_matmul::matmul_avx2_f32(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m,
            k,
            n,
        );
        return;
    }

    // Fallback seguro a dispatcher actual (APX 3.8 CPU)
    crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
}
