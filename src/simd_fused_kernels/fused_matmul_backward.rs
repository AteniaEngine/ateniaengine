pub fn fused_matmul_backward(
    a: &[f32],
    b: &[f32],
    grad_out: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    #[cfg(debug_assertions)]
    {
        eprintln!("[APX-2.5] fused_matmul_backward invoked");
        if std::env::var("ATENIA_TRACE").unwrap_or_default() == "1" {
            eprintln!("[APX TRACE] Using FUSED kernel matmul+rmsnorm");
        }
    }
    let mut da = vec![0.0; batch * m * k];
    let mut db = vec![0.0; batch * k * n];

    for t in 0..batch {
        let a_t = &a[t * m * k..][..m * k];
        let b_t = &b[t * k * n..][..k * n];
        let go = &grad_out[t * m * n..][..m * n];

        for i in 0..m {
            for p in 0..k {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += go[i * n + j] * b_t[p * n + j];
                }
                da[t * m * k + i * k + p] = acc;
            }
        }

        for p in 0..k {
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..m {
                    acc += a_t[i * k + p] * go[i * n + j];
                }
                db[t * k * n + p * n + j] = acc;
            }
        }
    }

    (da, db)
}
