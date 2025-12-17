pub fn matmul_tiled_flex(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bm: usize,
    bn: usize,
    bk: usize,
) {
    for i0 in (0..m).step_by(bm) {
        for j0 in (0..n).step_by(bn) {
            for p0 in (0..k).step_by(bk) {
                let imax = (i0 + bm).min(m);
                let jmax = (j0 + bn).min(n);
                let pmax = (p0 + bk).min(k);

                for i in i0..imax {
                    for j in j0..jmax {
                        let mut sum = 0.0;
                        for p in p0..pmax {
                            sum += a[i * k + p] * b[p * n + j];
                        }
                        out[i * n + j] += sum;
                    }
                }
            }
        }
    }
}
