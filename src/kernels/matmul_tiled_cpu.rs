pub fn matmul_tiled_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let tile_m = 32;
    let tile_k = 32;
    let tile_n = 32;

    for mm in (0..m).step_by(tile_m) {
        for nn in (0..n).step_by(tile_n) {
            for kk in (0..k).step_by(tile_k) {
                let m_end = (mm + tile_m).min(m);
                let n_end = (nn + tile_n).min(n);
                let k_end = (kk + tile_k).min(k);

                for i in mm..m_end {
                    for j in nn..n_end {
                        let mut sum = 0.0;
                        for p in kk..k_end {
                            sum += a[i * k + p] * b[p * n + j];
                        }
                        out[i * n + j] += sum;
                    }
                }
            }
        }
    }
}
