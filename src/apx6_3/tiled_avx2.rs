use std::arch::x86_64::{
    _mm_loadu_ps,
    _mm_fmadd_ps,
    _mm_prefetch,
    _MM_HINT_T0,
    _mm_setzero_ps,
    _mm_storeu_ps,
};

#[inline(always)]
pub fn matmul_tiled_avx2(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(out.len(), m * n);

    let tile_m = 64;
    let tile_n = 64;
    let unroll_k = 4;

    // Inicializar salida a cero
    for v in out.iter_mut() {
        *v = 0.0;
    }

    unsafe {
        for mm in (0..m).step_by(tile_m) {
            let m_end = (mm + tile_m).min(m);
            for nn in (0..n).step_by(tile_n) {
                let n_end = (nn + tile_n).min(n);

                for i in mm..m_end {
                    for j in nn..n_end {
                        let mut acc = _mm_setzero_ps();
                        let mut p = 0;

                        // Punteros base para esta fila/columna
                        let a_row_ptr = a.as_ptr().add(i * k);

                        while p + 4 <= k {
                            // Prefetch A y B un poco adelantados
                            _mm_prefetch(a_row_ptr.add(p) as *const i8, _MM_HINT_T0);
                            _mm_prefetch(b.as_ptr().add(p * n + j) as *const i8, _MM_HINT_T0);

                            let a_vec = _mm_loadu_ps(a_row_ptr.add(p));

                            // B está en [k x n], columna fija j: elementos b[(p+t)*n + j]
                            let mut b_tmp = [0.0f32; 4];
                            b_tmp[0] = *b.as_ptr().add((p + 0) * n + j);
                            b_tmp[1] = *b.as_ptr().add((p + 1) * n + j);
                            b_tmp[2] = *b.as_ptr().add((p + 2) * n + j);
                            b_tmp[3] = *b.as_ptr().add((p + 3) * n + j);
                            let b_vec = _mm_loadu_ps(b_tmp.as_ptr());

                            acc = _mm_fmadd_ps(a_vec, b_vec, acc);

                            p += unroll_k;
                        }

                        let mut tmp = [0.0f32; 4];
                        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
                        let mut sum = tmp.iter().sum::<f32>();

                        while p < k {
                            let a_val = *a_row_ptr.add(p);
                            let b_val = *b.as_ptr().add(p * n + j);
                            sum += a_val * b_val;
                            p += 1;
                        }

                        let out_idx = i * n + j;
                        out[out_idx] += sum;
                    }
                }
            }
        }
    }
}
