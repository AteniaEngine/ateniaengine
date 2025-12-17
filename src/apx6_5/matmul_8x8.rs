use std::arch::x86_64::*;

#[target_feature(enable = "avx2", enable = "fma")]
fn microkernel_8x8(
    a: *const f32, // A_panel [8 x k] row-major
    b: *const f32, // B_panel [k x 8] col-major (k contiguous)
    c: *mut f32,   // out tile [8 x ldc]
    ldc: usize,
    k: usize,
) {
    let mut c00 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c40 = _mm256_setzero_ps();
    let mut c50 = _mm256_setzero_ps();
    let mut c60 = _mm256_setzero_ps();
    let mut c70 = _mm256_setzero_ps();

    for p in 0..k {
        let bvec = unsafe { _mm256_loadu_ps(b.add(p * 8)) };

        let (a0, a1, a2, a3, a4, a5, a6, a7) = unsafe {
            (
                _mm256_set1_ps(*a.add(0 * k + p)),
                _mm256_set1_ps(*a.add(1 * k + p)),
                _mm256_set1_ps(*a.add(2 * k + p)),
                _mm256_set1_ps(*a.add(3 * k + p)),
                _mm256_set1_ps(*a.add(4 * k + p)),
                _mm256_set1_ps(*a.add(5 * k + p)),
                _mm256_set1_ps(*a.add(6 * k + p)),
                _mm256_set1_ps(*a.add(7 * k + p)),
            )
        };

        c00 = _mm256_fmadd_ps(a0, bvec, c00);
        c10 = _mm256_fmadd_ps(a1, bvec, c10);
        c20 = _mm256_fmadd_ps(a2, bvec, c20);
        c30 = _mm256_fmadd_ps(a3, bvec, c30);
        c40 = _mm256_fmadd_ps(a4, bvec, c40);
        c50 = _mm256_fmadd_ps(a5, bvec, c50);
        c60 = _mm256_fmadd_ps(a6, bvec, c60);
        c70 = _mm256_fmadd_ps(a7, bvec, c70);
    }

    unsafe {
        _mm256_storeu_ps(c.add(0 * ldc), c00);
        _mm256_storeu_ps(c.add(1 * ldc), c10);
        _mm256_storeu_ps(c.add(2 * ldc), c20);
        _mm256_storeu_ps(c.add(3 * ldc), c30);
        _mm256_storeu_ps(c.add(4 * ldc), c40);
        _mm256_storeu_ps(c.add(5 * ldc), c50);
        _mm256_storeu_ps(c.add(6 * ldc), c60);
        _mm256_storeu_ps(c.add(7 * ldc), c70);
    }
}

pub fn matmul_tiled_6_5(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let bm = 64usize;
    let bn = 64usize;

    // Inicializar salida a cero (forward puro).
    for v in out.iter_mut() {
        *v = 0.0;
    }

    for i0 in (0..m).step_by(bm) {
        let i_max = (i0 + bm).min(m);
        for j0 in (0..n).step_by(bn) {
            let j_max = (j0 + bn).min(n);

            // Bloques internos 8x8 dentro del tile 64x64.
            for ib in (i0..i_max).step_by(8) {
                let i_block_max = (ib + 8).min(i_max);
                for jb in (j0..j_max).step_by(8) {
                    let j_block_max = (jb + 8).min(j_max);

                    // Si no es un bloque completo 8x8, usar fallback escalar seguro.
                    if i_block_max - ib < 8 || j_block_max - jb < 8 {
                        for i in ib..i_block_max {
                            for j in jb..j_block_max {
                                let mut acc = 0.0f32;
                                for p in 0..k {
                                    acc += a[i * k + p] * b[p * n + j];
                                }
                                out[i * n + j] = acc;
                            }
                        }
                        continue;
                    }

                    let k_local = k;

                    // Packing de A: 8 filas x k_local, row-major contiguo.
                    let mut a_panel = vec![0f32; 8 * k_local];
                    for r in 0..8 {
                        let src_row = ib + r;
                        for p in 0..k_local {
                            a_panel[r * k_local + p] = a[src_row * k + p];
                        }
                    }

                    // Packing de B: k_local x 8, col-major respecto a N, K contigua.
                    let mut b_panel = vec![0f32; k_local * 8];
                    for p in 0..k_local {
                        for c in 0..8 {
                            let col = jb + c;
                            b_panel[p * 8 + c] = b[p * n + col];
                        }
                    }

                    let c_ptr = unsafe { out.as_mut_ptr().add(ib * n + jb) };

                    unsafe {
                        microkernel_8x8(
                            a_panel.as_ptr(),
                            b_panel.as_ptr(),
                            c_ptr,
                            n,
                            k_local,
                        );
                    }
                }
            }
        }
    }
}
