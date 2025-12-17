use std::arch::x86_64::*;

#[inline(always)]
fn load8(ptr: *const f32) -> __m256 {
    unsafe { _mm256_loadu_ps(ptr) }
}

pub fn matmul_4x8_avx2(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let mr = 4usize; // rows of A
    let nr = 8usize; // cols of B

    let mut b_panel = vec![0f32; k * nr];

    for j in (0..n).step_by(nr) {
        if j + nr > n {
            // Para bordes no múltiplos de 8, caemos fuera por ahora.
            // El dispatcher sólo llamará a este kernel cuando n >= 256
            // y alineado a bloques grandes; los bordes se pueden manejar
            // por el fallback 6.3B.
            break;
        }
        let jb = j;

        for p in 0..k {
            unsafe {
                let src = b.add(p * n + jb);
                let dst = b_panel.as_mut_ptr().add(p * nr);
                std::ptr::copy_nonoverlapping(src, dst, nr);
            }
        }

        for i in (0..m).step_by(mr) {
            if i + mr > m {
                break;
            }
            let ib = i;

            let mut acc = unsafe {
                [
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                ]
            };

            for p in 0..k {
                let b_vec = load8(unsafe { b_panel.as_ptr().add(p * nr) });

                let (a0, a1, a2, a3) = unsafe {
                    (
                        _mm256_set1_ps(*a.add(ib * k + p)),
                        _mm256_set1_ps(*a.add((ib + 1) * k + p)),
                        _mm256_set1_ps(*a.add((ib + 2) * k + p)),
                        _mm256_set1_ps(*a.add((ib + 3) * k + p)),
                    )
                };

                acc[0] = unsafe { _mm256_fmadd_ps(a0, b_vec, acc[0]) };
                acc[1] = unsafe { _mm256_fmadd_ps(a1, b_vec, acc[1]) };
                acc[2] = unsafe { _mm256_fmadd_ps(a2, b_vec, acc[2]) };
                acc[3] = unsafe { _mm256_fmadd_ps(a3, b_vec, acc[3]) };
            }

            for r in 0..mr {
                unsafe {
                    _mm256_storeu_ps(out.add((ib + r) * n + jb), acc[r]);
                }
            }
        }
    }
}
