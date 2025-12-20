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
            // For edges not multiples of 8, we bail out for now.
            // The dispatcher will only call this kernel when n >= 256
            // and aligned to large blocks; edges can be handled
            // by the 6.3B fallback.
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
