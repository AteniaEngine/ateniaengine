use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn matmul_avx2_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let simd_width = 8; // 256 bits = 8 floats

    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = _mm256_setzero_ps();
                let mut p = 0;

                while p + simd_width <= k {
                    let a_vec = _mm256_loadu_ps(a.add(i * k + p));
                    // B is in row-major layout [k x n], so column j is b[p * n + j]
                    let b_vec = _mm256_loadu_ps(b.add(p * n + j));
                    acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                    p += simd_width;
                }

                // Horizontal reduce
                let mut tmp = [0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                let mut sum = tmp.iter().sum::<f32>();

                // Resto escalar
                while p < k {
                    sum += *a.add(i * k + p) * *b.add(p * n + j);
                    p += 1;
                }

                *out.add(i * n + j) = sum;
            }
        }
    }
}
