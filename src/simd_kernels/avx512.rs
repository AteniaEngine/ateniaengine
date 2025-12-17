use std::arch::x86_64::*;

pub unsafe fn matmul_avx512(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let mut j = 0;
        while j + 16 <= n {
            let mut acc = unsafe { _mm512_setzero_ps() };
            for p in 0..k {
                let a_val = unsafe { _mm512_set1_ps(a[i * k + p]) };
                let b_vec = unsafe { _mm512_loadu_ps(b.as_ptr().add(p * n + j)) };
                acc = unsafe { _mm512_fmadd_ps(a_val, b_vec, acc) };
            }
            unsafe {
                _mm512_storeu_ps(out.as_mut_ptr().add(i * n + j), acc);
            }
            j += 16;
        }

        while j < n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
            j += 1;
        }
    }
}

pub unsafe fn batch_matmul_avx512(
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

    for t in 0..batch {
        unsafe {
            matmul_avx512(
                &a[t * stride_a..],
                &b[t * stride_b..],
                &mut out[t * stride_out..],
                m,
                k,
                n,
            );
        }
    }
}
