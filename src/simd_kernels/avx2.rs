use std::arch::x86_64::*;

pub unsafe fn matmul_avx2(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let mut j = 0;
        while j + 8 <= n {
            let mut acc = unsafe { _mm256_setzero_ps() };
            for p in 0..k {
                let a_val = unsafe { _mm256_set1_ps(a[i * k + p]) };
                let b_ptr = unsafe { b.as_ptr().add(p * n + j) };
                let b_vec = unsafe { _mm256_loadu_ps(b_ptr) };
                acc = unsafe { _mm256_fmadd_ps(a_val, b_vec, acc) };
            }
            unsafe {
                _mm256_storeu_ps(out.as_mut_ptr().add(i * n + j), acc);
            }
            j += 8;
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

pub unsafe fn batch_matmul_avx2(
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
        let a_slice = &a[t * stride_a..(t + 1) * stride_a];
        let b_slice = &b[t * stride_b..(t + 1) * stride_b];
        let out_slice = &mut out[t * stride_out..(t + 1) * stride_out];
        unsafe {
            matmul_avx2(a_slice, b_slice, out_slice, m, k, n);
        }
    }
}
