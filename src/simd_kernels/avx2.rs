use std::arch::x86_64::*;

/// M4.8.c — bulk BF16 → F32 decode using AVX2 (8 lanes per
/// step, 16 elements per loop iteration via two adjacent
/// 256-bit registers).
///
/// BF16 is the upper 16 bits of an F32; the decode is the
/// reverse of a truncation: zero-extend each `u16` to a `u32`
/// and shift left 16 bits. Per-element this is two cycles on
/// modern Intel; eight lanes at a time pushes the kernel to
/// memory bandwidth on the dev box.
///
/// `dst` must have `bits.len()` capacity. The function writes
/// `bits.len()` `f32` elements at the start of `dst`,
/// overwriting whatever was there. Trailing tail elements
/// (`len % 16`) are decoded scalar-style.
///
/// # Safety
/// The caller is responsible for verifying that the running
/// CPU supports AVX2; the registry/dispatcher seam in
/// `lib.rs:init_kernels` already gates on
/// `is_x86_feature_detected!("avx2")`. Inside this function
/// AVX2 is assumed; calling it on a non-AVX2 CPU is undefined
/// behaviour. The bulk loop uses unaligned loads / stores —
/// `Vec<u16>` and `Vec<f32>` from the standard allocator are
/// only 16-byte aligned by default on x86_64, but
/// `_mm256_loadu_*` / `_mm256_storeu_*` accept any alignment
/// at no measured cost on Raptor Lake.
#[target_feature(enable = "avx2")]
pub unsafe fn bf16_decode_avx2(bits: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(
        bits.len(),
        dst.len(),
        "bf16_decode_avx2: bits and dst must have equal length"
    );

    let n = bits.len();
    let mut i = 0;

    // Process 16 BF16 → 16 F32 per iteration: load a 256-bit
    // chunk of u16, split into low 8 / high 8, zero-extend
    // each half to u32 in 256-bit register, shift left 16,
    // reinterpret bits as f32, store.
    while i + 16 <= n {
        unsafe {
            let bits_ptr = bits.as_ptr().add(i) as *const __m256i;
            let chunk = _mm256_loadu_si256(bits_ptr);

            // Lower 128 bits of `chunk` → 8× u16 → 8× u32
            // (each u16 zero-extended into a 32-bit lane).
            let lo16 = _mm256_castsi256_si128(chunk);
            let hi16 = _mm256_extracti128_si256(chunk, 1);

            let lo32 = _mm256_cvtepu16_epi32(lo16);
            let hi32 = _mm256_cvtepu16_epi32(hi16);

            // Shift each lane left by 16 to reconstruct the
            // F32 bit pattern from the BF16 upper-half.
            let lo_shifted = _mm256_slli_epi32(lo32, 16);
            let hi_shifted = _mm256_slli_epi32(hi32, 16);

            // Reinterpret as f32 and store. `_mm256_castsi256_ps`
            // is a no-op at the hardware level (just a type
            // bitcast for the type-checker).
            let lo_f32 = _mm256_castsi256_ps(lo_shifted);
            let hi_f32 = _mm256_castsi256_ps(hi_shifted);

            let dst_ptr = dst.as_mut_ptr().add(i);
            _mm256_storeu_ps(dst_ptr, lo_f32);
            _mm256_storeu_ps(dst_ptr.add(8), hi_f32);
        }
        i += 16;
    }

    // Scalar tail.
    while i < n {
        // Same math as `bf16_bits_to_f32` (tensor.rs:139).
        dst[i] = f32::from_bits((bits[i] as u32) << 16);
        i += 1;
    }
}

/// M4.8.c — runtime-dispatched bulk BF16 → F32 decode.
///
/// On AVX2-capable CPUs (`is_x86_feature_detected!("avx2")`
/// returns true), this calls `bf16_decode_avx2` and benefits
/// from the 8-lane SIMD path. On non-AVX2 CPUs (very rare on
/// any x86_64 hardware sold since 2013) this falls back to a
/// scalar loop matching the pre-M4.8.c behaviour at
/// `tensor.rs:729`.
///
/// `dst` must have `bits.len()` capacity. Writes overwrite
/// whatever was there.
pub fn bf16_decode_bulk(bits: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(
        bits.len(),
        dst.len(),
        "bf16_decode_bulk: bits and dst must have equal length"
    );
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            bf16_decode_avx2(bits, dst);
        }
    } else {
        for (out_v, &b) in dst.iter_mut().zip(bits.iter()) {
            *out_v = f32::from_bits((b as u32) << 16);
        }
    }
}

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
