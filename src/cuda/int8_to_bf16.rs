//! M9.1 — Rust wrapper around the INT8 → BF16 per-channel dequant
//! kernel that lives in `src/cuda/int8_to_bf16.cu`.
//!
//! The kernel itself is the M9.0 microbench primitive (validated H2
//! PASS, ~2× speedup over Path B M8.4c on the four Llama 2 13B decode
//! shapes). M9.1 wraps it for use from the engine's loader / dispatch
//! paths in the same shape `bf16_to_f32_resident_in_vram` did for the
//! M8.4 BF16 → F32 upcast: take a host-side INT8 buffer + per-output-
//! channel F32 scales, upload everything to VRAM, dispatch the dequant
//! kernel, and return a BF16-resident [`TensorGPU`] whose buffer holds
//! the materialised BF16 weight.
//!
//! ## Why "to BF16" and not "to INT8 in VRAM"
//!
//! [`crate::gpu::tensor::TensorGPU`] cannot represent an INT8 buffer
//! today — the dtype field is `F32` / `BF16` only. Adding INT8 as a
//! first-class GPU dtype would touch every consumer; M9.1 deliberately
//! avoids that surface area. Instead, the M9.0 dispatch contract
//! (BF16-input cuBLAS GEMM with Tensor Cores) consumes a BF16 weight,
//! so the wrapper materialises BF16 in VRAM and returns it.
//!
//! That's the **load-time path**: M9.2 will switch to a per-matmul
//! dequant-into-transient pattern that keeps the INT8 source resident
//! and re-dequantises into a recycled BF16 staging slot per call (the
//! "true" capacity-saving path that delivers the 24 GiB → 12 GiB win
//! on Llama 2 13B). M9.1's primitive is the building block; M9.2 wires
//! it.
//!
//! ## Counter
//!
//! [`INT8_RESIDENT_COUNT`] mirrors `BF16_RESIDENT_COUNT` (M8.1) — every
//! successful upload + dequant increments by 1. Tests assert delta to
//! verify routing.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::gpu::tensor::TensorGPU;

/// **M9.1** — global counter advanced by every successful
/// [`int8_to_bf16_in_vram`] call. Mirrors the M8.1
/// `BF16_RESIDENT_COUNT` discipline; tests that pin the routing
/// snapshot before/after deltas.
static INT8_RESIDENT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Read-only accessor for [`INT8_RESIDENT_COUNT`].
pub fn cuda_int8_resident_count() -> usize {
    INT8_RESIDENT_COUNT.load(Ordering::Relaxed)
}

/// Test-only mutex serialising any test that mutates or asserts on
/// [`INT8_RESIDENT_COUNT`]. Same role as
/// `crate::cuda::bf16_to_f32::BF16_COUNTER_TEST_LOCK`.
#[cfg(test)]
pub(crate) static INT8_COUNTER_TEST_LOCK: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

#[link(name = "int8_to_bf16", kind = "static")]
unsafe extern "C" {
    fn int8_to_bf16_per_channel_launch_device(
        d_int8: *const c_void,
        d_scales: *const f32,
        d_bf16: *mut c_void,
        k: c_int,
        n: c_int,
    ) -> c_int;

    fn int8_to_bf16_per_group_launch_device(
        d_int8: *const c_void,
        d_scales: *const f32,
        d_bf16: *mut c_void,
        k: c_int,
        n: c_int,
        group_size: c_int,
    ) -> c_int;
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
}

unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut c_void, bytes: usize);
    fn cuda_free(ptr: *mut c_void);
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;

/// RAII guard mirroring `cuda::bf16_to_f32::DeviceAllocs`: every
/// successful raw `cuda_malloc` is freed on `Drop`. The BF16 output
/// is *not* registered here — that buffer is owned by the returned
/// [`TensorGPU`] and survives the function via its own `Drop`.
struct ScratchAllocs {
    ptrs: Vec<*mut c_void>,
}

impl ScratchAllocs {
    fn new() -> Self {
        Self { ptrs: Vec::with_capacity(2) }
    }

    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        let mut p: *mut c_void = std::ptr::null_mut();
        unsafe {
            cuda_malloc(&mut p, bytes);
        }
        if p.is_null() {
            return None;
        }
        self.ptrs.push(p);
        Some(p)
    }
}

impl Drop for ScratchAllocs {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            unsafe {
                cuda_free(p);
            }
        }
    }
}

/// **M9.1** — upload a host-side INT8 weight + per-output-channel F32
/// scales to VRAM, dispatch the M9.0 dequant kernel, and return a
/// BF16-resident [`TensorGPU`] holding the materialised weight.
///
/// `q.len()` must equal `product(shape)`; `scales.len()` must equal
/// `shape[shape.len() - 1]` (one F32 per output column). The shape
/// must be at least rank-1; for the standard 2D matmul weight,
/// `shape = [K, N]` row-major.
///
/// On success: `INT8_RESIDENT_COUNT` is incremented by 1 and the
/// returned `TensorGPU` carries `dtype = BF16` and
/// `(rows, cols) = (numel, 1)` (matching the convention of
/// `bf16_to_vram_no_upcast`). On any failure (cuda unavailable,
/// length mismatch, alloc failure, copy failure, kernel launch
/// failure) returns `None` and the counter is **not** advanced —
/// every transient buffer is freed via `ScratchAllocs::Drop` and
/// `TensorGPU::Drop` regardless of the early-return path.
pub fn int8_to_bf16_in_vram(
    q: &[i8],
    scales: &[f32],
    shape: &[usize],
) -> Option<TensorGPU> {
    if !super::cuda_available() {
        return None;
    }
    if shape.is_empty() {
        return None;
    }

    let numel: usize = shape.iter().product();
    if q.len() != numel {
        return None;
    }
    let n = *shape.last().unwrap();
    if scales.len() != n {
        return None;
    }
    let k = if n == 0 { 0 } else { numel / n };
    if numel == 0 {
        return None;
    }

    let bytes_int8 = numel; // 1 byte per element
    let bytes_scales = n * std::mem::size_of::<f32>();

    let mut scratch = ScratchAllocs::new();
    let d_int8 = scratch.alloc(bytes_int8)?;
    let d_scales = scratch.alloc(bytes_scales)?;

    // BF16 destination is owned through TensorGPU's RAII Drop so
    // any failure beyond this point frees it cleanly.
    let gpu_bf16 = TensorGPU::new_bf16(numel).ok()?;

    unsafe {
        // H→D: INT8 weight bytes.
        let rc = cudaMemcpy(
            d_int8,
            q.as_ptr() as *const c_void,
            bytes_int8,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }

        // H→D: per-channel scales.
        let rc = cudaMemcpy(
            d_scales,
            scales.as_ptr() as *const c_void,
            bytes_scales,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }

        // Dispatch dequant kernel writing into the BF16 device buffer.
        let rc = int8_to_bf16_per_channel_launch_device(
            d_int8,
            d_scales as *const f32,
            gpu_bf16.device_ptr() as *mut c_void,
            k as c_int,
            n as c_int,
        );
        if rc != 0 {
            return None;
        }
    }

    INT8_RESIDENT_COUNT.fetch_add(1, Ordering::Relaxed);
    Some(gpu_bf16)
    // `scratch` drops here — INT8 + scales staging is freed.
    // `gpu_bf16` moved into the return value, survives.
}

/// **M9.4** — per-group analogue of [`int8_to_bf16_in_vram`].
///
/// `q.len()` must equal `product(shape)`; `scales.len()` must
/// equal `ceil(K / group_size) * N` where
/// `K = product(shape[..-1])` and `N = shape[-1]`. Same RAII +
/// cleanup contract as the per-channel variant. Increments
/// [`INT8_RESIDENT_COUNT`] on success; counter remains untouched
/// on any failure.
pub fn int8_per_group_to_bf16_in_vram(
    q: &[i8],
    scales: &[f32],
    shape: &[usize],
    group_size: usize,
) -> Option<TensorGPU> {
    if !super::cuda_available() {
        return None;
    }
    if shape.is_empty() || group_size == 0 {
        return None;
    }

    let numel: usize = shape.iter().product();
    if q.len() != numel {
        return None;
    }
    let n = *shape.last().unwrap();
    if n == 0 {
        return None;
    }
    let k: usize = numel / n;
    let num_groups = (k + group_size - 1) / group_size;
    if scales.len() != num_groups * n {
        return None;
    }
    if numel == 0 {
        return None;
    }

    let bytes_int8 = numel;
    let bytes_scales = scales.len() * std::mem::size_of::<f32>();

    let mut scratch = ScratchAllocs::new();
    let d_int8 = scratch.alloc(bytes_int8)?;
    let d_scales = scratch.alloc(bytes_scales)?;

    let gpu_bf16 = TensorGPU::new_bf16(numel).ok()?;

    unsafe {
        let rc = cudaMemcpy(
            d_int8,
            q.as_ptr() as *const c_void,
            bytes_int8,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }
        let rc = cudaMemcpy(
            d_scales,
            scales.as_ptr() as *const c_void,
            bytes_scales,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }

        let rc = int8_to_bf16_per_group_launch_device(
            d_int8,
            d_scales as *const f32,
            gpu_bf16.device_ptr() as *mut c_void,
            k as c_int,
            n as c_int,
            group_size as c_int,
        );
        if rc != 0 {
            return None;
        }
    }

    INT8_RESIDENT_COUNT.fetch_add(1, Ordering::Relaxed);
    Some(gpu_bf16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `int8_to_bf16_in_vram` returns `None` for length mismatches
    /// without touching the counter.
    #[test]
    fn rejects_mismatched_q_length() {
        let _lock = INT8_COUNTER_TEST_LOCK.lock().unwrap();
        let before = cuda_int8_resident_count();
        // shape [2, 3] → product 6, but q has 5.
        let q = vec![0_i8; 5];
        let scales = vec![1.0_f32; 3];
        let r = int8_to_bf16_in_vram(&q, &scales, &[2, 3]);
        assert!(r.is_none(), "must return None on q.len() mismatch");
        assert_eq!(cuda_int8_resident_count(), before,
            "counter must not advance on mismatched input");
    }

    #[test]
    fn rejects_mismatched_scales_length() {
        let _lock = INT8_COUNTER_TEST_LOCK.lock().unwrap();
        let before = cuda_int8_resident_count();
        // shape [2, 3] → scales must have len 3, but we pass 2.
        let q = vec![0_i8; 6];
        let scales = vec![1.0_f32; 2];
        let r = int8_to_bf16_in_vram(&q, &scales, &[2, 3]);
        assert!(r.is_none(), "must return None on scales.len() mismatch");
        assert_eq!(cuda_int8_resident_count(), before,
            "counter must not advance on mismatched input");
    }

    #[test]
    fn rejects_empty_shape() {
        let _lock = INT8_COUNTER_TEST_LOCK.lock().unwrap();
        let before = cuda_int8_resident_count();
        let r = int8_to_bf16_in_vram(&[], &[], &[]);
        assert!(r.is_none());
        assert_eq!(cuda_int8_resident_count(), before);
    }

    /// On a CUDA-equipped host the round-trip uploads, dequantises
    /// on device, and returns a BF16 buffer whose downloaded bits
    /// match the host-side dequant of the same `(q, scales)` pair.
    /// Skipped when `cuda_available()` returns false.
    #[test]
    fn round_trip_advances_counter_and_matches_host_dequant() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let _lock = INT8_COUNTER_TEST_LOCK.lock().unwrap();

        // Small but non-trivial shape: K=4, N=3.
        let shape = [4, 3];
        let q: Vec<i8> = vec![
             10,  -5,  20,
            -127,   0,  64,
              5,  100, -64,
             32,   -1,   1,
        ];
        let scales: Vec<f32> = vec![0.1, 0.05, 0.025];

        let before = cuda_int8_resident_count();
        let gpu = int8_to_bf16_in_vram(&q, &scales, &shape)
            .expect("upload must succeed on CUDA-equipped host");
        assert_eq!(
            cuda_int8_resident_count(), before + 1,
            "counter must advance by exactly 1"
        );

        // Download BF16 bits and compare against host dequant.
        let bits = gpu.to_cpu_bf16_bits()
            .expect("BF16 download must succeed");
        assert_eq!(bits.len(), q.len());

        // Host dequant: (q[k,n] as f32) * scales[n], then BF16 truncate.
        let n = shape[1];
        for idx in 0..q.len() {
            let col = idx % n;
            let f = (q[idx] as f32) * scales[col];
            let expected_bf16 = (f.to_bits() >> 16) as u16;
            assert_eq!(
                bits[idx], expected_bf16,
                "dequant mismatch at idx {idx}: got 0x{:04x}, expected 0x{:04x}",
                bits[idx], expected_bf16,
            );
        }
    }
}
