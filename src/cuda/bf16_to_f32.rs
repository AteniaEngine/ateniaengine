//! M6 step 4a — safe Rust wrapper around the
//! `bf16_to_f32_launch_device` CUDA kernel (defined in
//! `src/cuda/bf16_to_f32.cu`).
//!
//! The kernel itself only operates on device pointers; this wrapper
//! owns the host↔device alloc/copy/free lifecycle so callers can
//! pass `&[u16]` (a BF16 buffer in host memory, e.g. an
//! `Arc<Vec<u16>>` slice from a `WeightStore`) and receive a fresh
//! `Vec<f32>` materialised from the device-side upcast.
//!
//! At this commit (4a) the function has no production caller — the
//! standalone `examples/test_bf16_upload.rs` validation already
//! exercised the kernel end to end with bit-exact agreement against
//! `simd_kernels::avx2::bf16_decode_bulk`. Sub-paths 4b → 4d will
//! add callers under the M6 wire-up plan.
//!
//! # Why this lives behind a safe wrapper
//!
//! Direct callers would otherwise have to redeclare the cudart and
//! `cuda_malloc`/`cuda_free` FFI symbols, manage error paths around
//! three potential failure points (`cudaMalloc` → `cudaMemcpy` H→D
//! → kernel → `cudaMemcpy` D→H → `cudaFree`), and remember the
//! `cuda_available()` guard. Encapsulating it here keeps every
//! future caller on the same vetted error-handling chain.

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::gpu::tensor::TensorGPU;

// FFI symbols needed by this module. `bf16_to_f32_launch_device`
// lives in the `bf16_to_f32` static library produced by
// `build.rs`. `cudaMemcpy` lives in `cudart`. `cuda_malloc` /
// `cuda_free` live in `atenia_kernels` (already linked elsewhere).
#[link(name = "bf16_to_f32", kind = "static")]
unsafe extern "C" {
    fn bf16_to_f32_launch_device(
        d_src_bf16: *const c_void,
        d_dst_f32: *mut f32,
        n: c_int,
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
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// RAII guard mirroring the pattern in
/// `cuda::matmul::cuda_matmul_non_pooled`: every successfully
/// allocated device buffer is freed on drop, so all `?` /
/// early-return branches release VRAM before propagating `None`.
struct DeviceAllocs {
    ptrs: Vec<*mut c_void>,
}

impl DeviceAllocs {
    fn new() -> Self {
        Self {
            ptrs: Vec::with_capacity(2),
        }
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

impl Drop for DeviceAllocs {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            unsafe {
                cuda_free(p);
            }
        }
    }
}

/// Upload a BF16 buffer to VRAM, run the GPU upcast kernel, and
/// download the resulting F32 buffer back to host memory.
///
/// `src` is a host-side BF16 buffer (e.g. a slice into an
/// `Arc<Vec<u16>>` from `WeightStore`). `dst_len` is the number of
/// F32 elements expected — must equal `src.len()` because the
/// kernel is a 1-to-1 element upcast. The parameter is kept
/// explicit in the public API to make the contract obvious at
/// callsites (a future BF16 → F32 packed/unpacked variant would
/// have a different ratio).
///
/// # Returns
///
/// `Some(Vec<f32>)` of length `dst_len` on success.
/// `None` if `cuda_available()` is false, any `cuda_malloc`
/// returns null, any `cudaMemcpy` returns non-zero, or the kernel
/// launcher returns non-zero. Every device buffer allocated up to
/// the failure point is freed by [`DeviceAllocs`]'s `Drop` before
/// `None` is returned, so failures do not leak VRAM.
///
/// # Behaviour contract
///
/// On success, every element of the returned `Vec<f32>` is
/// bit-exactly equal to the host AVX2 decode `f32::from_bits((src[i]
/// as u32) << 16)`. The CUDA `__bfloat162float()` intrinsic
/// implements the same bit-shift on Ampere+ silicon (RTX 4070 is
/// Ada). Bit-exactness was validated end to end against a 70.7M-
/// element FFN-down weight in `examples/test_bf16_upload.rs`.
pub fn bf16_to_f32_on_device(src: &[u16], dst_len: usize) -> Option<Vec<f32>> {
    debug_assert_eq!(
        src.len(),
        dst_len,
        "bf16_to_f32_on_device: kernel is 1-to-1; src.len() must equal dst_len"
    );

    if !super::cuda_available() {
        return None;
    }

    let bytes_bf16 = src.len() * std::mem::size_of::<u16>();
    let bytes_f32 = dst_len * std::mem::size_of::<f32>();

    let mut allocs = DeviceAllocs::new();
    let d_bf16 = allocs.alloc(bytes_bf16)?;
    let d_f32 = allocs.alloc(bytes_f32)?;

    unsafe {
        // Upload BF16 H→D.
        let rc = cudaMemcpy(
            d_bf16,
            src.as_ptr() as *const c_void,
            bytes_bf16,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }

        // GPU upcast → writes into d_f32.
        let rc = bf16_to_f32_launch_device(d_bf16, d_f32 as *mut f32, dst_len as c_int);
        if rc != 0 {
            return None;
        }

        // Download F32 D→H into a freshly-allocated host Vec.
        let mut out = vec![0.0_f32; dst_len];
        let rc = cudaMemcpy(
            out.as_mut_ptr() as *mut c_void,
            d_f32,
            bytes_f32,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );
        if rc != 0 {
            return None;
        }

        // `allocs` drops here when `Some(...)` returns, freeing both
        // device buffers before the host Vec is handed back.
        Some(out)
    }
}

/// **M6 step 4b** — upload BF16 to VRAM, run the GPU upcast
/// kernel, and **leave the resulting F32 buffer resident on the
/// device**. The transient BF16 device buffer is freed before
/// return; the F32 buffer is wrapped in a [`TensorGPU`] whose
/// `Drop` impl will eventually `cuda_free` it through the engine.
///
/// This is the residency-true variant of [`bf16_to_f32_on_device`]:
/// callers building a persistent VRAM weight store
/// (`WeightStore::upload_layer_bf16_to_vram`, M6 step 4b) need
/// the F32 buffer to outlive the call and serve subsequent
/// matmuls without re-uploading. The non-residency wrapper
/// downloads the F32 result back to host memory.
///
/// # Returns
///
/// `Some(TensorGPU)` whose underlying device buffer holds the F32
/// upcast of `src`. Logical shape is given by the caller and
/// preserved alongside the `TensorGPU` at the `SharedParam` layer
/// (the `TensorGPU` itself stores `(rows = numel, cols = 1)` as
/// the engine convention; see `Tensor::zeros_new_cuda`).
///
/// `None` if `cuda_available()` is false, the BF16 transient
/// `cuda_malloc` fails, the F32 `TensorGPU::empty` fails, the
/// H→D `cudaMemcpy` returns non-zero, or the kernel launcher
/// returns non-zero. On any error path:
/// - The BF16 transient is freed by the [`DeviceAllocs`] guard.
/// - The F32 `TensorGPU` (if allocated) is dropped, which
///   `cuda_free`s its buffer via `InnerGpuPtr::drop`.
/// - No VRAM leak.
///
/// # Bit-exactness
///
/// The CUDA `__bfloat162float` intrinsic and the host AVX2
/// formula `f32::from_bits((bf16_bits as u32) << 16)` produce
/// identical F32 patterns for every BF16 input. Validated end
/// to end on a 70.7M-element FFN-down weight in
/// `examples/test_bf16_upload.rs` (0 mismatches over 70M
/// elements).
pub fn bf16_to_f32_resident_in_vram(
    src: &[u16],
    shape: &[usize],
) -> Option<TensorGPU> {
    if !super::cuda_available() {
        return None;
    }

    let numel: usize = shape.iter().product();
    if src.len() != numel {
        // Caller-provided shape disagrees with the buffer size;
        // refusing to upload is safer than truncating or
        // over-reading. None lets the caller fall back to CPU.
        return None;
    }

    let bytes_bf16 = numel * std::mem::size_of::<u16>();

    // Allocate the BF16 transient via the same raw `cuda_malloc`
    // path used by `cuda::matmul::cuda_matmul_non_pooled`. Will
    // be freed by `DeviceAllocs::drop` on every return path.
    let mut transient = DeviceAllocs::new();
    let d_bf16 = transient.alloc(bytes_bf16)?;

    // Allocate the F32 destination via the engine-managed
    // `TensorGPU::empty` so the buffer's RAII lifecycle is
    // consistent with the rest of the engine. If the call
    // beyond this point fails, dropping `gpu_f32` triggers
    // `InnerGpuPtr::drop` which frees the buffer.
    let gpu_f32 = TensorGPU::empty(numel, 1).ok()?;

    unsafe {
        let rc = cudaMemcpy(
            d_bf16,
            src.as_ptr() as *const c_void,
            bytes_bf16,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }

        let rc = bf16_to_f32_launch_device(
            d_bf16,
            gpu_f32.device_ptr() as *mut f32,
            numel as c_int,
        );
        if rc != 0 {
            return None;
        }
    }

    // `transient` (BF16) drops here on success, freeing the
    // staging buffer. `gpu_f32` is moved into the return value
    // and survives.
    Some(gpu_f32)
}

#[cfg(test)]
mod tests {
    use super::bf16_to_f32_on_device;
    use crate::cuda::cuda_available;

    /// Bit-exactness against the host AVX2 decode formula
    /// `f32::from_bits((bits as u32) << 16)`. Skips on hosts
    /// without a CUDA driver — same convention as
    /// `tests/cuda_matmul_residency_test.rs`.
    #[test]
    fn bf16_to_f32_on_device_matches_host_decode() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        // Deterministic small buffer with values that exercise sign,
        // exponent, and mantissa bits. 64 elements is enough to
        // cover one CUDA block of 256 threads in the kernel
        // (the launcher uses block=256, so 64 fits in one block
        // with most threads idle — exactly the smallest-shape
        // case we care about).
        let host_bf16: Vec<u16> = (0..64)
            .map(|i| {
                // Build a varied F32 pattern, round-half-to-nearest
                // to BF16 by keeping the high 16 bits.
                let f = ((i as f32) * 0.3 - 4.0).sin();
                (f.to_bits() >> 16) as u16
            })
            .collect();

        let host_ref: Vec<f32> = host_bf16
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();

        let gpu = bf16_to_f32_on_device(&host_bf16, host_bf16.len())
            .expect("bf16_to_f32_on_device returned None on a known-good small buffer");

        assert_eq!(gpu.len(), host_ref.len());

        let mut bitwise_mismatches = 0_usize;
        for (g, h) in gpu.iter().zip(host_ref.iter()) {
            if g.to_bits() != h.to_bits() {
                bitwise_mismatches += 1;
            }
        }
        assert_eq!(
            bitwise_mismatches, 0,
            "GPU upcast produced {} bit-level mismatches vs host decode \
             on {} elements; the design assumption \
             `__bfloat162float() == AVX2 decode` is violated",
            bitwise_mismatches,
            host_bf16.len()
        );
    }
}
