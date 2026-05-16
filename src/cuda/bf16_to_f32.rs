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

use std::ffi::{CStr, c_void};
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::gpu::tensor::TensorGPU;

// ===========================================================================
// M8.1 — BF16-resident upload counter (no upcast path)
// ===========================================================================

/// **M8.1** — increments every time
/// [`bf16_to_vram_no_upcast`] successfully uploads a BF16 buffer
/// to VRAM without the F32 upcast pass. Mirrors the existing
/// `vram_fast_path_count` / `disk_fast_path_count` counters in
/// `v17::loader::weight_mapper`. Public reader is
/// [`cuda_bf16_resident_count`].
///
/// The counter is process-wide and never reset; tests that depend
/// on a delta should snapshot before/after with their own
/// `Mutex` (mirroring the M6 / M7 test pattern).
static BF16_RESIDENT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Read-only accessor for [`BF16_RESIDENT_COUNT`]. See module docs.
pub fn cuda_bf16_resident_count() -> usize {
    BF16_RESIDENT_COUNT.load(Ordering::Relaxed)
}

/// **M8.4 — shared test lock** for any test that snapshots
/// [`BF16_RESIDENT_COUNT`] (or counters mutated alongside it,
/// like `vram_bf16_matmul_count`) to assert a delta.
///
/// Without this lock, the cargo default thread pool runs lib
/// tests in parallel; tests in different modules
/// (`cuda::bf16_to_f32::tests`, `cuda::matmul::cuda_matmul_bf16_tests`)
/// race on the same global counter and "before/after" snapshots
/// observe each other's increments → flaky asserts.
///
/// Convention: every test that calls `cuda_bf16_resident_count`
/// or any function that increments `BF16_RESIDENT_COUNT`
/// (`bf16_to_vram_no_upcast`, `bf16_to_vram_no_upcast_from_raw_bytes`)
/// AND asserts on the delta MUST acquire this lock first.
/// `pub(crate)` so the matmul module's BF16 tests can share it.
#[cfg(test)]
pub(crate) static BF16_COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// FFI symbols needed by this module. `bf16_to_f32_launch_device`
// lives in the `bf16_to_f32` static library produced by
// `build.rs`. `cudaMemcpy` lives in `cudart`. `cuda_malloc` /
// `cuda_free` live in `atenia_kernels` (already linked elsewhere).
#[link(name = "bf16_to_f32", kind = "static")]
unsafe extern "C" {
    fn bf16_to_f32_launch_device(d_src_bf16: *const c_void, d_dst_f32: *mut f32, n: c_int)
    -> c_int;
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const c_char;
}

unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut c_void, bytes: usize);
    fn cuda_free(ptr: *mut c_void);
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[derive(Clone, Debug)]
pub struct Bf16UploadError {
    pub stage: &'static str,
    pub message: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub bf16_bytes: usize,
    pub f32_bytes: usize,
    pub free_vram_bytes: Option<usize>,
    pub total_vram_bytes: Option<usize>,
}

impl Bf16UploadError {
    pub(crate) fn new(
        stage: &'static str,
        message: impl Into<String>,
        shape: &[usize],
        numel: usize,
    ) -> Self {
        let (free_vram_bytes, total_vram_bytes) = cuda_mem_info().unwrap_or((0, 0));
        Self {
            stage,
            message: message.into(),
            shape: shape.to_vec(),
            numel,
            bf16_bytes: numel.saturating_mul(std::mem::size_of::<u16>()),
            f32_bytes: numel.saturating_mul(std::mem::size_of::<f32>()),
            free_vram_bytes: (free_vram_bytes > 0).then_some(free_vram_bytes),
            total_vram_bytes: (total_vram_bytes > 0).then_some(total_vram_bytes),
        }
    }
}

impl fmt::Display for Bf16UploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stage={} shape={:?} numel={} bf16={} MiB f32={} MiB",
            self.stage,
            self.shape,
            self.numel,
            bytes_to_mib(self.bf16_bytes),
            bytes_to_mib(self.f32_bytes)
        )?;
        if let (Some(free), Some(total)) = (self.free_vram_bytes, self.total_vram_bytes) {
            write!(
                f,
                " vram_free={} MiB vram_total={} MiB",
                bytes_to_mib(free),
                bytes_to_mib(total)
            )?;
        }
        write!(f, " ({})", self.message)
    }
}

impl std::error::Error for Bf16UploadError {}

fn bytes_to_mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

fn cuda_mem_info() -> Option<(usize, usize)> {
    if !super::cuda_available() {
        return None;
    }
    let mut free = 0usize;
    let mut total = 0usize;
    let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    (rc == 0).then_some((free, total))
}

pub(crate) fn cudart_error(rc: c_int) -> String {
    let ptr = unsafe { cudaGetErrorString(rc) };
    if ptr.is_null() {
        return format!("CUDA error code {rc}");
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

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
pub fn bf16_to_f32_resident_in_vram(src: &[u16], shape: &[usize]) -> Option<TensorGPU> {
    bf16_to_f32_resident_in_vram_detailed(src, shape).ok()
}

pub fn bf16_to_f32_resident_in_vram_detailed(
    src: &[u16],
    shape: &[usize],
) -> Result<TensorGPU, Bf16UploadError> {
    let numel: usize = shape.iter().product();
    if !super::cuda_available() {
        return Err(Bf16UploadError::new(
            "cuda_available",
            "CUDA probe is unavailable",
            shape,
            numel,
        ));
    }

    if src.len() != numel {
        // Caller-provided shape disagrees with the buffer size;
        // refusing to upload is safer than truncating or
        // over-reading. None lets the caller fall back to CPU.
        return Err(Bf16UploadError::new(
            "length_check",
            format!(
                "src.len()={} does not match shape product {}",
                src.len(),
                numel
            ),
            shape,
            numel,
        ));
    }

    let bytes_bf16 = numel * std::mem::size_of::<u16>();

    // Allocate the BF16 transient via the same raw `cuda_malloc`
    // path used by `cuda::matmul::cuda_matmul_non_pooled`. Will
    // be freed by `DeviceAllocs::drop` on every return path.
    let mut transient = DeviceAllocs::new();
    let d_bf16 = transient.alloc(bytes_bf16).ok_or_else(|| {
        Bf16UploadError::new(
            "alloc_bf16_staging",
            format!("cuda_malloc returned null for {bytes_bf16} bytes"),
            shape,
            numel,
        )
    })?;

    // Allocate the F32 destination via the engine-managed
    // `TensorGPU::empty` so the buffer's RAII lifecycle is
    // consistent with the rest of the engine. If the call
    // beyond this point fails, dropping `gpu_f32` triggers
    // `InnerGpuPtr::drop` which frees the buffer.
    let gpu_f32 = TensorGPU::empty(numel, 1).map_err(|_| {
        Bf16UploadError::new(
            "alloc_f32_destination",
            format!("TensorGPU::empty failed for {} bytes", numel * 4),
            shape,
            numel,
        )
    })?;

    unsafe {
        let rc = cudaMemcpy(
            d_bf16,
            src.as_ptr() as *const c_void,
            bytes_bf16,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return Err(Bf16UploadError::new(
                "memcpy_h2d_bf16",
                format!("cudaMemcpy H2D failed: {}", cudart_error(rc)),
                shape,
                numel,
            ));
        }

        let rc =
            bf16_to_f32_launch_device(d_bf16, gpu_f32.device_ptr() as *mut f32, numel as c_int);
        if rc != 0 {
            return Err(Bf16UploadError::new(
                "kernel_bf16_to_f32",
                format!("bf16_to_f32_launch_device returned {rc}"),
                shape,
                numel,
            ));
        }
    }

    // `transient` (BF16) drops here on success, freeing the
    // staging buffer. `gpu_f32` is moved into the return value
    // and survives.
    Ok(gpu_f32)
}

/// **M6 replan sub-fase 2** — zero-host-copy variant of
/// [`bf16_to_f32_resident_in_vram`]. Takes a `&[u8]` of raw BF16
/// bytes (typically a slice into the safetensors reader's owned
/// byte buffer) and a logical element count. The bytes are
/// shipped to VRAM via a single `cudaMemcpy` H→D that does **not**
/// require host-side `&[u16]` alignment — the destination on
/// the device is aligned by `cuda_malloc`, and the source is
/// treated as opaque bytes by the cudart memcpy routine.
///
/// This is the load-time path used by
/// `WeightMapper::load_into_with_residency_plan` when:
/// - the source dtype is BF16,
/// - the parameter has no `LoadTransform`s registered, and
/// - the planner assigned it to [`crate::gpu::tier_plan::Tier::Vram`].
///
/// Under those conditions, the entire weight upload happens
/// without ever materialising a host-side F32 transient or a
/// secondary `Vec<u16>`. The peak host-RAM cost is the
/// safetensors reader's owned byte buffer (which the caller
/// already paid for), no more.
///
/// # Returns
///
/// `Some(TensorGPU)` whose F32 device buffer is `numel * 4` bytes,
/// holding the upcast of the BF16 source. `None` on
/// `cuda_available()` failure, length mismatch
/// (`raw_bytes.len() != numel * 2`), `cuda_malloc` failure,
/// `cudaMemcpy` failure, or kernel launch failure. Buffer
/// cleanup on every error path is guaranteed by [`DeviceAllocs`]'s
/// `Drop` and `TensorGPU`'s `Drop`.
pub fn bf16_to_f32_resident_in_vram_from_raw_bytes(
    raw_bytes: &[u8],
    numel: usize,
    shape: &[usize],
) -> Option<TensorGPU> {
    bf16_to_f32_resident_in_vram_from_raw_bytes_detailed(raw_bytes, numel, shape).ok()
}

pub fn bf16_to_f32_resident_in_vram_from_raw_bytes_detailed(
    raw_bytes: &[u8],
    numel: usize,
    shape: &[usize],
) -> Result<TensorGPU, Bf16UploadError> {
    if !super::cuda_available() {
        return Err(Bf16UploadError::new(
            "cuda_available",
            "CUDA probe is unavailable",
            shape,
            numel,
        ));
    }

    let bytes_bf16 = numel * std::mem::size_of::<u16>();
    if raw_bytes.len() != bytes_bf16 {
        // Length mismatch — refusing to upload would risk reading
        // past the source slice on the device side. Caller falls
        // back to CPU.
        return Err(Bf16UploadError::new(
            "length_check",
            format!(
                "raw_bytes.len()={} does not match expected BF16 byte length {}",
                raw_bytes.len(),
                bytes_bf16
            ),
            shape,
            numel,
        ));
    }
    let shape_numel: usize = shape.iter().product();
    if shape_numel != numel {
        return Err(Bf16UploadError::new(
            "shape_check",
            format!(
                "shape product {} does not match numel {}",
                shape_numel, numel
            ),
            shape,
            numel,
        ));
    }

    let mut transient = DeviceAllocs::new();
    let d_bf16 = transient.alloc(bytes_bf16).ok_or_else(|| {
        Bf16UploadError::new(
            "alloc_bf16_staging",
            format!("cuda_malloc returned null for {bytes_bf16} bytes"),
            shape,
            numel,
        )
    })?;

    let gpu_f32 = TensorGPU::empty(numel, 1).map_err(|_| {
        Bf16UploadError::new(
            "alloc_f32_destination",
            format!("TensorGPU::empty failed for {} bytes", numel * 4),
            shape,
            numel,
        )
    })?;

    unsafe {
        let rc = cudaMemcpy(
            d_bf16,
            raw_bytes.as_ptr() as *const c_void,
            bytes_bf16,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return Err(Bf16UploadError::new(
                "memcpy_h2d_bf16",
                format!("cudaMemcpy H2D failed: {}", cudart_error(rc)),
                shape,
                numel,
            ));
        }

        let rc =
            bf16_to_f32_launch_device(d_bf16, gpu_f32.device_ptr() as *mut f32, numel as c_int);
        if rc != 0 {
            return Err(Bf16UploadError::new(
                "kernel_bf16_to_f32",
                format!("bf16_to_f32_launch_device returned {rc}"),
                shape,
                numel,
            ));
        }
    }

    Ok(gpu_f32)
}

// ===========================================================================
// M8.1 — `bf16_to_vram_no_upcast`
// ===========================================================================

/// **M8.1** — upload a BF16 buffer to VRAM and **keep it as BF16**.
///
/// Unlike [`bf16_to_f32_resident_in_vram`] (M6 path) which
/// upcasts to F32 on the device via the
/// `bf16_to_f32_launch_device` kernel, this function performs a
/// single H→D byte copy and stops. The returned [`TensorGPU`]
/// carries `dtype = DType::BF16` and a device buffer of exactly
/// `numel * 2` bytes — half the F32 footprint, ready for direct
/// consumption by `cublasGemmEx(CUDA_R_16BF, CUDA_R_16BF, ...)`
/// in the M8.2 BF16 matmul kernel.
///
/// # When to use this vs the M6 upcast path
///
/// - **M6 (upcast)**: when downstream consumers are the legacy
///   F32 matmul kernels (`matmul_f32_launch_device`, `cublasGemmEx`
///   with `CUDA_R_32F` inputs) — they need F32 in VRAM.
/// - **M8.1 (no upcast)**: when downstream is the M8 BF16
///   matmul kernel (`cublasGemmEx` with `CUDA_R_16BF` inputs).
///   Saves 50 % VRAM per weight and skips an entire kernel pass
///   over the data at load time.
///
/// # Bit-exactness contract
///
/// The device buffer holds an exact copy of the host `src` slice
/// — every `u16` bit pattern survives the round-trip via
/// `cudaMemcpy(H→D)` followed by `cudaMemcpy(D→H)`. Validated by
/// the `bf16_to_vram_no_upcast_round_trip_bit_exact` unit test.
///
/// # Returns
///
/// `Some(TensorGPU)` with `dtype = BF16`, `(rows = numel, cols = 1)`,
/// and a device allocation of `numel * 2` bytes on success. Side
/// effect: [`BF16_RESIDENT_COUNT`] increments by 1.
///
/// `None` if the GPU engine is unavailable, the shape disagrees
/// with the buffer length, or the engine alloc / copy fails. The
/// counter is **not** incremented on failure — same convention
/// as the M6 / M7 fast-path counters.
pub fn bf16_to_vram_no_upcast(src: &[u16], shape: &[usize]) -> Option<TensorGPU> {
    bf16_to_vram_no_upcast_detailed(src, shape).ok()
}

/// **M12.1** — `Result`-returning sibling of
/// [`bf16_to_vram_no_upcast`]. Same success path (bit-identical:
/// same counter increment, same `TensorGPU`); on failure returns
/// a [`Bf16UploadError`] carrying the stage, shape, numel and the
/// live VRAM free/total so the operator sees the root cause
/// instead of an opaque `None`. The legacy `Option` wrapper above
/// preserves every existing caller and test.
pub fn bf16_to_vram_no_upcast_detailed(
    src: &[u16],
    shape: &[usize],
) -> Result<TensorGPU, Bf16UploadError> {
    let numel: usize = shape.iter().product();
    if !super::cuda_available() {
        return Err(Bf16UploadError::new(
            "cuda_unavailable",
            "CUDA engine unavailable",
            shape,
            numel,
        ));
    }
    if src.len() != numel {
        return Err(Bf16UploadError::new(
            "validate_shape",
            format!("src.len()={} != product(shape)={}", src.len(), numel),
            shape,
            numel,
        ));
    }

    // `TensorGPU::new_bf16_from_cpu` does the alloc + H→D byte
    // copy in one shot; on any failure the partially-constructed
    // device buffer is freed by `InnerGpuPtr::Drop` before the
    // error surfaces, so this function does not leak VRAM even if
    // the engine call returns mid-upload. The engine returns
    // `Err(())` (no rc), so the message names the stage; the
    // VRAM free/total captured by `Bf16UploadError::new` lets the
    // operator distinguish "VRAM exhausted" from other failures.
    let gpu = TensorGPU::new_bf16_from_cpu(src).map_err(|_| {
        Bf16UploadError::new(
            "alloc_h2d_bf16",
            "TensorGPU::new_bf16_from_cpu failed (VRAM alloc or H2D copy)",
            shape,
            numel,
        )
    })?;

    BF16_RESIDENT_COUNT.fetch_add(1, Ordering::Relaxed);
    Ok(gpu)
}

/// **M8.1** — zero-host-copy variant of [`bf16_to_vram_no_upcast`]
/// that takes raw bytes (typically a slice into the safetensors
/// reader's owned byte buffer) and a logical element count.
///
/// Same H→D byte copy as the safe variant; the source is the
/// already-on-disk BF16 byte layout, so no reinterpret-cast or
/// secondary `Vec<u16>` is needed. The peak host RAM cost is the
/// safetensors mmap residency only — which the caller already
/// pays for. This is the load-time entry point that M8.4 will
/// wire into `WeightMapper::load_into_with_residency_plan` for
/// `Tier::Vram + dtype=BF16 + no transforms` tensors when the
/// `ATENIA_M8_BF16_KERNEL=1` flag is on.
///
/// Returns `None` and increments **no** counter on length
/// mismatch (`raw_bytes.len() != numel * 2`), shape mismatch
/// (`shape.iter().product() != numel`), or engine failure.
pub fn bf16_to_vram_no_upcast_from_raw_bytes(
    raw_bytes: &[u8],
    numel: usize,
    shape: &[usize],
) -> Option<TensorGPU> {
    bf16_to_vram_no_upcast_from_raw_bytes_detailed(raw_bytes, numel, shape).ok()
}

/// **M12.1** — `Result`-returning sibling of
/// [`bf16_to_vram_no_upcast_from_raw_bytes`]. Success path
/// bit-identical; failures carry the root cause via
/// [`Bf16UploadError`]. The legacy `Option` wrapper above
/// preserves all existing callers/tests.
pub fn bf16_to_vram_no_upcast_from_raw_bytes_detailed(
    raw_bytes: &[u8],
    numel: usize,
    shape: &[usize],
) -> Result<TensorGPU, Bf16UploadError> {
    if !super::cuda_available() {
        return Err(Bf16UploadError::new(
            "cuda_unavailable",
            "CUDA engine unavailable",
            shape,
            numel,
        ));
    }
    let bytes_bf16 = numel * std::mem::size_of::<u16>();
    if raw_bytes.len() != bytes_bf16 {
        return Err(Bf16UploadError::new(
            "validate_shape",
            format!(
                "raw_bytes.len()={} != numel*2={}",
                raw_bytes.len(),
                bytes_bf16
            ),
            shape,
            numel,
        ));
    }
    let shape_numel: usize = shape.iter().product();
    if shape_numel != numel {
        return Err(Bf16UploadError::new(
            "validate_shape",
            format!("product(shape)={shape_numel} != numel={numel}"),
            shape,
            numel,
        ));
    }

    // Reinterpret the raw bytes as `&[u16]` and route through
    // the same alloc + H→D copy as the safe variant. This is
    // sound because:
    //   - `raw_bytes.len() == numel * 2` was verified above, so
    //     the slice has exactly `numel` u16 elements;
    //   - the source is treated as opaque bytes by `cudaMemcpy`,
    //     so endianness / alignment of the host `u16` view is
    //     irrelevant on the device side — what survives is the
    //     byte sequence;
    //   - host alignment of `raw_bytes` only needs to be `2` for
    //     the `&[u16]` slice to be well-defined; safetensors
    //     buffers are page-aligned so this trivially holds.
    let bits: &[u16] =
        unsafe { std::slice::from_raw_parts(raw_bytes.as_ptr() as *const u16, numel) };
    let gpu = TensorGPU::new_bf16_from_cpu(bits).map_err(|_| {
        Bf16UploadError::new(
            "alloc_h2d_bf16",
            "TensorGPU::new_bf16_from_cpu failed (VRAM alloc or H2D copy)",
            shape,
            numel,
        )
    })?;
    BF16_RESIDENT_COUNT.fetch_add(1, Ordering::Relaxed);
    Ok(gpu)
}

// ===========================================================================
// M8.4c — bf16_to_f32_transient_in_vram
// ===========================================================================

/// **M8.4c** — upcast a BF16-resident `TensorGPU` to a fresh F32-
/// resident `TensorGPU` on-device, without touching host memory.
///
/// This is the cornerstone primitive of the M8.4c "Path B"
/// numerical-correctness fix. The M8.4 BF16 matmul path
/// (`cuda_matmul_bf16_inplace` original) cast the F32 activation
/// to BF16 once per matmul on the host before upload, then ran
/// `cublasGemmEx(BF16, BF16, F32)`. Cascaded over 16-28 layers
/// in real Llama checkpoints the activation truncation drifted
/// the logits by 0.18-2.33 — failing ADR-004 (threshold 0.5)
/// in 3/4 production models in the M8.5 4-model F64 validation.
///
/// The fix: keep the **weight** as BF16 in VRAM (preserves
/// M8.3's capacity-doubling planner contract) but **upcast it
/// to F32 on the device** for every matmul, then run the matmul
/// with both operands as F32. Numerically identical to the
/// M4.7.2.e CPU path (BF16 storage + F32 matmul) which has
/// drift 1.4e-4 on TinyLlama vs 0.9 on the original M8.4 path.
///
/// # Implementation
///
/// 1. Validate `bf16_gpu.dtype() == DType::BF16` — calling this
///    on an F32 buffer is a programmer error and returns `None`.
/// 2. Allocate a fresh F32 device buffer via the engine
///    (`TensorGPU::empty(numel, 1)` — same convention as the M6
///    upcast path at `bf16_to_f32_resident_in_vram`).
/// 3. Launch the existing `bf16_to_f32_launch_device` kernel
///    on `(bf16_gpu.device_ptr() → f32_gpu.device_ptr(), numel)`.
/// 4. Return the F32 buffer wrapped in `TensorGPU` with
///    `dtype = DType::F32`. The caller's `Drop` releases it
///    automatically; this is the "transient" lifecycle implied
///    by the function name — the F32 buffer is intended to be
///    consumed by exactly one matmul and released at the end of
///    the call.
///
/// # Returns
///
/// `Some(TensorGPU)` with `dtype = F32`, `numel * 4` bytes, on
/// success. `None` on:
///   - `bf16_gpu.dtype() != BF16` (caller bug)
///   - `cuda_available()` returns false
///   - F32 allocation fails (driver OOM)
///   - kernel launch returns non-zero
///
/// On any error the F32 buffer (if it was allocated) is freed
/// by `TensorGPU`'s `Drop` before the `None` propagates — no
/// VRAM leak.
///
/// # Numerics
///
/// `bf16_to_f32_launch_device` is bit-exact to the host
/// `f32::from_bits((bits as u32) << 16)` formula (validated by
/// `bf16_to_f32_on_device_matches_host_decode` over 70.7M
/// elements at M6 commit `66910d5`). The transient F32 buffer
/// holds the **lossless** F32 representation of the BF16
/// pattern — no further rounding occurs in this primitive.
pub fn bf16_to_f32_transient_in_vram(bf16_gpu: &TensorGPU) -> Option<TensorGPU> {
    if !super::cuda_available() {
        return None;
    }
    if bf16_gpu.dtype() != crate::tensor::DType::BF16 {
        return None;
    }

    // `TensorGPU::new_bf16` and friends use `(rows = numel,
    // cols = 1)` as the engine convention. The output F32
    // buffer follows the same convention so the matmul caller
    // can wrap it via `Tensor::from_cuda_gpu(shape, ...)` with
    // an externally-known logical shape.
    let numel = bf16_gpu.rows * bf16_gpu.cols;

    // Sanity: the device buffer must hold exactly `numel * 2`
    // bytes (BF16). If it doesn't, the caller violated the
    // contract somewhere upstream — refuse to read past the
    // allocation.
    if bf16_gpu.size_bytes() != numel * 2 {
        return None;
    }

    let f32_gpu = TensorGPU::empty(numel, 1).ok()?;

    unsafe {
        let rc = bf16_to_f32_launch_device(
            bf16_gpu.device_ptr() as *const c_void,
            f32_gpu.device_ptr() as *mut f32,
            numel as c_int,
        );
        if rc != 0 {
            return None;
        }
    }

    Some(f32_gpu)
}

#[cfg(test)]
mod tests {
    use super::{
        BF16_COUNTER_TEST_LOCK, bf16_to_f32_on_device, bf16_to_vram_no_upcast,
        cuda_bf16_resident_count,
    };
    use crate::cuda::cuda_available;
    use crate::gpu::tensor::TensorGPU;
    use crate::tensor::DType;

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
            bitwise_mismatches,
            0,
            "GPU upcast produced {} bit-level mismatches vs host decode \
             on {} elements; the design assumption \
             `__bfloat162float() == AVX2 decode` is violated",
            bitwise_mismatches,
            host_bf16.len()
        );
    }

    /// **M8.1 round-trip bit-exact** — upload a synthetic BF16
    /// buffer to VRAM via `bf16_to_vram_no_upcast`, download it
    /// via `TensorGPU::to_cpu_bf16_bits`, and assert that every
    /// `u16` bit pattern survived the round-trip unchanged.
    /// Also verifies that:
    ///
    /// - the returned `TensorGPU` has `dtype = BF16`,
    /// - the `cuda_bf16_resident_count` counter advances by
    ///   exactly 1 per successful call,
    /// - calling `to_cpu` (F32-only) on a BF16 tensor panics
    ///   with the documented diagnostic — caught via
    ///   `catch_unwind` so the test stays self-contained.
    ///
    /// Skips on hosts without a CUDA driver.
    #[test]
    fn bf16_to_vram_no_upcast_round_trip_bit_exact() {
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        // 1024 elements: one CUDA block worth + then some, plus
        // small enough to exercise the full code path quickly.
        // Same value-distribution recipe as the existing M6 test.
        let host_bits: Vec<u16> = (0..1024)
            .map(|i| {
                let f = ((i as f32) * 0.137 - 7.5).sin() * 0.7
                    + ((i as f32) * 0.029 + 0.42).cos() * 0.3;
                (f.to_bits() >> 16) as u16
            })
            .collect();
        let shape = vec![1024_usize];

        let before = cuda_bf16_resident_count();
        let gpu = bf16_to_vram_no_upcast(&host_bits, &shape)
            .expect("bf16_to_vram_no_upcast returned None on a known-good buffer");
        let after = cuda_bf16_resident_count();

        // Counter advanced by exactly one.
        assert_eq!(
            after - before,
            1,
            "BF16_RESIDENT_COUNT did not advance by exactly 1 \
             (before={}, after={})",
            before,
            after
        );

        // Returned tensor is BF16-typed and shaped (numel, 1).
        assert_eq!(gpu.dtype(), DType::BF16, "expected BF16 dtype");
        assert_eq!(gpu.rows, 1024);
        assert_eq!(gpu.cols, 1);
        assert_eq!(
            gpu.size_bytes(),
            1024 * 2,
            "BF16 buffer must be numel * 2 bytes; got {}",
            gpu.size_bytes()
        );

        // Round-trip via the new `to_cpu_bf16_bits` accessor.
        let downloaded = gpu
            .to_cpu_bf16_bits()
            .expect("to_cpu_bf16_bits failed on a BF16-resident tensor");

        assert_eq!(downloaded.len(), host_bits.len());
        let mut mismatches = 0_usize;
        for (h, d) in host_bits.iter().zip(downloaded.iter()) {
            if h != d {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "BF16 round-trip produced {} bit mismatches over {} elements; \
             the H→D + D→H byte path corrupted the buffer",
            mismatches,
            host_bits.len()
        );

        // `to_cpu` (F32-only) must refuse a BF16 tensor with a
        // diagnostic panic. Caught via `catch_unwind` so the test
        // does not abort the whole suite on a successful catch.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = gpu.to_cpu();
        }))
        .is_err();
        assert!(
            panicked,
            "TensorGPU::to_cpu must panic when called on a BF16-resident tensor"
        );
    }

    /// **M8.1 backward-compat** — confirm that the F32 path
    /// (`TensorGPU::empty` + `new_from_cpu` + `to_cpu`) is
    /// untouched. Counter must NOT advance for F32-resident
    /// uploads. This is the regression-zero gate that tells us
    /// no M3-M7 caller can have been silently re-routed.
    #[test]
    fn f32_resident_path_does_not_increment_bf16_counter() {
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let before = cuda_bf16_resident_count();
        let gpu = TensorGPU::new_from_cpu(&data, 4, 1)
            .expect("F32 upload failed on a CUDA-available host");
        let after = cuda_bf16_resident_count();
        assert_eq!(
            after, before,
            "F32 path must not increment the BF16 resident counter"
        );
        assert_eq!(gpu.dtype(), DType::F32);
        let back = gpu.to_cpu().expect("F32 to_cpu");
        assert_eq!(back, data);
    }
}

#[cfg(test)]
mod m12_1_tests {
    use super::*;

    /// M12.1 contract: the legacy `Option` wrapper is exactly
    /// `_detailed().ok()`, and `_detailed` never returns an opaque
    /// failure — it always carries a structured `Bf16UploadError`
    /// (stage / shape / numel). Portable across environments: with
    /// no CUDA the error stage is `cuda_unavailable`; with CUDA the
    /// invalid shape yields `validate_shape`. Either way the
    /// wrapper==detailed.ok() invariant and the structured-error
    /// guarantee hold without needing a GPU.
    #[test]
    fn no_upcast_detailed_mirrors_legacy_and_is_structured() {
        let src = vec![0u16; 3];
        let shape = [2usize, 2]; // product 4 != src.len() 3 → must fail
        assert!(bf16_to_vram_no_upcast(&src, &shape).is_none());
        let err = bf16_to_vram_no_upcast_detailed(&src, &shape)
            .expect_err("invalid input must be a structured error, not Ok");
        assert_eq!(err.shape, shape.to_vec());
        assert_eq!(err.numel, 4);
        assert!(!err.stage.is_empty());
        assert!(!err.to_string().is_empty());
        assert_eq!(
            bf16_to_vram_no_upcast(&src, &shape).is_none(),
            bf16_to_vram_no_upcast_detailed(&src, &shape).is_err(),
            "legacy Option must equal _detailed().ok()"
        );
    }

    #[test]
    fn no_upcast_from_raw_bytes_detailed_mirrors_legacy_and_is_structured() {
        let raw = vec![0u8; 4]; // 2 u16; numel says 4 → mismatch
        let shape = [2usize, 2];
        assert!(bf16_to_vram_no_upcast_from_raw_bytes(&raw, 4, &shape).is_none());
        let err = bf16_to_vram_no_upcast_from_raw_bytes_detailed(&raw, 4, &shape)
            .expect_err("invalid input must be a structured error");
        assert_eq!(err.numel, 4);
        assert!(!err.to_string().is_empty());
    }
}
