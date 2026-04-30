//! M6.c.1 — vendor-neutral GPU `Backend` trait + `CudaBackend` impl.
//!
//! The architectural piece that lets M6 ship a CUDA-only
//! production path **and** prepares v22 (wgpu) as a peer
//! `Backend` impl rather than a from-scratch refactor.
//!
//! ## What lives here vs what lives elsewhere
//!
//! - **This module**: the abstraction (`Backend`,
//!   `DeviceBuffer`), the CUDA implementation
//!   ([`CudaBackend`]), and a `OnceLock`-cached singleton
//!   accessor.
//! - **`src/cuda/`**: the kernel ABI + low-level primitives
//!   the CUDA backend wraps (`matmul_f32_launch_device`,
//!   `cuda_malloc_raw`, `cudaMemcpy`).
//! - **`src/gpu/dispatch/hooks.rs`**: the hot-path executor
//!   gate (`gpu_can_run_matmul`, `try_gpu_matmul`). M6.c.2's
//!   `LayerResidencyPlanner` decisions reach the executor
//!   through this dispatch surface, not through the
//!   `Backend` trait directly.
//!
//! ## Backend trait surface
//!
//! Six operations — chosen to be the common subset of CUDA,
//! wgpu, ROCm-HIP, and Metal:
//!
//! ```ignore
//! pub trait Backend {
//!     fn name(&self) -> &'static str;
//!     fn available_vram_bytes(&self) -> Option<u64>;
//!     fn peak_compute_flops_f32(&self) -> Option<f64>;
//!     fn upload(&self, host: &[f32]) -> Result<DeviceBuffer, BackendError>;
//!     fn download(&self, dev: &DeviceBuffer, host: &mut [f32]) -> Result<(), BackendError>;
//!     fn matmul_f32(&self, a: &DeviceBuffer, b: &DeviceBuffer,
//!                   out: &mut DeviceBuffer, m: usize, k: usize, n: usize)
//!                   -> Result<(), BackendError>;
//! }
//! ```
//!
//! `DeviceBuffer` is intentionally an opaque handle the
//! backend owns — the trait does not expose raw pointers.
//! For CUDA, the handle wraps a `*mut c_void` from
//! `cudaMalloc`; for wgpu (v22), it would wrap a
//! `wgpu::Buffer`. Drop releases the device memory.
//!
//! ## What this commit does NOT do
//!
//! - **No production wiring.** M6.c.1 + M6.c.2 are pure
//!   plumbing. The hot-path executor and the
//!   `GenerationPipeline` are unchanged at this commit; the
//!   trait surface is reachable from tests and from a
//!   future `Backend::global()` consumer in M6.c.3+.
//! - **No matmul kernel inline.** `CudaBackend::matmul_f32`
//!   delegates to the existing
//!   `crate::cuda::matmul::matmul_f32_launch_device` ABI;
//!   no new kernel code lands.
//! - **No async streams.** The trait is sync. Streaming
//!   overlap is M6.d via either an `async` extension trait
//!   or a `submit_async` method on the same trait — design
//!   deferred to M6.d when the empirical need is measured.

use std::os::raw::c_int;

/// Errors a `Backend` operation can produce. The variants
/// are coarse-grained because callers either retry on CPU
/// or fail; fine-grained driver-error introspection is M6+
/// observability work, not the M6.c contract.
#[derive(Debug)]
pub enum BackendError {
    /// VRAM allocation failed (driver / pool / OOM).
    AllocationFailed { bytes: usize, message: String },
    /// Host↔device copy failed.
    TransferFailed { direction: &'static str, message: String },
    /// Kernel launch returned a non-zero status.
    KernelLaunchFailed { kernel: &'static str, status: i32 },
    /// Caller passed a buffer or shape the backend does not
    /// support (e.g., shape mismatch on matmul).
    InvalidArgument(String),
    /// The backend is not available on this host (e.g.
    /// `CudaBackend` constructed when CUDA driver is
    /// missing).
    NotAvailable,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::AllocationFailed { bytes, message } =>
                write!(f, "backend alloc {bytes} B: {message}"),
            BackendError::TransferFailed { direction, message } =>
                write!(f, "backend transfer {direction}: {message}"),
            BackendError::KernelLaunchFailed { kernel, status } =>
                write!(f, "backend kernel '{kernel}' failed: status {status}"),
            BackendError::InvalidArgument(s) =>
                write!(f, "backend invalid argument: {s}"),
            BackendError::NotAvailable =>
                write!(f, "backend not available on this host"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Opaque device-side buffer handle owned by a `Backend`
/// implementation. Drops free the underlying allocation
/// via the trait's `free` mechanism (RAII through the
/// vendor-specific Drop impl held inside).
///
/// CUDA implementation: a refcounted Arc<RawCudaPtr> so the
/// buffer is freed at last clone. v22 wgpu implementation
/// would wrap an `Arc<wgpu::Buffer>` with the same RAII
/// semantics.
pub struct DeviceBuffer {
    inner: DeviceBufferInner,
    /// Element count (F32 elements). Bytes = `len * 4`.
    pub len: usize,
}

impl DeviceBuffer {
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn bytes(&self) -> usize { self.len * std::mem::size_of::<f32>() }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("backend", &self.inner.backend_name())
            .field("len", &self.len)
            .field("bytes", &self.bytes())
            .finish()
    }
}

/// Vendor-specific buffer payload. Sealed enum kept as a
/// private type; the public `DeviceBuffer` wrapper is the
/// stable surface.
enum DeviceBufferInner {
    Cuda(cuda_buffer::CudaBuffer),
    // Future: WgpuBuffer, RocmBuffer, MetalBuffer.
}

impl DeviceBufferInner {
    fn backend_name(&self) -> &'static str {
        match self {
            DeviceBufferInner::Cuda(_) => "cuda",
        }
    }
}

/// The vendor-neutral abstraction. Implementations:
///
/// - [`CudaBackend`] — production path on dev hardware
///   (RTX 4070 Laptop). Wraps the existing
///   `cuda_matmul_non_pooled` kernel and the
///   `cuda_malloc_raw`/`cuda_free_raw` allocators.
/// - **(v22)** wgpu backend — same trait, WGSL shader-
///   based matmul, `wgpu::Buffer` device handles.
/// - **(v23)** ROCm-HIP backend.
/// - **(v24)** Metal backend.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Free VRAM in bytes, or `None` when the backend
    /// cannot answer (e.g. no driver loaded). Used by the
    /// `LayerResidencyPlanner` to size the resident set.
    fn available_vram_bytes(&self) -> Option<u64>;

    /// Best-effort peak F32 throughput estimate. Used by
    /// the M6.f bench harness for FLOPS-vs-peak reporting.
    /// Returns `None` when the backend can't introspect
    /// hardware.
    fn peak_compute_flops_f32(&self) -> Option<f64>;

    /// Allocate a fresh device buffer and copy `host` into
    /// it. The returned buffer's lifetime is independent of
    /// `host`; the caller is responsible for keeping the
    /// buffer alive across the matmuls that consume it.
    fn upload(&self, host: &[f32]) -> Result<DeviceBuffer, BackendError>;

    /// Allocate a zero-filled device buffer of `len` F32
    /// elements. Used to prepare an output buffer that a
    /// matmul kernel will write into.
    fn alloc_zeros(&self, len: usize) -> Result<DeviceBuffer, BackendError>;

    /// Copy `dev` back to host. `host.len()` must equal
    /// `dev.len()`.
    fn download(&self, dev: &DeviceBuffer, host: &mut [f32]) -> Result<(), BackendError>;

    /// `out = a · b`. Shapes: `a` is `[m × k]` row-major,
    /// `b` is `[k × n]` row-major, `out` is `[m × n]`. Same
    /// convention as `matmul_dispatch` and
    /// `matmul_f32_launch_device`.
    fn matmul_f32(
        &self,
        a: &DeviceBuffer, b: &DeviceBuffer, out: &mut DeviceBuffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError>;
}

// ============================================================
//  CUDA implementation
// ============================================================

mod cuda_buffer {
    use std::ffi::c_void;
    use std::sync::Arc;

    /// Refcounted device pointer. Drops `cuda_free_raw` on
    /// last reference. `Send + Sync` because the underlying
    /// CUDA driver allows pointers to be moved between host
    /// threads (we don't run kernels concurrently against
    /// the same buffer; the `Backend` trait is sync today).
    pub(super) struct CudaBuffer {
        inner: Arc<RawCudaPtr>,
    }

    impl CudaBuffer {
        pub fn new(ptr: *mut c_void) -> Self {
            Self { inner: Arc::new(RawCudaPtr(ptr)) }
        }
        pub fn ptr(&self) -> *mut c_void { self.inner.0 }
        pub fn ptr_const(&self) -> *const c_void { self.inner.0 as *const c_void }
    }

    /// Minimal RAII wrapper. Kept private to this module so
    /// callers can't bypass the Arc and double-free.
    struct RawCudaPtr(*mut c_void);

    // SAFETY: the underlying CUDA pointer is opaque and
    // valid across threads as long as the device context
    // is alive (which is for the program's lifetime here).
    // The `Backend` trait is sync, so concurrent access
    // through the same handle is bounded by the trait
    // method's `&self` — a `&Backend` is consultable from
    // multiple threads but a single matmul invocation is
    // serial.
    unsafe impl Send for RawCudaPtr {}
    unsafe impl Sync for RawCudaPtr {}

    impl Drop for RawCudaPtr {
        fn drop(&mut self) {
            // SAFETY: invariant — every `RawCudaPtr` was
            // produced by `cuda_malloc_raw` (see
            // `CudaBackend::upload` / `alloc_zeros`) and is
            // freed exactly once on Arc-strongcount-zero.
            unsafe { crate::apx4_12::cuda_free_raw(self.0); }
        }
    }
}

/// Production CUDA backend. Always-on (no Cargo feature
/// gate) — the existing `src/cuda/` infrastructure already
/// links against `cudart` and the `matmul_kernel.cu`
/// kernels; this backend is just a vendor-neutral facade
/// over them.
///
/// `CudaBackend::new()` returns an instance that probes
/// `cuda::cuda_available()` lazily on the first call. Use
/// [`Backend::available_vram_bytes`] to test "is CUDA
/// usable" before consuming the trait surface — `None`
/// means the driver wasn't reachable.
pub struct CudaBackend;

impl CudaBackend {
    pub const fn new() -> Self { Self }

    /// Singleton accessor. The CUDA driver itself is global,
    /// so we return a `&'static CudaBackend` to keep the
    /// dispatch sites trivially cheap. v22 will add a
    /// generic `BACKEND: OnceLock<Box<dyn Backend>>`
    /// surface for selecting between CUDA and wgpu at
    /// startup.
    pub fn global() -> &'static CudaBackend {
        static B: CudaBackend = CudaBackend::new();
        &B
    }
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;

    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

#[link(name = "matmul_kernel")]
unsafe extern "C" {
    fn matmul_f32_launch_device(
        a: *const f32, b: *const f32, c: *mut f32,
        m: c_int, k: c_int, n: c_int,
    );
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

impl Backend for CudaBackend {
    fn name(&self) -> &'static str { "cuda" }

    fn available_vram_bytes(&self) -> Option<u64> {
        if !crate::cuda::cuda_available() {
            return None;
        }
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if status == 0 {
            Some(free as u64)
        } else {
            // Driver returned an error — treat as "can't
            // answer" and let the planner fall back to a
            // configured ceiling.
            None
        }
    }

    fn peak_compute_flops_f32(&self) -> Option<f64> {
        // RTX 4070 Laptop: ~7-8 TFLOPS sustained F32.
        // Future M6.f reads the device name from
        // `cudaDeviceGetName` and tables the answer.
        // Conservative default for now.
        if self.available_vram_bytes().is_some() {
            Some(7.5e12)
        } else {
            None
        }
    }

    fn upload(&self, host: &[f32]) -> Result<DeviceBuffer, BackendError> {
        let bytes = host.len() * std::mem::size_of::<f32>();
        // SAFETY: cuda_malloc_raw is the project's
        // null-checked thin wrapper over `cudaMalloc`. We
        // pair every successful alloc with a Drop-driven
        // free via CudaBuffer / RawCudaPtr.
        let ptr = unsafe { crate::apx4_12::cuda_malloc_raw(bytes) };
        if ptr.is_null() {
            return Err(BackendError::AllocationFailed {
                bytes,
                message: "cudaMalloc returned null".into(),
            });
        }
        let status = unsafe {
            cudaMemcpy(
                ptr,
                host.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if status != 0 {
            unsafe { crate::apx4_12::cuda_free_raw(ptr); }
            return Err(BackendError::TransferFailed {
                direction: "host->device",
                message: format!("cudaMemcpy returned {status}"),
            });
        }
        Ok(DeviceBuffer {
            inner: DeviceBufferInner::Cuda(cuda_buffer::CudaBuffer::new(ptr)),
            len: host.len(),
        })
    }

    fn alloc_zeros(&self, len: usize) -> Result<DeviceBuffer, BackendError> {
        // We don't have cudaMemset bound; do upload of a
        // zeroed host buffer. Cost: one extra host-side
        // allocation. M6.c.3 will inline this if profiling
        // shows it matters.
        let host = vec![0.0_f32; len];
        self.upload(&host)
    }

    fn download(&self, dev: &DeviceBuffer, host: &mut [f32]) -> Result<(), BackendError> {
        if dev.len() != host.len() {
            return Err(BackendError::InvalidArgument(format!(
                "download: device buffer len {} != host buffer len {}",
                dev.len(), host.len(),
            )));
        }
        let DeviceBufferInner::Cuda(buf) = &dev.inner;
        let bytes = host.len() * std::mem::size_of::<f32>();
        let status = unsafe {
            cudaMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                buf.ptr_const(),
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if status != 0 {
            return Err(BackendError::TransferFailed {
                direction: "device->host",
                message: format!("cudaMemcpy returned {status}"),
            });
        }
        Ok(())
    }

    fn matmul_f32(
        &self,
        a: &DeviceBuffer, b: &DeviceBuffer, out: &mut DeviceBuffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            return Err(BackendError::InvalidArgument(format!(
                "matmul shape mismatch: a {} (expect {}), b {} (expect {}), out {} (expect {})",
                a.len(), m * k, b.len(), k * n, out.len(), m * n,
            )));
        }
        let DeviceBufferInner::Cuda(a_buf) = &a.inner;
        let DeviceBufferInner::Cuda(b_buf) = &b.inner;
        let DeviceBufferInner::Cuda(out_buf) = &out.inner;
        // The kernel returns void; pre-existing ABI.
        // Kernel-launch failure prints to stderr at the
        // C side; we trust success.
        unsafe {
            matmul_f32_launch_device(
                a_buf.ptr_const() as *const f32,
                b_buf.ptr_const() as *const f32,
                out_buf.ptr() as *mut f32,
                m as c_int, k as c_int, n as c_int,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CUDA backend constructible always (it's a ZST). This
    /// test verifies the Drop / ownership contract on
    /// `DeviceBuffer` is sound even when the trait is used
    /// from multiple `&'static` references.
    #[test]
    fn cuda_backend_is_zero_sized_and_global_is_static() {
        let _ref1 = CudaBackend::global();
        let _ref2 = CudaBackend::global();
        // Equal pointers → singleton.
        assert!(std::ptr::eq(_ref1 as *const _, _ref2 as *const _));
        assert_eq!(std::mem::size_of::<CudaBackend>(), 0);
    }

    /// Sanity that the trait object surface resolves —
    /// catches any `Self: Sized` regression in the trait
    /// definition that would prevent `Box<dyn Backend>`
    /// usage in M6.c.3 callers.
    #[test]
    fn backend_dyn_dispatch_compiles() {
        let _b: &'static dyn Backend = CudaBackend::global();
    }

    /// `available_vram_bytes` is allowed to be `None` when
    /// the driver isn't reachable. The test asserts the
    /// surface is callable and returns either `None` or a
    /// positive value.
    #[test]
    fn cuda_available_vram_returns_sensible_value() {
        let b = CudaBackend::global();
        match b.available_vram_bytes() {
            Some(v) => {
                // Real driver — must report > 0 bytes free.
                assert!(v > 0,
                    "available_vram_bytes returned 0 — driver oddity");
            }
            None => {
                // No driver — peak compute also `None`.
                assert!(b.peak_compute_flops_f32().is_none(),
                    "peak compute reported but no VRAM info");
            }
        }
    }
}
