use std::fmt;
use std::sync::Arc;

use crate::gpu::memory::GpuPtr;
use crate::tensor::DType;

struct InnerGpuPtr {
    gpu_ptr: GpuPtr,
}

impl Drop for InnerGpuPtr {
    fn drop(&mut self) {
        if let Some(engine) = crate::gpu::gpu_engine() {
            let _ = engine.free(&self.gpu_ptr);
        }
    }
}

// `GpuPtr` holds an opaque device handle (u64) and a size (usize); both
// are `Send + Sync`. Sharing `InnerGpuPtr` across threads is sound under
// the shared CUDA context provided by the singleton engine.
unsafe impl Send for InnerGpuPtr {}
unsafe impl Sync for InnerGpuPtr {}

/// Logical GPU tensor handle.
///
/// Holds an [`Arc`]-shared device pointer plus the `(rows, cols)`
/// logical shape and a `dtype` describing the in-VRAM byte layout.
///
/// # M8.1 — `dtype` field
///
/// Pre-M8 `TensorGPU` always held F32 in VRAM; that assumption was
/// scattered across the loader, the matmul kernel ABI, and every
/// downstream consumer. M8.1 adds an explicit `dtype: DType` field
/// so a single buffer can hold either F32 or BF16 (the M8 BF16-
/// resident path uses BF16 in VRAM consumed directly by
/// `cublasGemmEx` without an upcast pass).
///
/// All pre-existing constructors (`empty`, `new_from_cpu`) default
/// to [`DType::F32`] for backward compatibility — every M3-M7
/// callsite keeps its prior contract. The new
/// [`TensorGPU::new_bf16`] constructor is the M8 entry point.
pub struct TensorGPU {
    inner: Arc<InnerGpuPtr>,
    pub rows: usize,
    pub cols: usize,
    /// **M8.1** — in-VRAM byte layout of this tensor.
    ///
    /// `F32` is the legacy path (4 bytes / element) — every
    /// pre-M8 caller of `empty` / `new_from_cpu` / `to_cpu`
    /// expects this and the constructors set it by default.
    ///
    /// `BF16` is the M8 path (2 bytes / element) — only allocated
    /// via `new_bf16` and read back via `to_cpu_bf16_bits`.
    /// Mixing the two on a single buffer is a programmer bug;
    /// `to_cpu` panics if dtype is not F32 (defensive against a
    /// future call site that forgets to switch).
    dtype: DType,
    /// **M10.3.1.1** — per-tensor matmul precision policy. `None`
    /// means "use the global `FAST_MODE_ACTIVE` fallback"
    /// (M10.3.1.0 contract). `Some(0)` = certified, `Some(1)` =
    /// fast. Encoded as `Option<u8>` instead of
    /// `Option<MatmulMode>` to keep this module dependency-free
    /// (the enum lives in `nn::llama::numcert`); the dispatcher
    /// translates back to the enum at the read site.
    matmul_policy_byte: Option<u8>,
}

/// Sentinel byte values mirroring `nn::llama::numcert::MatmulMode`.
/// Kept in sync by the inverse mapping in
/// `cuda_matmul_policy_to_byte` / the dispatcher's read site.
pub const MATMUL_POLICY_BYTE_CERTIFIED: u8 = 0;
pub const MATMUL_POLICY_BYTE_FAST: u8 = 1;

impl Clone for TensorGPU {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            rows: self.rows,
            cols: self.cols,
            dtype: self.dtype,
            matmul_policy_byte: self.matmul_policy_byte,
        }
    }
}

impl fmt::Debug for TensorGPU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorGPU")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("dtype", &self.dtype)
            .field("device_ptr", &format_args!("0x{:x}", self.device_ptr()))
            .field("size_bytes", &self.size_bytes())
            .finish()
    }
}

impl TensorGPU {
    /// Allocate a VRAM buffer and upload F32 host data into it.
    /// Pre-M8 default; sets `dtype = F32`.
    pub fn new_from_cpu(data: &[f32], rows: usize, cols: usize) -> Result<Self, ()> {
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let size = rows * cols * 4;
        let gpu_ptr = engine.alloc(size).map_err(|_| ())?;
        if engine.copy_htod(&gpu_ptr, data).is_err() {
            let _ = engine.free(&gpu_ptr);
            return Err(());
        }
        Ok(Self {
            inner: Arc::new(InnerGpuPtr { gpu_ptr }),
            rows,
            cols,
            dtype: DType::F32,
            matmul_policy_byte: None,
        })
    }

    /// Allocate an empty VRAM buffer for F32. Pre-M8 default;
    /// sets `dtype = F32`.
    pub fn empty(rows: usize, cols: usize) -> Result<Self, ()> {
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let size = rows * cols * 4;
        let gpu_ptr = engine.alloc(size).map_err(|_| ())?;
        Ok(Self {
            inner: Arc::new(InnerGpuPtr { gpu_ptr }),
            rows,
            cols,
            dtype: DType::F32,
            matmul_policy_byte: None,
        })
    }

    /// **M8.1** — allocate an empty VRAM buffer sized for `numel`
    /// BF16 elements (2 bytes each, no F32 inflation). The
    /// returned `TensorGPU` carries `dtype = BF16` and follows
    /// the engine convention `(rows = numel, cols = 1)`.
    ///
    /// This is the device-side allocation half of the BF16-
    /// resident path. The caller is responsible for populating
    /// the buffer (typically via
    /// [`crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast`] which
    /// allocates **and** uploads in one shot, or via a custom
    /// `cudaMemcpyAsync` pipeline in the M8.7 JIT path).
    ///
    /// Returns `Err(())` if the GPU engine is unavailable or the
    /// allocation fails (driver OOM).
    pub fn new_bf16(numel: usize) -> Result<Self, ()> {
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let size = numel * 2;
        let gpu_ptr = engine.alloc(size).map_err(|_| ())?;
        Ok(Self {
            inner: Arc::new(InnerGpuPtr { gpu_ptr }),
            rows: numel,
            cols: 1,
            dtype: DType::BF16,
            matmul_policy_byte: None,
        })
    }

    /// **M8.1** — allocate a VRAM buffer sized for `numel` BF16
    /// elements and upload the source `bits` slice into it via a
    /// single H→D byte copy. No upcast kernel; the device buffer
    /// holds raw BF16 bits identical to `bits`.
    ///
    /// The returned `TensorGPU` carries `dtype = BF16` and
    /// `(rows = numel, cols = 1)`. On error every device
    /// allocation made up to the failure point is freed via
    /// `InnerGpuPtr::Drop`.
    pub fn new_bf16_from_cpu(bits: &[u16]) -> Result<Self, ()> {
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let numel = bits.len();
        let size = numel * 2;
        let gpu_ptr = engine.alloc(size).map_err(|_| ())?;
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(bits.as_ptr() as *const u8, size)
        };
        if engine.copy_htod_bytes(&gpu_ptr, raw_bytes).is_err() {
            let _ = engine.free(&gpu_ptr);
            return Err(());
        }
        Ok(Self {
            inner: Arc::new(InnerGpuPtr { gpu_ptr }),
            rows: numel,
            cols: 1,
            dtype: DType::BF16,
            matmul_policy_byte: None,
        })
    }

    /// Download the F32 buffer into a fresh host `Vec<f32>`.
    ///
    /// **Panics** if `self.dtype != DType::F32` — the historical
    /// contract of `to_cpu` was F32-only and every legacy caller
    /// (autodiff, dispatcher, manager) relies on that. Mixing
    /// dtypes silently would corrupt the host buffer.
    pub fn to_cpu(&self) -> Result<Vec<f32>, ()> {
        assert_eq!(
            self.dtype,
            DType::F32,
            "TensorGPU::to_cpu requires F32 dtype; use to_cpu_bf16_bits for BF16 \
             tensors (got dtype={:?})",
            self.dtype
        );
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let mut out = vec![0.0f32; self.rows * self.cols];
        if engine.copy_dtoh(&self.inner.gpu_ptr, &mut out).is_err() {
            return Err(());
        }
        Ok(out)
    }

    /// **M8.1** — download a BF16-resident buffer's raw bit
    /// pattern as `Vec<u16>`. Returns `Err(())` if the dtype is
    /// not BF16 (mirrors the F32-only assertion of `to_cpu` but
    /// fails soft because callers of this method are M8-aware
    /// and may want to fallback rather than panic).
    pub fn to_cpu_bf16_bits(&self) -> Result<Vec<u16>, ()> {
        if self.dtype != DType::BF16 {
            return Err(());
        }
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let numel = self.rows * self.cols;
        let mut out = vec![0u16; numel];
        let raw: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, numel * 2)
        };
        if engine.copy_dtoh_bytes(&self.inner.gpu_ptr, raw).is_err() {
            return Err(());
        }
        Ok(out)
    }

    pub fn raw_ptr(&self) -> &GpuPtr {
        &self.inner.gpu_ptr
    }

    pub fn device_ptr(&self) -> u64 {
        self.inner.gpu_ptr.ptr
    }

    pub fn size_bytes(&self) -> usize {
        self.inner.gpu_ptr.size
    }

    /// **M8.1** — in-VRAM byte layout of this tensor. F32 unless
    /// the buffer was allocated via [`new_bf16`] / [`new_bf16_from_cpu`]
    /// or one of the M8 BF16-resident upload helpers in
    /// [`crate::cuda::bf16_to_f32`].
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// **M10.3.1.1** — read the per-tensor matmul precision
    /// policy byte. `None` means "use the global
    /// `FAST_MODE_ACTIVE` fallback" (M10.3.1.0 contract).
    ///
    /// The byte is `MATMUL_POLICY_BYTE_CERTIFIED` (0) or
    /// `MATMUL_POLICY_BYTE_FAST` (1). The dispatcher in
    /// `gpu::dispatch::hooks` reads this and routes between
    /// `cuda_matmul_bf16_inplace` and
    /// `cuda_matmul_bf16_native_inplace` accordingly.
    pub fn matmul_policy_byte(&self) -> Option<u8> {
        self.matmul_policy_byte
    }

    /// **M10.3.1.1** — set the per-tensor matmul precision
    /// policy. Called by `WeightStore::apply_per_tensor_policy`
    /// once after load with the manifest-resolved value for
    /// each weight tensor's name.
    pub fn set_matmul_policy_byte(&mut self, byte: Option<u8>) {
        self.matmul_policy_byte = byte;
    }
}
