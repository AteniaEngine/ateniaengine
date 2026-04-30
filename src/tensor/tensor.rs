//! Core tensor structure definitions.

use rand::random;
use std::f32::consts::TAU;
use std::fmt;
use crate::gpu::tensor::manager::GpuTensorManager;
use crate::gpu::tensor::TensorGPU;
use crate::tensor::disk_tier::{self, DiskTensorHandle};

/// Supported data types for Atenia tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    FP8,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::FP8 => 1,
        }
    }
}

/// Supported execution devices for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    GPU,
}

/// Memory layout options for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Standard row-major contiguous memory.
    Contiguous,
    /// Channels-first configuration (NCHW).
    ChannelsFirst,
    /// Channels-last configuration (NHWC).
    ChannelsLast,
}

/// Physical storage backend for a [`Tensor`]'s element data.
///
/// `TensorStorage` encapsulates *where* the raw numeric bytes live (host RAM,
/// device VRAM, on-disk spillover, eventually ROCm / Metal buffers)
/// independent of the tensor's logical shape, dtype, or layout — those are
/// properties of the [`Tensor`] wrapper, not of the storage.
///
/// # Why an enum instead of a trait
///
/// - Dispatch through `dyn Storage` would add indirection on every element
///   access; we want the hot path to be a direct `match`.
/// - An exhaustive `match` forces every new backend to be handled explicitly
///   at each consumer — the right default for vendor-specific behavior.
///   CUDA / ROCm / Metal APIs diverge enough that a shared trait surface
///   would be a useless lowest common denominator.
///
/// # Variants
///
/// - `Cpu(Vec<f32>)` — host-resident dense storage; the canonical backend.
/// - `Cuda(TensorGPU)` — VRAM-resident storage owned through the refcounted
///   `Arc<InnerGpuPtr>` introduced in M3-d.1. Cloning a `Tensor` whose
///   storage is `Cuda` shares the same VRAM region; VRAM is released when
///   the last clone is dropped.
/// - `Disk(DiskTensorHandle)` — on-disk spillover introduced in M3-e.11.2
///   for the dual-pressure (VRAM + RAM saturated) reaction path. The
///   handle refcounts an on-disk file via `Arc<InnerDiskFile>`; the file
///   is removed best-effort when the last clone drops. Disk tensors are
///   opaque to ops — every consumer that reaches a `Disk` variant must
///   call [`Tensor::ensure_cpu`] first to materialize the data back in
///   host memory. In M3-e.11.2 the variant is **only produced by
///   external code** (none of the engine's current migration paths
///   produces it yet); bring-back via `ensure_cpu` is lazy and tested.
///
/// Additional backends (`Rocm`, `Metal`) are left for future milestones.
#[derive(Clone, Debug)]
pub enum TensorStorage {
    /// Host-resident dense `f32` storage.
    ///
    /// Element order follows the contiguous / NCHW / NHWC ordering declared
    /// by the owning [`Tensor`]'s `shape` and `layout` fields. The storage
    /// variant itself is layout-agnostic.
    Cpu(Vec<f32>),
    /// Host-resident dense BF16 storage as raw `u16` bits (M4.7.2).
    ///
    /// Each element holds the upper 16 bits of an F32 (sign + 8 exponent
    /// bits + 7 mantissa bits); decode is the lossless shift
    /// `f32::from_bits((bits as u32) << 16)`. Halves the RAM footprint of
    /// model parameters vs `Cpu(Vec<f32>)`, which is the load-bearing
    /// property for fitting 13B-class models on a 32 GB box.
    ///
    /// **Decode-on-access semantics.** This variant has no F32 cache:
    /// every `copy_to_cpu_vec()` materialises a fresh `Vec<f32>` by
    /// upcasting all elements; `as_cpu_slice()` and `as_cpu_slice_mut()`
    /// **panic** because exposing a borrowed `&[f32]` would require an
    /// internal cache or a temporary that outlives the borrow. Hot-path
    /// executors that need `&[f32]` operands materialise a transient
    /// `Vec<f32>` themselves and feed it to F32-only kernels — the
    /// pattern documented in M4.7.2's investigation report and applied
    /// inside `src/amg/graph.rs` MatMul / IndexSelect / BroadcastMul /
    /// BroadcastAdd arms.
    ///
    /// Produced today only by the loader (`WeightMapper` with
    /// `store_params_as_bf16=true`); never by ops, never by user code
    /// constructing tensors directly. Backward, training, GPU dispatch,
    /// and disk-spill paths panic with milestone-tagged messages when
    /// they encounter this variant.
    CpuBf16(Vec<u16>),
    /// Device-resident storage backed by a refcounted [`TensorGPU`].
    ///
    /// Entered via [`Tensor::ensure_gpu`] and exited via
    /// [`Tensor::ensure_cpu`]. Ownership of the VRAM region is shared
    /// between all clones of the owning `Tensor`; the region is freed when
    /// the last clone drops.
    Cuda(TensorGPU),
    /// Disk-spilled storage backed by a refcounted [`DiskTensorHandle`].
    ///
    /// Introduced in M3-e.11.2. Hot-path ops panic when encountering this
    /// variant via `as_cpu_slice` / `as_cpu_slice_mut`; callers must call
    /// [`Tensor::ensure_cpu`] first to bring the bytes back to host
    /// memory. `ensure_gpu` from `Disk` is a two-hop (disk → cpu → gpu)
    /// that reuses `ensure_cpu` and then the pre-existing CPU → VRAM
    /// path.
    Disk(DiskTensorHandle),
    /// **M5.c.2.a — host-resident shared F32 storage.**
    ///
    /// Identical contract to [`Self::Cpu`] for read access (the same
    /// `&[f32]` semantics) but the underlying buffer is owned by an
    /// [`Arc`] so multiple `Tensor` instances can reference the same
    /// physical bytes without duplication. The load-bearing M5 use
    /// case: prefill graph and decode graph register parameter slots
    /// whose `Tensor::storage` both point at the same `Arc`, so a
    /// 26 GB BF16 13B model stays at 26 GB even with two graphs
    /// instead of doubling to 52 GB.
    ///
    /// **Read-only.** `as_cpu_slice_mut` panics on this variant —
    /// the whole point of sharing is that no graph mutates the
    /// underlying buffer. `ensure_cpu` is a no-op on this variant
    /// (already CPU-resident); call sites that genuinely need a
    /// mutable owned buffer should call [`Tensor::ensure_owned`]
    /// first to clone-out into [`Self::Cpu`].
    ///
    /// **Spill is not supported.** The M4.7.5 selective-spill path
    /// (`graph.rs::migrate_selected_cpu_to_disk`) treats this
    /// variant as `Skipped`: spilling to disk would race with the
    /// other graph's view, and the `Arc` makes the answer
    /// "leave it where it is". M6 spill-aware sharing is out of
    /// M5 scope.
    CpuShared(std::sync::Arc<Vec<f32>>),
    /// **M5.c.2.a — host-resident shared BF16 storage.**
    ///
    /// BF16 counterpart of [`Self::CpuShared`]. Same Arc-sharing
    /// semantics; same decode-on-access contract as
    /// [`Self::CpuBf16`] (`copy_to_cpu_vec` materialises a fresh
    /// `Vec<f32>`; `as_cpu_slice` panics because exposing a
    /// borrowed `&[f32]` over BF16 storage would require an
    /// internal cache the M4.7.2 design deliberately rejects).
    /// `ensure_cpu` upcasts the BF16 buffer into a fresh
    /// owned `Cpu(Vec<f32>)` — this transitions the *owning*
    /// `Tensor` away from the shared view; siblings that still
    /// hold the `Arc` keep their BF16 view intact.
    CpuBf16Shared(std::sync::Arc<Vec<u16>>),
}

/// Decode a `u16` BF16 bit pattern into the corresponding `f32`.
///
/// BF16 occupies the upper 16 bits of an F32; the lower 16 mantissa bits
/// are zero. The conversion is lossless and round-trip-exact with the
/// down-convert `(f.to_bits() >> 16) as u16`.
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Down-convert an `f32` into the `u16` BF16 representation by truncation.
///
/// Round-trip-exact when `f` was previously produced by [`bf16_bits_to_f32`].
/// For arbitrary `f32` values this is a truncating cast that drops the
/// lower 16 mantissa bits — the M4.7.2 spike (commit `a786837`)
/// validated empirically that the resulting drift on the M4.6 family of
/// models stays well under the ADR-004 threshold.
#[inline]
pub fn f32_to_bf16_bits(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Failure modes for storage transitions (Cpu ↔ Cuda, Cpu ↔ Disk) in
/// [`Tensor::ensure_cpu`] / [`Tensor::ensure_gpu`] / disk spill APIs.
///
/// Renamed from `GpuTransferError` in M3-e.11.2 because the addition of
/// `TensorStorage::Disk` makes the old name inaccurate — the error is no
/// longer GPU-specific.
#[derive(Debug, Clone)]
pub enum StorageTransferError {
    /// `crate::gpu::gpu_engine()` returned `None` — no CUDA driver or
    /// compatible device is available in this process.
    EngineUnavailable,
    /// VRAM allocation or host→device copy failed. Both paths collapse to
    /// the same variant because `TensorGPU::new_from_cpu` does not
    /// distinguish the sub-step.
    AllocationFailed,
    /// Device→host copy failed on an already-resident GPU tensor.
    TransferFailed,
    /// M3-e.11.2: writing a tensor to disk failed. Typical causes are
    /// full disk, permission denied, or fs-level corruption. Produced by
    /// the disk-spill migration path (lands in M3-e.11.4) but defined
    /// here to give the error enum its final shape early.
    DiskWriteFailed(String),
    /// M3-e.11.2: reading a tensor back from disk failed during
    /// [`Tensor::ensure_cpu`] on a `TensorStorage::Disk` variant. The
    /// inner string carries the underlying [`std::io::Error`] message.
    DiskReadFailed(String),
    /// M3-e.11.2: the file on disk did not have the expected byte
    /// count — usually indicates a crash mid-write in a prior process
    /// that left a truncated file, or a shape mismatch between the
    /// handle and the owning [`Tensor`].
    DiskSizeMismatch { expected: usize, got: usize },
    /// Debt #3 Fase 3.2: the APX 4.12 pool has no block available to
    /// serve a request of `size_bytes`. Produced by
    /// `src/cuda/pool_helpers::with_pooled_device_buffers` when any
    /// `pool_alloc` call returns null. Before this variant existed,
    /// pool exhaustion surfaced as `cudaErrorInvalidDevicePointer`
    /// (code 11) because a null pointer would flow into `cudaMemcpy`
    /// — an actionable root cause masked as a CUDA-driver error.
    PoolExhausted { size_bytes: usize },
}

/// Minimal tensor container backing data with owned storage.
#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    /// Physical storage backend. In APX v20 M3-a only `TensorStorage::Cpu`
    /// exists; additional variants are added in later sub-milestones.
    pub storage: TensorStorage,
    pub device: Device,
    pub dtype: DType,
    pub layout: Layout,
    pub strides: Vec<usize>,
    pub grad: Option<Vec<f32>>,
    /// APX 11.1: optional reference to the operation that produced this tensor.
    pub op: Option<crate::ops::op_ref::OpRef>,
}

/// Lightweight alias used in IR/APX layers to reference tensors.
pub type TensorRef = Tensor;

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("layout", &self.layout)
            .field("strides", &self.strides)
            .finish()
    }
}

impl Tensor {
    /// Creates a tensor with samples from a standard normal distribution.
    pub fn randn(shape: &[usize], device: Device) -> Self {
        let shape_vec = shape.to_vec();
        let num_elements = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(num_elements);
        while data.len() < num_elements {
            let u1 = random::<f32>().max(1e-7);
            let u2 = random::<f32>();
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = TAU * u2;
            data.push(radius * theta.cos());
            if data.len() < num_elements {
                data.push(radius * theta.sin());
            }
        }
        let strides = Self::compute_strides(&shape_vec, &Layout::Contiguous);
        Self {
            shape: shape_vec,
            storage: TensorStorage::Cpu(data),
            device,
            dtype: DType::F32,
            layout: Layout::Contiguous,
            strides,
            grad: None,
            op: None,
        }
    }

    /// Computes strides for the provided `shape` given a particular `layout`.
    pub fn compute_strides(shape: &Vec<usize>, layout: &Layout) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }

        let rank = shape.len();
        let mut order: Vec<usize> = (0..rank).collect();
        if rank == 4 {
            match layout {
                Layout::Contiguous | Layout::ChannelsFirst => {
                    order = vec![0, 1, 2, 3];
                }
                Layout::ChannelsLast => {
                    // Move channel dimension (index 1) to the end: N, H, W, C ordering.
                    order = vec![0, 2, 3, 1];
                }
            }
        }

        let mut strides = vec![0; rank];
        let mut stride = 1usize;
        for &axis in order.iter().rev() {
            strides[axis] = stride;
            stride *= shape[axis].max(1);
        }

        strides
    }

    /// Creates a new tensor with the provided `shape`, filling its contents with `fill`.
    /// Uses a default contiguous layout.
    pub fn new(shape: Vec<usize>, fill: f32, device: Device, dtype: DType) -> Self {
        Self::with_layout(shape, fill, device, Layout::Contiguous, dtype)
    }

    /// Creates a tensor with the provided layout.
    pub fn with_layout(
        shape: Vec<usize>,
        fill: f32,
        device: Device,
        layout: Layout,
        dtype: DType,
    ) -> Self {
        let num_elements = shape.iter().product::<usize>();
        let strides = Self::compute_strides(&shape, &layout);
        Self {
            shape,
            storage: TensorStorage::Cpu(vec![fill; num_elements]),
            device,
            dtype,
            layout,
            strides,
            grad: None,
            op: None,
        }
    }

    /// Returns a tensor filled with zeros for the given shape.
    pub fn zeros(shape: Vec<usize>, device: Device, dtype: DType) -> Self {
        Self::new(shape, 0.0, device, dtype)
    }

    /// Convenience zero tensor constructor with default dtype F32.
    pub fn zeros_like_shape(shape: &[usize], device: Device) -> Self {
        Self::with_layout(shape.to_vec(), 0.0, device, Layout::Contiguous, DType::F32)
    }

    /// Creates a zero tensor from a slice shape and default dtype F32.
    pub fn zeros_from_slice(shape: &[usize], device: Device) -> Self {
        Self::zeros_like_shape(shape, device)
    }

    /// Returns a tensor filled with zeros for the given shape (slice variant).
    pub fn zeros_slice(shape: &[usize], device: Device) -> Self {
        Self::zeros_like_shape(shape, device)
    }

    /// Returns a tensor filled with zeros for the given shape (public slice API).
    pub fn zeros_new(shape: &[usize], device: Device) -> Self {
        Self::zeros_like_shape(shape, device)
    }

    /// Constructs a tensor of the given shape with `TensorStorage::Cuda`
    /// storage (uninitialised VRAM, M4.7.3.a).
    ///
    /// Used by residency-aware ops to pre-allocate the output buffer
    /// directly on VRAM, skipping the CPU-roundtrip that the legacy
    /// `cuda_matmul`/`try_gpu_matmul` path performs. The VRAM region is
    /// owned through the `Arc<InnerGpuPtr>` inside the `TensorGPU`;
    /// dropping the `Tensor` (or replacing its storage) frees the VRAM
    /// when the last clone goes away — same lifecycle as
    /// `Tensor::ensure_gpu`.
    ///
    /// Internally flattens `shape` into the `(rows, cols)` view that
    /// `TensorGPU` expects: `rows = numel`, `cols = 1`. The owning
    /// `Tensor::shape` field stays authoritative; `TensorGPU`'s rows/cols
    /// are an implementation detail of the storage.
    ///
    /// # Errors
    /// Returns [`StorageTransferError::EngineUnavailable`] if no CUDA
    /// engine is available, or [`StorageTransferError::AllocationFailed`]
    /// if VRAM allocation fails.
    pub fn zeros_new_cuda(shape: &[usize]) -> Result<Self, StorageTransferError> {
        let numel: usize = shape.iter().product();
        let gpu = TensorGPU::empty(numel, 1)
            .map_err(|_| StorageTransferError::AllocationFailed)?;
        let shape_vec = shape.to_vec();
        let strides = Self::compute_strides(&shape_vec, &Layout::Contiguous);
        Ok(Self {
            shape: shape_vec,
            storage: TensorStorage::Cuda(gpu),
            device: Device::GPU,
            dtype: DType::F32,
            layout: Layout::Contiguous,
            strides,
            grad: None,
            op: None,
        })
    }

    /// Decode-aware materialisation that preserves GPU residency.
    ///
    /// Brings the storage into a numerically-ready state — `Cpu(Vec<f32>)`
    /// or `Cuda(TensorGPU)` — without materialising a Cuda tensor back
    /// to host memory. Use in executor arms whose op has a residency-
    /// aware GPU kernel (M4.7.3): we want `CpuBf16`/`Disk` to decode but
    /// `Cuda` to flow through to the GPU dispatch unchanged.
    ///
    /// Behaviour by current variant:
    /// - `Cpu(_)`: no-op.
    /// - `Cuda(_)`: no-op (preserve residency).
    /// - `CpuBf16(_)`: decode to `Cpu(Vec<f32>)`, flip dtype to F32 (same
    ///   contract as [`Self::ensure_cpu`]).
    /// - `Disk(_)`: read to `Cpu(Vec<f32>)`.
    ///
    /// Arms whose op has no GPU kernel (RmsNorm, Softmax, RoPE, SiLU, …)
    /// continue to use the unconditional [`Self::ensure_cpu`].
    pub fn ensure_decoded(&mut self) -> Result<&mut Self, StorageTransferError> {
        match &self.storage {
            TensorStorage::Cpu(_) | TensorStorage::Cuda(_) => Ok(self),
            // M5.c.2.a — CpuShared is already F32-decoded; no-op
            // (sharing preserved). CpuBf16Shared decodes through
            // `ensure_cpu`, which transitions away from sharing.
            TensorStorage::CpuShared(_) => Ok(self),
            TensorStorage::CpuBf16(_) | TensorStorage::Disk(_)
            | TensorStorage::CpuBf16Shared(_) => self.ensure_cpu(),
        }
    }

    /// Returns a tensor filled with ones for the given shape.
    pub fn ones(shape: Vec<usize>, device: Device, dtype: DType) -> Self {
        Self::new(shape, 1.0, device, dtype)
    }

    /// Returns a tensor filled with random values sampled from `rand::random`.
    pub fn random(shape: Vec<usize>, device: Device, dtype: DType) -> Self {
        let layout = Layout::Contiguous;
        let strides = Self::compute_strides(&shape, &layout);
        let num_elements = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..num_elements).map(|_| random::<f32>()).collect();
        Self {
            shape,
            storage: TensorStorage::Cpu(data),
            device,
            dtype,
            layout,
            strides,
            grad: None,
            op: None,
        }
    }

    // ================================================================
    //  APX v20 M3-a — vendor-neutral storage API
    // ================================================================

    /// Constructs a CPU-resident tensor from an existing `Vec<f32>` payload.
    ///
    /// Uses `Device::CPU`, `DType::F32`, and `Layout::Contiguous`; strides
    /// are derived from `shape`. This is the canonical constructor for
    /// tensors with known initial contents — it replaces the pre-0.20
    /// pattern of building a `Tensor` struct literal with `data: vec![...]`.
    ///
    /// # Panics
    /// Panics if `data.len() != shape.iter().product()`.
    pub fn new_cpu(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Tensor::new_cpu: data length {} does not match shape product {} (shape = {:?})",
            data.len(),
            expected,
            shape
        );
        let strides = Self::compute_strides(&shape, &Layout::Contiguous);
        Self {
            shape,
            storage: TensorStorage::Cpu(data),
            device: Device::CPU,
            dtype: DType::F32,
            layout: Layout::Contiguous,
            strides,
            grad: None,
            op: None,
        }
    }

    /// Constructs a CPU-resident tensor with explicit device/dtype/layout.
    ///
    /// Wrapper around [`Tensor::new_cpu`] for call sites that historically
    /// built a `Tensor` struct literal with non-default `device`, `dtype`,
    /// or `layout` fields. Strides are computed from `shape` and `layout`;
    /// callers that need custom strides can assign `t.strides = ...` after
    /// construction.
    ///
    /// # Panics
    /// Panics if `data.len() != shape.iter().product()`.
    pub fn new_cpu_with_layout(
        shape: Vec<usize>,
        data: Vec<f32>,
        device: Device,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Tensor::new_cpu_with_layout: data length {} does not match shape product {} (shape = {:?})",
            data.len(),
            expected,
            shape
        );
        let strides = Self::compute_strides(&shape, &layout);
        Self {
            shape,
            storage: TensorStorage::Cpu(data),
            device,
            dtype,
            layout,
            strides,
            grad: None,
            op: None,
        }
    }

    /// Constructs a CPU-resident BF16 tensor from raw `u16` bit
    /// patterns (M4.7.2).
    ///
    /// `bits` must hold `shape.iter().product()` BF16 elements, each
    /// already in the canonical "upper 16 F32 bits" representation
    /// (the same form produced by [`f32_to_bf16_bits`] or by reading
    /// BF16 bytes off a safetensors checkpoint with little-endian
    /// `u16::from_le_bytes`). The tensor's `dtype` is set to
    /// [`DType::BF16`] to reflect the storage type; consumers that
    /// need an `&[f32]` view materialise one via
    /// [`Tensor::copy_to_cpu_vec`] (decode-on-access) or
    /// [`Tensor::ensure_cpu`] (eager upcast, transitions storage).
    ///
    /// # Panics
    /// Panics if `bits.len() != shape.iter().product()`.
    /// **M5.c.2.a** — construct a CPU-resident F32 tensor that
    /// shares its underlying buffer with other tensors via
    /// [`std::sync::Arc`]. The returned tensor's `storage` is
    /// [`TensorStorage::CpuShared`]; cloning the tensor (or
    /// constructing another tensor from `Arc::clone(&arc)`) does
    /// not duplicate the F32 buffer.
    ///
    /// # Panics
    /// Panics if `arc.len() != shape.iter().product()`.
    pub fn cpu_shared(shape: Vec<usize>, arc: std::sync::Arc<Vec<f32>>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            arc.len(),
            expected,
            "Tensor::cpu_shared: arc length {} does not match shape product {} (shape = {:?})",
            arc.len(),
            expected,
            shape
        );
        let strides = Self::compute_strides(&shape, &Layout::Contiguous);
        Self {
            shape,
            storage: TensorStorage::CpuShared(arc),
            device: Device::CPU,
            dtype: DType::F32,
            layout: Layout::Contiguous,
            strides,
            grad: None,
            op: None,
        }
    }

    /// **M5.c.2.a** — BF16 counterpart of [`Self::cpu_shared`].
    pub fn cpu_bf16_shared(shape: Vec<usize>, arc: std::sync::Arc<Vec<u16>>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            arc.len(),
            expected,
            "Tensor::cpu_bf16_shared: arc length {} does not match shape product {} (shape = {:?})",
            arc.len(),
            expected,
            shape
        );
        let strides = Self::compute_strides(&shape, &Layout::Contiguous);
        Self {
            shape,
            storage: TensorStorage::CpuBf16Shared(arc),
            device: Device::CPU,
            dtype: DType::BF16,
            layout: Layout::Contiguous,
            strides,
            grad: None,
            op: None,
        }
    }

    /// **M5.c.2.a** — break out of Arc-shared storage into an
    /// owned `Cpu(Vec<f32>)`. No-op for non-shared variants.
    /// After this call, mutating the tensor (`as_cpu_slice_mut`)
    /// no longer requires going through `ensure_cpu` first; the
    /// operation is also a one-shot transition that will not
    /// silently revert to shared on subsequent operations.
    pub fn ensure_owned(&mut self) -> Result<&mut Self, StorageTransferError> {
        match &self.storage {
            TensorStorage::CpuShared(arc) => {
                // Try to take ownership cheaply if uniquely
                // owned; otherwise clone the inner Vec.
                let owned = match std::sync::Arc::try_unwrap(arc.clone()) {
                    Ok(v) => v,
                    Err(arc) => (*arc).clone(),
                };
                self.storage = TensorStorage::Cpu(owned);
                Ok(self)
            }
            TensorStorage::CpuBf16Shared(_) => {
                // Route through the existing BF16-decode path to
                // get an owned Cpu(Vec<f32>). Same upcast as the
                // ensure_cpu CpuBf16Shared arm.
                self.ensure_cpu()
            }
            _ => Ok(self),
        }
    }

    pub fn new_cpu_bf16(shape: Vec<usize>, bits: Vec<u16>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            bits.len(),
            expected,
            "Tensor::new_cpu_bf16: bits length {} does not match shape product {} (shape = {:?})",
            bits.len(),
            expected,
            shape
        );
        let strides = Self::compute_strides(&shape, &Layout::Contiguous);
        Self {
            shape,
            storage: TensorStorage::CpuBf16(bits),
            device: Device::CPU,
            dtype: DType::BF16,
            layout: Layout::Contiguous,
            strides,
            grad: None,
            op: None,
        }
    }

    /// Replaces the current storage with raw BF16 `bits` (M4.7.2).
    ///
    /// Mirrors [`Self::set_cpu_data`] for the BF16 path. Used by the
    /// loader (`WeightMapper` with `store_params_as_bf16=true`) to
    /// down-convert the post-`LoadTransform` F32 working buffer into
    /// the persistent BF16 storage of a graph parameter that was
    /// pre-allocated with `Cpu` or `CpuBf16`. Updates `dtype` to
    /// [`DType::BF16`] to reflect the post-call storage.
    ///
    /// # Panics
    /// Panics if `bits.len() != self.numel()`.
    pub fn set_cpu_bf16_bits(&mut self, bits: Vec<u16>) {
        let expected = self.numel();
        assert_eq!(
            bits.len(),
            expected,
            "Tensor::set_cpu_bf16_bits: bits length {} does not match shape product {} (shape = {:?})",
            bits.len(),
            expected,
            self.shape
        );
        self.storage = TensorStorage::CpuBf16(bits);
        self.dtype = DType::BF16;
    }

    /// Replaces the current storage with CPU-resident `data`.
    ///
    /// Does not modify `shape` or `strides`; callers are expected to have
    /// constructed the tensor with the correct shape already. The previous
    /// storage is dropped. Replaces the pre-0.20 pattern `t.data = vec![...]`.
    ///
    /// # Panics
    /// Panics if `data.len() != self.numel()`.
    pub fn set_cpu_data(&mut self, data: Vec<f32>) {
        let expected = self.numel();
        assert_eq!(
            data.len(),
            expected,
            "Tensor::set_cpu_data: data length {} does not match shape product {} (shape = {:?})",
            data.len(),
            expected,
            self.shape
        );
        self.storage = TensorStorage::Cpu(data);
    }

    /// Returns an immutable view of the CPU-resident element buffer.
    ///
    /// # Panics
    /// Panics if the storage is not currently CPU-resident. Callers that
    /// need to work with tensors of any backend should call
    /// [`Tensor::ensure_cpu`] first.
    pub fn as_cpu_slice(&self) -> &[f32] {
        match &self.storage {
            TensorStorage::Cpu(v) => v.as_slice(),
            // M5.c.2.a — shared F32 storage exposes the borrow
            // through `Arc::as_ref`. The slice's lifetime is tied
            // to `&self`, which is correct: we never extract the
            // slice past the tensor's borrow.
            TensorStorage::CpuShared(arc) => arc.as_slice(),
            TensorStorage::CpuBf16(_) => panic!(
                "Tensor::as_cpu_slice called on a CpuBf16 tensor. \
                 CpuBf16 storage requires decode-on-access via \
                 copy_to_cpu_vec() (or ensure_cpu() to transition the \
                 storage variant); direct &[f32] borrow is not supported \
                 because exposing it would require an internal cache that \
                 M4.7.2 deliberately defers."
            ),
            TensorStorage::CpuBf16Shared(_) => panic!(
                "Tensor::as_cpu_slice called on a CpuBf16Shared tensor. \
                 Same decode-on-access contract as CpuBf16: route \
                 through copy_to_cpu_vec() or ensure_cpu()."
            ),
            TensorStorage::Cuda(_) => panic!(
                "Tensor::as_cpu_slice called on a GPU-resident tensor. \
                 Call ensure_cpu() first to transfer data to host memory."
            ),
            TensorStorage::Disk(_) => panic!(
                "Tensor::as_cpu_slice called on a Disk-resident tensor. \
                 Call ensure_cpu() first to materialize data back in host memory."
            ),
        }
    }

    /// Returns a mutable view of the CPU-resident element buffer.
    ///
    /// # Panics
    /// Panics if the storage is not currently CPU-resident.
    pub fn as_cpu_slice_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            TensorStorage::Cpu(v) => v.as_mut_slice(),
            // M5.c.2.a — Arc-shared storage is read-only by
            // construction. The whole point of sharing is that
            // no graph mutates the underlying buffer; if a caller
            // genuinely needs `&mut [f32]` they must first call
            // `ensure_owned()` to clone-out into Cpu storage.
            TensorStorage::CpuShared(_) => panic!(
                "Tensor::as_cpu_slice_mut called on CpuShared storage. \
                 Arc-shared parameter buffers are read-only by \
                 construction (M5.c.2.a). Call ensure_owned() first \
                 to clone-out into Cpu storage if mutation is needed."
            ),
            TensorStorage::CpuBf16Shared(_) => panic!(
                "Tensor::as_cpu_slice_mut called on CpuBf16Shared storage. \
                 Both BF16 (precision-preservation) and Shared (read-only) \
                 contracts forbid in-place mutation."
            ),
            TensorStorage::CpuBf16(_) => panic!(
                "Tensor::as_cpu_slice_mut called on a CpuBf16 tensor. \
                 BF16 parameters are read-only after load; mutating them \
                 in place would lose the precision contract. If a write \
                 path is genuinely needed, transition the variant via \
                 ensure_cpu() first."
            ),
            TensorStorage::Cuda(_) => panic!(
                "Tensor::as_cpu_slice_mut called on a GPU-resident tensor. \
                 Call ensure_cpu() first to transfer data to host memory."
            ),
            TensorStorage::Disk(_) => panic!(
                "Tensor::as_cpu_slice_mut called on a Disk-resident tensor. \
                 Call ensure_cpu() first to materialize data back in host memory."
            ),
        }
    }

    /// Returns an owned `Vec<f32>` copy of the element buffer.
    ///
    /// For CPU storage this is an O(n) clone. For `Cuda` storage this
    /// performs a device → host copy and returns a newly allocated vector;
    /// the tensor's storage variant is **not** changed.
    ///
    /// # Panics
    /// Panics if the device → host copy fails catastrophically (driver
    /// error). Callers that need structured error handling should call
    /// [`Tensor::ensure_cpu`] first and then [`Tensor::as_cpu_slice`].
    pub fn copy_to_cpu_vec(&self) -> Vec<f32> {
        match &self.storage {
            TensorStorage::Cpu(v) => v.clone(),
            // M5.c.2.a — Arc-shared F32: clone the inner Vec.
            // Same `O(n)` cost as the `Cpu` arm; the Arc just
            // enables the *underlying buffer* to be shared
            // across multiple Tensor wrappers without each
            // wrapper holding its own copy.
            TensorStorage::CpuShared(arc) => (**arc).clone(),
            // M5.c.2.a — Arc-shared BF16: identical decode
            // path to CpuBf16, just reading through the Arc.
            TensorStorage::CpuBf16Shared(arc) => {
                let bits = arc.as_slice();
                let mut out = vec![0.0_f32; bits.len()];
                crate::simd_kernels::avx2::bf16_decode_bulk(bits, &mut out);
                out
            }
            TensorStorage::CpuBf16(bits) => {
                // Decode-on-access (M4.7.2): materialise a fresh F32 vec
                // by upcasting every BF16 bit pattern. The upcast
                // `f32::from_bits((b as u32) << 16)` is lossless.
                //
                // M4.8.c: routed through `bf16_decode_bulk` (8-lane
                // AVX2 when available, scalar fallback otherwise) so
                // every consumer of `copy_to_cpu_vec` benefits from
                // the same decode-bandwidth lift `ensure_cpu` got.
                // No cache: each call allocates a new Vec<f32>;
                // callers that need to reuse the F32 view across
                // multiple ops should hold the returned Vec, or
                // transition the storage via `ensure_cpu`.
                let mut out = vec![0.0_f32; bits.len()];
                crate::simd_kernels::avx2::bf16_decode_bulk(bits, &mut out);
                out
            }
            TensorStorage::Cuda(g) => g
                .to_cpu()
                .expect("copy_to_cpu_vec: device->host copy failed (driver error)"),
            TensorStorage::Disk(handle) => {
                // Read the tensor bytes from disk without mutating
                // the storage variant — `copy_to_cpu_vec` is a
                // non-mutating accessor by contract. Callers that
                // want the storage transitioned should call
                // `ensure_cpu` instead.
                //
                // M4.7.4.d: dispatch on the on-disk dtype. BF16
                // files are upcast to f32 here, mirroring what
                // `ensure_cpu` does on its Disk arm — the
                // contract is "produce a Vec<f32>".
                match handle.dtype() {
                    disk_tier::DiskDtype::F32 => disk_tier::read_f32_tensor(handle).expect(
                        "copy_to_cpu_vec: disk read failed — use ensure_cpu for \
                         structured error handling, or verify that the underlying \
                         file still exists and the handle's numel is accurate.",
                    ),
                    disk_tier::DiskDtype::BF16 => {
                        let bits = disk_tier::read_bf16_tensor(handle).expect(
                            "copy_to_cpu_vec: disk read (bf16) failed — use ensure_cpu \
                             for structured error handling.",
                        );
                        // M4.8.c: SIMD bulk decode (see ensure_cpu BF16 arm).
                        let mut out = vec![0.0_f32; bits.len()];
                        crate::simd_kernels::avx2::bf16_decode_bulk(&bits, &mut out);
                        out
                    }
                }
            }
        }
    }

    /// Ensures the storage is CPU-resident after this call.
    ///
    /// - `Cpu` storage: no-op, returns `Ok`.
    /// - `Cuda` storage: performs a device → host copy, replaces the
    ///   storage with `TensorStorage::Cpu`, dropping the `Arc<InnerGpuPtr>`
    ///   (which frees the VRAM if this was the last clone).
    /// - `Disk` storage (M3-e.11.2): reads the file referenced by the
    ///   `DiskTensorHandle`, validates the byte count against the
    ///   tensor's `numel`, and replaces the storage with
    ///   `TensorStorage::Cpu`. Dropping the `Arc<InnerDiskFile>` from
    ///   the old storage variant triggers the file's best-effort
    ///   deletion via `InnerDiskFile::Drop`.
    ///
    /// Returns `&mut Self` on success to allow chaining. Disk size
    /// mismatches (file corrupted / truncated by a prior crash)
    /// produce [`StorageTransferError::DiskSizeMismatch`]; generic
    /// I/O errors produce [`StorageTransferError::DiskReadFailed`]
    /// with the underlying message.
    pub fn ensure_cpu(&mut self) -> Result<&mut Self, StorageTransferError> {
        match &self.storage {
            TensorStorage::Cpu(_) => Ok(self),
            // M5.c.2.a — Arc-shared F32 is already CPU-resident.
            // No-op; the Arc stays intact (siblings keep sharing).
            // Callers that want to break sharing should call
            // `ensure_owned()` instead.
            TensorStorage::CpuShared(_) => Ok(self),
            // M5.c.2.a — Arc-shared BF16: upcast into a fresh
            // owned `Cpu(Vec<f32>)`. This transitions the *owning*
            // tensor away from the shared view; siblings that
            // still hold the Arc keep their BF16 view intact.
            // Same upcast kernel as the CpuBf16 arm.
            TensorStorage::CpuBf16Shared(arc) => {
                let bits = arc.as_slice();
                let mut cpu_vec: Vec<f32> = vec![0.0; bits.len()];
                crate::simd_kernels::avx2::bf16_decode_bulk(bits, &mut cpu_vec);
                self.storage = TensorStorage::Cpu(cpu_vec);
                self.dtype = DType::F32;
                Ok(self)
            }
            TensorStorage::CpuBf16(bits) => {
                // M4.7.2: eager upcast BF16 → F32. After this the
                // storage is `Cpu(Vec<f32>)` and the original `Vec<u16>`
                // is dropped. Use this when a consumer needs `&[f32]`
                // residency rather than a transient decoded copy.
                //
                // M4.8.c: bulk decode via the SIMD-accelerated
                // `bf16_decode_bulk` (8-lane AVX2 when available,
                // scalar fallback otherwise). M4.8.a measured the
                // pre-M4.8.c scalar `iter().map().collect()` path at
                // 5.75 GB/s on a 70.78 M element tensor (49 ms);
                // M4.8.c's bulk decode targets ~30+ GB/s on the same
                // shape on the dev box.
                //
                // The `dtype` tag is also flipped to `F32` to match
                // the new storage. Downstream ops (e.g. `with_layout`
                // output construction in `MatMul`, the
                // `assert_eq!(a.dtype, b.dtype)` checks in `add` /
                // `mul` / `broadcast_add`) read `dtype` to decide
                // output dtype and to reject mixed-precision inputs;
                // leaving the clone tagged `BF16` while its bytes
                // are F32 surfaces as a "BF16 vs F32" panic on the
                // first inter-op boundary. Sweeping the tag here
                // keeps the contract local: after `ensure_cpu`, the
                // tensor is F32 in every observable sense.
                let mut cpu_vec: Vec<f32> = vec![0.0; bits.len()];
                crate::simd_kernels::avx2::bf16_decode_bulk(bits, &mut cpu_vec);
                self.storage = TensorStorage::Cpu(cpu_vec);
                self.dtype = DType::F32;
                Ok(self)
            }
            TensorStorage::Cuda(g) => {
                let cpu_vec = g
                    .to_cpu()
                    .map_err(|_| StorageTransferError::TransferFailed)?;
                self.storage = TensorStorage::Cpu(cpu_vec);
                Ok(self)
            }
            TensorStorage::Disk(handle) => {
                // M4.7.4.d: dispatch on the on-disk dtype tagged
                // by the M4.7.4.a writer. The Disk arm always
                // restores to `Cpu(Vec<f32>)` per decision #4 of
                // the M4.7.4 plan — `ensure_cpu` is the F32
                // residency primitive; a future M5
                // `ensure_resident` may keep the BF16 view for
                // memory-pressured intermediate tensors.
                //
                // For `DiskDtype::BF16` the upcast is a per-element
                // bf16 → f32 expansion (shift the 16 bits left by
                // 16, plus zero-fill of the trailing mantissa
                // bits — same arithmetic the M4.7.2 BF16
                // decode-on-access path uses). The transient
                // `Vec<u16>` returned by `read_bf16_tensor` lives
                // only across the `map` and is dropped before the
                // Cpu storage is assigned, so peak transient
                // memory is one BF16 buffer + one F32 buffer
                // (1.5 × the F32 footprint, no worse than the
                // M4.7.2 BF16 → F32 decode itself).
                let expected = self.numel();
                let data: Vec<f32> = match handle.dtype() {
                    disk_tier::DiskDtype::F32 => disk_tier::read_f32_tensor(handle)
                        .map_err(|e| StorageTransferError::DiskReadFailed(e.to_string()))?,
                    disk_tier::DiskDtype::BF16 => {
                        let bits = disk_tier::read_bf16_tensor(handle)
                            .map_err(|e| StorageTransferError::DiskReadFailed(e.to_string()))?;
                        // M4.8.c: SIMD bulk decode replaces scalar
                        // `iter().map().collect()`. The transient
                        // BF16 buffer is dropped after the call.
                        let mut out = vec![0.0_f32; bits.len()];
                        crate::simd_kernels::avx2::bf16_decode_bulk(&bits, &mut out);
                        out
                    }
                };
                if data.len() != expected {
                    return Err(StorageTransferError::DiskSizeMismatch {
                        expected,
                        got: data.len(),
                    });
                }
                // Assigning a new `storage` drops the previous
                // `TensorStorage::Disk(handle)`, which drops the
                // `Arc<InnerDiskFile>`. If this was the last clone,
                // `InnerDiskFile::Drop` removes the file from disk.
                self.storage = TensorStorage::Cpu(data);
                // M4.7.4.d: a BF16-spilled tensor whose `dtype`
                // was still `DType::BF16` (it was Cpu-bf16 at
                // spill time) now has F32 storage; flip the
                // logical dtype tag to match. Same rationale as
                // the M4.7.2 CpuBf16 → Cpu arm above: downstream
                // ops read `dtype` to dispatch and would panic
                // on a BF16-tagged tensor whose bytes are F32.
                self.dtype = DType::F32;
                Ok(self)
            }
        }
    }

    /// Ensures the storage is GPU-resident after this call.
    ///
    /// - `Cuda` storage: no-op, returns `Ok`.
    /// - `Cpu` storage: allocates VRAM, performs a host → device copy, and
    ///   replaces the storage with `TensorStorage::Cuda(TensorGPU)`. The
    ///   `TensorGPU` uses a flat `(1, numel)` view of the buffer; the
    ///   `Tensor`'s `shape` field remains authoritative.
    ///
    /// Does not touch `self.grad` (which remains CPU-resident per the M3
    /// contract that backward runs on CPU), `self.device` (logical flag
    /// remains decoupled from physical storage in M3-d.2), nor `self.gpu`
    /// (the APX 8.4 mirror stub is independent of this storage).
    ///
    /// Returns `&mut Self` on success to allow chaining.
    pub fn ensure_gpu(&mut self) -> Result<&mut Self, StorageTransferError> {
        if matches!(self.storage, TensorStorage::Cuda(_)) {
            return Ok(self);
        }

        // M3-e.11.2: Disk → Cuda is a two-hop (Disk → Cpu → Cuda).
        // We delegate to `ensure_cpu` to read the file back and
        // then fall through to the Cpu → Cuda path below. No
        // direct disk→device copy: the data has to pass through a
        // host buffer anyway for the H→D memcpy, so the extra hop
        // is cost-free in wall time and keeps the paths orthogonal.
        // M4.7.2: CpuBf16 → Cuda follows the same two-hop pattern
        // (CpuBf16 → Cpu via the BF16 decode, then Cpu → Cuda
        // through the F32 path). VRAM is F32 only today; M4.7.3
        // will introduce a residency-aware GPU path.
        if matches!(
            self.storage,
            TensorStorage::Disk(_) | TensorStorage::CpuBf16(_)
                | TensorStorage::CpuBf16Shared(_)
        ) {
            self.ensure_cpu()?;
            // Fall through to the Cpu branch below.
        }

        if crate::gpu::gpu_engine().is_none() {
            return Err(StorageTransferError::EngineUnavailable);
        }

        let cpu_data: &[f32] = match &self.storage {
            TensorStorage::Cpu(v) => v.as_slice(),
            // M5.c.2.a — Arc-shared F32: GPU upload reads
            // through the Arc; the shared tensor itself stays
            // CPU-resident (this method takes &mut self and
            // overwrites storage to Cuda below, breaking the
            // sharing locally to the calling tensor).
            TensorStorage::CpuShared(arc) => arc.as_slice(),
            // After the Disk → Cpu / CpuBf16 → Cpu hops above, we
            // are guaranteed Cpu here. Cuda was already handled by
            // the early return at the top of the function.
            TensorStorage::CpuBf16(_)
            | TensorStorage::CpuBf16Shared(_)
            | TensorStorage::Cuda(_)
            | TensorStorage::Disk(_) => unreachable!(
                "ensure_gpu reached a non-Cpu variant after the normalization step; \
                 this is a bug"
            ),
        };

        let numel = self.numel();
        let gpu = TensorGPU::new_from_cpu(cpu_data, 1, numel)
            .map_err(|_| StorageTransferError::AllocationFailed)?;

        self.storage = TensorStorage::Cuda(gpu);
        Ok(self)
    }

    /// Total number of elements in the tensor, computed from `shape`.
    ///
    /// Independent of the storage backend. O(rank).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Accessor to the underlying storage variant.
    ///
    /// Intended for code that needs an explicit `match` against specific
    /// backends (e.g. dispatch helpers). For element access prefer
    /// [`Tensor::as_cpu_slice`] / [`Tensor::as_cpu_slice_mut`] /
    /// [`Tensor::copy_to_cpu_vec`].
    pub fn storage(&self) -> &TensorStorage {
        &self.storage
    }

    // ================================================================
    //  Other pre-existing methods
    // ================================================================

    /// Returns a CPU-resident clone of this tensor.
    pub fn to_cpu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            storage: TensorStorage::Cpu(self.copy_to_cpu_vec()),
            device: Device::CPU,
            dtype: self.dtype,
            layout: self.layout,
            strides: self.strides.clone(),
            grad: self.grad.clone(),
            op: None,
        }
    }

    /// Returns a GPU-resident clone of this tensor.
    ///
    /// In M3-a this is still a logical relabeling — the storage stays
    /// CPU-resident but `device` is set to `GPU`. Real GPU residency is
    /// introduced in a later sub-milestone.
    pub fn to_gpu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            storage: TensorStorage::Cpu(self.copy_to_cpu_vec()),
            device: Device::GPU,
            dtype: self.dtype,
            layout: self.layout,
            strides: self.strides.clone(),
            grad: self.grad.clone(),
            op: None,
        }
    }

    /// Returns a clone of this tensor residing on the requested device.
    pub fn to_device(&self, dev: Device) -> Self {
        match dev {
            Device::CPU => self.to_cpu(),
            Device::GPU => self.to_gpu(),
        }
    }

    /// Returns a tensor view cast to a new dtype.
    pub fn cast_to(&self, dtype: DType) -> Self {
        if self.dtype == dtype {
            return self.clone();
        }

        Self {
            shape: self.shape.clone(),
            storage: self.storage.clone(),
            device: self.device,
            dtype,
            layout: self.layout,
            strides: self.strides.clone(),
            grad: self.grad.clone(),
            op: None,
        }
    }

    /// Estimates the amount of memory consumed by the tensor in bytes.
    pub fn estimated_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    /// Returns a zero-initialized tensor sharing metadata with `self`.
    pub fn zeros_like(&self) -> Tensor {
        let n = self.numel();
        Tensor {
            shape: self.shape.clone(),
            storage: TensorStorage::Cpu(vec![0.0; n]),
            device: self.device,
            dtype: self.dtype,
            layout: self.layout,
            strides: self.strides.clone(),
            grad: None,
            op: None,
        }
    }

    /// Initializes the gradient buffer to zeros if not already present.
    pub fn init_grad(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(vec![0.0; self.numel()]);
        }
    }

    /// Accumulates into the gradient buffer.
    pub fn add_grad(&mut self, g: &[f32]) {
        if self.grad.is_none() {
            self.init_grad();
        }
        if let Some(grad) = &mut self.grad {
            for (dst, src) in grad.iter_mut().zip(g.iter()) {
                *dst += *src;
            }
        }
    }

    /// Clears the gradient contents back to zero.
    pub fn clear_grad(&mut self) {
        if let Some(grad) = &mut self.grad {
            for v in grad.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// APX 11.6 — real GPU transfer using the manager's CUDA context.
    pub fn to_gpu_real(&self, mgr: &GpuTensorManager) -> Result<TensorGPU, ()> {
        if self.shape.is_empty() {
            return Err(());
        }
        let rows = self.shape[0];
        let cols = if self.shape.len() > 1 { self.shape[1] } else { 1 };
        mgr.from_cpu_vec(self.as_cpu_slice(), rows, cols)
    }

    /// Returns a reference to the operation that produced this tensor, if any.
    pub fn op(&self) -> Option<&crate::ops::op_ref::OpRef> {
        self.op.as_ref()
    }

    // === APX 8.4 GPU mirror / APX 8.5 persistence — removed ===
    //
    // Debt #2 cleanup: the metadata-only `GPUMirror` (APX 8.4) and
    // the unused `GPUPersistenceInfo` eviction heuristic (APX 8.5)
    // were removed. Their functionality is fully covered by
    // `TensorStorage::Cuda` (M3-d), which owns real VRAM via
    // `Arc<InnerGpuPtr>`. The `sync_cpu` / `sync_gpu` methods were
    // no-ops on `Cuda` and `Disk` storage and had no device pointer
    // of their own; consumers that need host-side data must call
    // `ensure_cpu()` (M3-d.2 invariant #4). Real eviction over
    // `TensorStorage::Cuda` is deferred to post-M4 pending real
    // workload measurement.
}
