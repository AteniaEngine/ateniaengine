//! Core tensor structure definitions.

use rand::random;
use std::f32::consts::TAU;
use std::fmt;
use crate::apx8::mirror::GPUMirror;
use crate::apx8::persistent::{GPUPersistenceInfo, next_global_step};
use crate::gpu::tensor::manager::GpuTensorManager;
use crate::gpu::tensor::TensorGPU;

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
/// device VRAM, eventually ROCm / Metal buffers) independent of the tensor's
/// logical shape, dtype, or layout — those are properties of the [`Tensor`]
/// wrapper, not of the storage.
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
    /// Device-resident storage backed by a refcounted [`TensorGPU`].
    ///
    /// Entered via [`Tensor::ensure_gpu`] and exited via
    /// [`Tensor::ensure_cpu`]. Ownership of the VRAM region is shared
    /// between all clones of the owning `Tensor`; the region is freed when
    /// the last clone drops.
    Cuda(TensorGPU),
}

/// Failure modes for host⇄device transfers in
/// [`Tensor::ensure_gpu`] / [`Tensor::ensure_cpu`].
#[derive(Debug, Clone)]
pub enum GpuTransferError {
    /// `crate::gpu::gpu_engine()` returned `None` — no CUDA driver or
    /// compatible device is available in this process.
    EngineUnavailable,
    /// VRAM allocation or host→device copy failed. Both paths collapse to
    /// the same variant because `TensorGPU::new_from_cpu` does not
    /// distinguish the sub-step.
    AllocationFailed,
    /// Device→host copy failed on an already-resident GPU tensor.
    TransferFailed,
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
    /// APX 8.4: optional GPU mirror. Does not affect numeric semantics.
    pub gpu: Option<GPUMirror>,
    /// APX 8.5: optional GPU persistence metadata.
    pub persistence: Option<GPUPersistenceInfo>,
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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
            op: None,
        }
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
            TensorStorage::Cuda(_) => panic!(
                "Tensor::as_cpu_slice called on a GPU-resident tensor. \
                 Call ensure_cpu() first to transfer data to host memory."
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
            TensorStorage::Cuda(_) => panic!(
                "Tensor::as_cpu_slice_mut called on a GPU-resident tensor. \
                 Call ensure_cpu() first to transfer data to host memory."
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
            TensorStorage::Cuda(g) => g
                .to_cpu()
                .expect("copy_to_cpu_vec: device->host copy failed (driver error)"),
        }
    }

    /// Ensures the storage is CPU-resident after this call.
    ///
    /// - `Cpu` storage: no-op, returns `Ok`.
    /// - `Cuda` storage: performs a device → host copy, replaces the
    ///   storage with `TensorStorage::Cpu`, dropping the `Arc<InnerGpuPtr>`
    ///   (which frees the VRAM if this was the last clone).
    ///
    /// Returns `&mut Self` on success to allow chaining.
    pub fn ensure_cpu(&mut self) -> Result<&mut Self, GpuTransferError> {
        if matches!(self.storage, TensorStorage::Cpu(_)) {
            return Ok(self);
        }

        let gpu = match &self.storage {
            TensorStorage::Cuda(g) => g,
            TensorStorage::Cpu(_) => unreachable!(),
        };
        let cpu_vec = gpu
            .to_cpu()
            .map_err(|_| GpuTransferError::TransferFailed)?;

        self.storage = TensorStorage::Cpu(cpu_vec);
        Ok(self)
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
    pub fn ensure_gpu(&mut self) -> Result<&mut Self, GpuTransferError> {
        if matches!(self.storage, TensorStorage::Cuda(_)) {
            return Ok(self);
        }

        if crate::gpu::gpu_engine().is_none() {
            return Err(GpuTransferError::EngineUnavailable);
        }

        let cpu_data: &[f32] = match &self.storage {
            TensorStorage::Cpu(v) => v.as_slice(),
            TensorStorage::Cuda(_) => unreachable!(),
        };

        let numel = self.numel();
        let gpu = TensorGPU::new_from_cpu(cpu_data, 1, numel)
            .map_err(|_| GpuTransferError::AllocationFailed)?;

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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
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
            gpu: None,
            persistence: None,
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

    // === APX 8.4: GPU mirroring helpers ===

    /// Ensure a GPU mirror exists for this tensor. Does not move real data in 8.4.
    pub fn ensure_gpu_mirror(&mut self) {
        if self.gpu.is_none() {
            let mut m = GPUMirror::new_empty();
            let bytes = self.estimated_bytes();
            m.allocate(bytes);
            self.gpu = Some(m);
        }
    }

    pub fn mark_gpu_dirty(&mut self) {
        if let Some(ref mut m) = self.gpu {
            m.mark_dirty_gpu();
        }
    }

    pub fn mark_cpu_dirty(&mut self) {
        if let Some(ref mut m) = self.gpu {
            m.mark_dirty_cpu();
        }
    }

    /// Synchronize from GPU to CPU at state level; in 8.4 does not touch real data.
    pub fn sync_cpu(&mut self) {
        let bytes = self.estimated_bytes();
        if let Some(ref mut m) = self.gpu {
            match &mut self.storage {
                TensorStorage::Cpu(v) => {
                    m.download_to_cpu(v.as_mut_ptr(), bytes);
                }
                // APX 8.4 mirror is metadata-only and independent of the
                // real VRAM path introduced via TensorStorage::Cuda.
                // Reconciliation of the two paths is pending.
                TensorStorage::Cuda(_) => {}
            }
            m.mark_synced();
        }
    }

    /// Synchronize from CPU to GPU at state level; in 8.4 does not touch real data.
    pub fn sync_gpu(&mut self) {
        let bytes = self.estimated_bytes();
        if let Some(ref mut m) = self.gpu {
            match &self.storage {
                TensorStorage::Cpu(v) => {
                    m.upload_from_cpu(v.as_ptr(), bytes);
                }
                // APX 8.4 mirror is metadata-only and independent of the
                // real VRAM path introduced via TensorStorage::Cuda.
                // Reconciliation of the two paths is pending.
                TensorStorage::Cuda(_) => {}
            }
            m.mark_synced();
        }
    }

    // === APX 8.5: GPU persistence helpers ===

    /// Enable GPU persistence for this tensor (metadata only; does not move data).
    pub fn enable_gpu_persistence(&mut self) {
        if self.persistence.is_none() {
            let bytes = self.estimated_bytes();
            self.persistence = Some(GPUPersistenceInfo {
                reuse_score: 0,
                last_used_step: 0,
                tensor_bytes: bytes,
                pinned: false,
            });
        }
    }

    /// Record a GPU use of the tensor for the persistence heuristic.
    pub fn note_gpu_use(&mut self) {
        if let Some(ref mut p) = self.persistence {
            p.reuse_score = p.reuse_score.saturating_add(1);
            p.last_used_step = next_global_step();
        }
    }

    /// Naive heuristic for cleaning the GPU mirror: if reuse_score is low and
    /// the tensor has not been used for many steps, drop the mirror.
    pub fn maybe_drop_gpu(&mut self, current_step: u64, max_age: u64) {
        if let (Some(p), Some(_)) = (&self.persistence, &self.gpu) {
            if !p.pinned && p.reuse_score < 2 {
                if current_step.saturating_sub(p.last_used_step) > max_age {
                    self.gpu = None;
                }
            }
        }
    }
}
