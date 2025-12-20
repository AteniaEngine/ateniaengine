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

/// Minimal tensor container backing data with owned storage.
#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
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
            data,
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
            data: vec![fill; num_elements],
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
        let data = (0..num_elements).map(|_| random::<f32>()).collect();
        Self {
            shape,
            data,
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

    /// Returns the total number of elements represented by the tensor shape.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns a CPU-resident clone of this tensor.
    pub fn to_cpu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
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
    pub fn to_gpu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
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
            data: self.data.clone(),
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
        self.num_elements() * self.dtype.size_in_bytes()
    }

    /// Returns a zero-initialized tensor sharing metadata with `self`.
    pub fn zeros_like(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: vec![0.0; self.data.len()],
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
            self.grad = Some(vec![0.0; self.data.len()]);
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
        mgr.from_cpu_vec(&self.data, rows, cols)
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
            m.download_to_cpu(self.data.as_mut_ptr(), bytes);
            m.mark_synced();
        }
    }

    /// Synchronize from CPU to GPU at state level; in 8.4 does not touch real data.
    pub fn sync_gpu(&mut self) {
        let bytes = self.estimated_bytes();
        if let Some(ref mut m) = self.gpu {
            m.upload_from_cpu(self.data.as_ptr(), bytes);
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
