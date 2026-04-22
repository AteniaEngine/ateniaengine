use std::sync::Arc;

use crate::gpu::memory::GpuPtr;

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

pub struct TensorGPU {
    inner: Arc<InnerGpuPtr>,
    pub rows: usize,
    pub cols: usize,
}

impl Clone for TensorGPU {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl TensorGPU {
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
        })
    }

    pub fn empty(rows: usize, cols: usize) -> Result<Self, ()> {
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let size = rows * cols * 4;
        let gpu_ptr = engine.alloc(size).map_err(|_| ())?;
        Ok(Self {
            inner: Arc::new(InnerGpuPtr { gpu_ptr }),
            rows,
            cols,
        })
    }

    pub fn to_cpu(&self) -> Result<Vec<f32>, ()> {
        let engine = crate::gpu::gpu_engine().ok_or(())?;
        let mut out = vec![0.0f32; self.rows * self.cols];
        if engine.copy_dtoh(&self.inner.gpu_ptr, &mut out).is_err() {
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
}
