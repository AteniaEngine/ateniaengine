use crate::gpu::memory::{GpuPtr, GpuMemoryEngine};

pub struct TensorGPU {
    pub ptr: GpuPtr,
    pub rows: usize,
    pub cols: usize,
}

impl TensorGPU {
    pub fn new_from_cpu(
        mem: &GpuMemoryEngine,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self, ()> {
        let size = rows * cols * 4;
        let ptr = match mem.alloc(size) {
            Ok(p) => p,
            Err(_) => return Err(()),
        };
        if mem.copy_htod(&ptr, data).is_err() {
            let _ = mem.free(&ptr);
            return Err(());
        }
        Ok(Self { ptr, rows, cols })
    }

    pub fn empty(mem: &GpuMemoryEngine, rows: usize, cols: usize) -> Result<Self, ()> {
        let size = rows * cols * 4;
        let ptr = match mem.alloc(size) {
            Ok(p) => p,
            Err(_) => return Err(()),
        };
        Ok(Self { ptr, rows, cols })
    }

    pub fn to_cpu(&self, mem: &GpuMemoryEngine) -> Result<Vec<f32>, ()> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        if mem.copy_dtoh(&self.ptr, &mut out).is_err() {
            return Err(());
        }
        Ok(out)
    }
}
