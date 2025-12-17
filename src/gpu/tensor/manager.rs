use crate::gpu::{
    memory::GpuMemoryEngine,
    ops::linear::LinearOp,
    tensor::TensorGPU,
};
use crate::tensor::{Tensor, Device, DType, Layout};

pub struct GpuTensorManager {
    pub mem: GpuMemoryEngine,
}

impl GpuTensorManager {
    pub fn new() -> Result<Self, ()> {
        let mem = match GpuMemoryEngine::new() {
            Ok(m) => m,
            Err(_) => return Err(()),
        };
        Ok(Self { mem })
    }

    pub fn linear(&self, x: &TensorGPU, w: &TensorGPU, b: &TensorGPU) -> Result<TensorGPU, ()> {
        let out = match TensorGPU::empty(&self.mem, x.rows, w.rows) {
            Ok(t) => t,
            Err(_) => return Err(()),
        };
        LinearOp::run(&x.ptr, &w.ptr, &b.ptr, &out.ptr, x.rows, x.cols, w.rows);
        Ok(out)
    }

    /// APX 11.5 — helper: subir datos CPU a un TensorGPU usando el engine interno.
    pub fn from_cpu_vec(&self, data: &[f32], rows: usize, cols: usize) -> Result<TensorGPU, ()> {
        TensorGPU::new_from_cpu(&self.mem, data, rows, cols)
    }

    /// APX 11.5 — helper: bajar datos GPU a Vec<f32> usando el engine interno.
    pub fn to_cpu_vec(&self, t: &TensorGPU) -> Result<Vec<f32>, ()> {
        t.to_cpu(&self.mem)
    }

    /// APX 11.6 — bring GPU data back into a CPU Tensor.
    pub fn from_gpu(&self, t: &TensorGPU) -> Result<Tensor, ()> {
        let cpu_vec = self.to_cpu_vec(t)?;
        let shape = vec![t.rows, t.cols];

        let mut tensor = Tensor::with_layout(
            shape,
            0.0,
            Device::CPU,
            Layout::Contiguous,
            DType::F32,
        );

        tensor.data.copy_from_slice(&cpu_vec);
        Ok(tensor)
    }
}
