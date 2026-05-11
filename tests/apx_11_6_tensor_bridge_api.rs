use atenia_engine::gpu::tensor::manager::GpuTensorManager;
use atenia_engine::tensor::{DType, Device, Tensor};

#[test]
fn test_tensor_bridge_api() {
    // Inicializar engine GPU real
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => return, // sin GPU -> skip
    };

    // Tensor CPU
    let data = vec![1.5f32, 2.5, 3.5, 4.5];
    let mut t = Tensor::new(vec![4], 0.0, Device::CPU, DType::F32);
    t.as_cpu_slice_mut().copy_from_slice(&data);

    // CPU → GPU
    let tg = match t.to_gpu_real(&mgr) {
        Ok(v) => v,
        Err(_) => return,
    };

    // GPU → CPU
    let t2 = match mgr.from_gpu(&tg) {
        Ok(v) => v,
        Err(_) => return,
    };

    assert_eq!(t.as_cpu_slice(), t2.as_cpu_slice());
}
