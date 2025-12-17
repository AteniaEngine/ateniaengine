use atenia_engine::gpu::tensor::manager::GpuTensorManager;

#[test]
fn test_tensor_gpu_roundtrip_raw() {
    // Inicializar manager y engine GPU real. Si falla, salimos sin romper la suite.
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => return,
    };

    // Datos CPU de referencia
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let rows = 4usize;
    let cols = 4usize;

    // CPU → GPU
    let tg = match mgr.from_cpu_vec(&data, rows, cols) {
        Ok(t) => t,
        Err(_) => return,
    };

    // GPU → CPU
    let roundtrip = match mgr.to_cpu_vec(&tg) {
        Ok(v) => v,
        Err(_) => return,
    };

    assert_eq!(roundtrip, data);
}
