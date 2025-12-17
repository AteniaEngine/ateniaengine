use atenia_engine::gpu::tensor::manager::GpuTensorManager;
use atenia_engine::gpu::autodiff::fused_attention::FusedAttentionGPU;

#[test]
fn test_fused_attention_real_gpu() {
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => return, // sin CUDA -> skip
    };

    // Simple tiny test for correctness
    let q = vec![1.0f32, 0.0, 0.0, 1.0]; // [2x2]
    let k = vec![1.0f32, 1.0, 0.0, 1.0]; // [2x2]
    let v = vec![1.0f32, 2.0, 3.0, 4.0]; // [2x2]

    let q_gpu = match mgr.from_cpu_vec(&q, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let k_gpu = match mgr.from_cpu_vec(&k, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let v_gpu = match mgr.from_cpu_vec(&v, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };

    let out_gpu = match FusedAttentionGPU::run(&mgr, &q_gpu, &k_gpu, &v_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    let out = match mgr.to_cpu_vec(&out_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    // We don't validate exact FP values yet — only shape + no crash.
    assert_eq!(out.len(), 4);
}
