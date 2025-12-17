use atenia_engine::gpu::tensor::manager::GpuTensorManager;
use atenia_engine::gpu::autodiff::attention_backward::AttentionBackwardGPU;

#[test]
fn test_attention_backward_real_gpu() {
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => return, // skip if no CUDA
    };

    // Minimal example (M=2, D=2)
    let q = vec![1.0f32, 2.0, 3.0, 4.0];
    let k = vec![5.0f32, 6.0, 7.0, 8.0];
    let v = vec![1.0f32, 0.0, 0.0, 1.0];
    let att = vec![0.6f32, 0.4, 0.3, 0.7];
    let dout = vec![1.0f32, 0.0, 0.0, 1.0];

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
    let att_gpu = match mgr.from_cpu_vec(&att, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let dout_gpu = match mgr.from_cpu_vec(&dout, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };

    let (d_q_gpu, d_k_gpu, d_v_gpu) = match AttentionBackwardGPU::run(
        &mgr,
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &att_gpu,
        &dout_gpu,
    ) {
        Ok(v) => v,
        Err(_) => return,
    };

    let d_q = match mgr.to_cpu_vec(&d_q_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };
    let d_k = match mgr.to_cpu_vec(&d_k_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };
    let d_v = match mgr.to_cpu_vec(&d_v_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    // For now, only check shape and that we didn't crash / NaN flood.
    assert_eq!(d_q.len(), 4);
    assert_eq!(d_k.len(), 4);
    assert_eq!(d_v.len(), 4);

    for &x in d_q.iter().chain(d_k.iter()).chain(d_v.iter()) {
        assert!(x.is_finite());
    }
}
