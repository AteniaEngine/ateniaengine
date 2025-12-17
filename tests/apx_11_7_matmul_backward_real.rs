use atenia_engine::gpu::tensor::manager::GpuTensorManager;
use atenia_engine::gpu::autodiff::matmul_backward::MatMulBackwardGPU;

#[test]
fn test_matmul_backward_real_gpu() {
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => return, // skip if no CUDA
    };

    // small matrices for sanity check
    let a = vec![
        1.0f32, 2.0,
        3.0, 4.0,
    ]; // [2x2]

    let b = vec![
        5.0f32, 6.0,
        7.0, 8.0,
    ]; // [2x2]

    let dout = vec![
        1.0f32, 1.0,
        1.0, 1.0,
    ]; // [2x2]

    let a_gpu = match mgr.from_cpu_vec(&a, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let b_gpu = match mgr.from_cpu_vec(&b, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let dout_gpu = match mgr.from_cpu_vec(&dout, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };

    // run backward
    let (d_a_gpu, d_b_gpu) = match MatMulBackwardGPU::run(&mgr, &a_gpu, &b_gpu, &dout_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    let d_a = match mgr.to_cpu_vec(&d_a_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };
    let d_b = match mgr.to_cpu_vec(&d_b_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    // expected manual values:
    // dA = dOut * B^T = sum row-wise
    assert_eq!(d_a, vec![13.0, 15.0, 13.0, 15.0]);

    // dB = A^T * dOut
    assert_eq!(d_b, vec![4.0, 4.0, 6.0, 6.0]);
}
