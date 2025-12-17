use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::cuda::fused_linear_silu::cuda_fused_linear_silu;
use atenia_engine::apx4_8::fused_linear_activation::exec_fused_linear_silu;

#[test]
fn test_fused_linear_silu_gpu_matches_cpu() {
    let m = 4;
    let k = 8;
    let n = 16;

    // Para este APX usamos Device::CPU como backend lógico (igual que cuda_matmul).
    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);

    let mut out_gpu = Tensor::zeros_new(&[m, n], Device::CPU);

    cuda_fused_linear_silu(&x.data, &w.data, &b.data, &mut out_gpu.data, m, k, n);

    // Referencia CPU usando implementación fusionada existente.
    let out_cpu = exec_fused_linear_silu(&x, &w, Some(&b));

    let mut max_diff = 0.0f32;
    for i in 0..(m * n) {
        let d = (out_cpu.data[i] - out_gpu.data[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("[APX 4.10] max_diff = {}", max_diff);
    assert!(max_diff < 1e-3);
}
