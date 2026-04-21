use crate::cuda::matmul::cuda_matmul;
use crate::tensor::Tensor;

pub fn gpu_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    // Build temporary CPU tensors to reuse the existing cuda_matmul path.
    // This does not change MatMul math; it only delegates computation to the
    // CUDA kernel when available.

    // Tensor A with shape [m, k]
    let ta = Tensor::new_cpu(vec![m, k], a.to_vec());

    // Tensor B with shape [k, n]
    let tb = Tensor::new_cpu(vec![k, n], b.to_vec());

    let tc = cuda_matmul(&ta, &tb, m, k, n);

    // Copy the result into the flat output buffer.
    assert_eq!(tc.numel(), out.len(), "gpu_matmul: unexpected output size");
    out.copy_from_slice(tc.as_cpu_slice());
}

pub fn gpu_add(_a: &[f32], _b: &[f32], _out: &mut [f32]) {
    unimplemented!("CUDA add not implemented yet (APX 4.0)");
}
