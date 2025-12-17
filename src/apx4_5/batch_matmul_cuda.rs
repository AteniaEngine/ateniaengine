use crate::cuda::batch_matmul::cuda_batch_matmul;

pub fn batch_matmul_cuda(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    cuda_batch_matmul(a, b, out, batch, m, k, n);
}
