use crate::apx4_5::batch_matmul_cuda::batch_matmul_cuda;

pub fn dispatch_batch_matmul_cuda(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    batch_matmul_cuda(a, b, out, batch, m, k, n);
}
