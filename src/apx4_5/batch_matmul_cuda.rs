use crate::cuda::batch_matmul::cuda_batch_matmul;
use crate::tensor::Tensor;

pub fn batch_matmul_cuda(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    cuda_batch_matmul(a, b, out, batch, m, k, n);
}
