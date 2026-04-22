use crate::apx4_5::batch_matmul_cuda::batch_matmul_cuda;
use crate::tensor::Tensor;

pub fn dispatch_batch_matmul_cuda(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    batch_matmul_cuda(a, b, out, batch, m, k, n);
}
