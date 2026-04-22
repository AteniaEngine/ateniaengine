//! Tensor-based batch matmul dispatch, moved from
//! `crate::apx4_5::batch_matmul_cuda`. Distinct from
//! [`crate::gpu::ops::batch_matmul`], which is the `TensorGPU`-based
//! real GPU op; this file is the thin dispatch layer used by the APX
//! 4.5 path, delegating to `crate::cuda::batch_matmul::cuda_batch_matmul`.

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
