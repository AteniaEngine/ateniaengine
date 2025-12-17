use crate::matmul_dispatcher::batch_matmul_dispatch;
use crate::tensor::{Layout, Tensor};

pub fn batch_matmul_parallel(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 3, "batch_matmul_parallel: lhs must be [batch, m, k]");
    assert_eq!(b.shape.len(), 3, "batch_matmul_parallel: rhs must be [batch, k, n]");
    let batch = a.shape[0];
    assert_eq!(b.shape[0], batch, "batch_matmul_parallel: batch mismatch");
    let m = a.shape[1];
    let k = a.shape[2];
    assert_eq!(b.shape[1], k, "batch_matmul_parallel: inner dim mismatch");
    let n = b.shape[2];

    let mut out = Tensor::with_layout(
        vec![batch, m, n],
        0.0,
        a.device,
        Layout::Contiguous,
        a.dtype,
    );

    batch_matmul_dispatch(&a.data, &b.data, &mut out.data, batch, m, k, n);

    out
}
