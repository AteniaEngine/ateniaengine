use crate::matmul_dispatcher::matmul_dispatch;
use crate::tensor::{Layout, Tensor};

pub fn matmul_parallel(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 2, "matmul_parallel: lhs must be 2D");
    assert_eq!(b.shape.len(), 2, "matmul_parallel: rhs must be 2D");
    assert_eq!(a.shape[1], b.shape[0], "matmul_parallel: inner dims must match");

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let mut out = Tensor::with_layout(
        vec![m, n],
        0.0,
        a.device,
        Layout::Contiguous,
        a.dtype,
    );

    matmul_dispatch(&a.data, &b.data, &mut out.data, m, k, n);

    out
}
