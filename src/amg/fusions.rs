use crate::tensor::{Tensor, Layout};
use crate::nn::linear as nn_linear;
use crate::nn::softmax as nn_softmax;

/// APX 6.10: auxiliary forward path for full attention.
/// Does NOT record BackOps and does NOT touch graph tensors.
pub fn execute_fused_attention_full(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    bq: Option<&Tensor>,
    bk: Option<&Tensor>,
    bv: Option<&Tensor>,
    wproj: &Tensor,
    bias: Option<&Tensor>,
) -> Tensor {
    // 1) Q, K, V = X·W (+ optional bias)
    let q = nn_linear::linear(x, wq, bq);
    let k = nn_linear::linear(x, wk, bk);
    let v = nn_linear::linear(x, wv, bv);

    // 2) scores = Q·K^T, same as the naive attention graph.
    let k_t = transpose_2d(&k);
    let scores = nn_linear::matmul(&q, &k_t);

    // 3) probs = softmax(scores) over the last dimension
    let probs = nn_softmax::softmax_last_dim(&scores);

    // 4) out = probs·V
    let out = nn_linear::matmul(&probs, &v);

    // 5) proj = out·Wproj (+ optional bias)
    nn_linear::linear(&out, wproj, bias)
}

fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d expects a 2D tensor");
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut data = vec![0.0; t.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = t.data[r * cols + c];
        }
    }
    let new_shape = vec![cols, rows];
    let strides = Tensor::compute_strides(&new_shape, &Layout::Contiguous);
    Tensor {
        shape: new_shape,
        data,
        device: t.device,
        dtype: t.dtype,
        layout: Layout::Contiguous,
        strides,
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    }
}
