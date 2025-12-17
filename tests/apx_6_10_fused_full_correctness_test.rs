use atenia_engine::tensor::{Tensor, Device, Layout};
use atenia_engine::amg::fusions;
use atenia_engine::nn::softmax as nn_softmax;
use atenia_engine::nn::linear as nn_linear;

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

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape, "Tensors must have same shape to compare");
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, v| acc.max(v))
}

#[test]
fn apx_6_10_fused_full_correctness() {
    // Dimensiones pequeñas 2D para correctness
    let m = 4;
    let dim = 8;

    let x = Tensor::randn(&[m, dim], Device::CPU);
    let wq = Tensor::randn(&[dim, dim], Device::CPU);
    let wk = Tensor::randn(&[dim, dim], Device::CPU);
    let wv = Tensor::randn(&[dim, dim], Device::CPU);
    let wproj = Tensor::randn(&[dim, dim], Device::CPU);
    let bias = Tensor::randn(&[dim], Device::CPU);

    // Ruta baseline
    let q = x.matmul(&wq);
    let k = x.matmul(&wk);
    let v = x.matmul(&wv);
    let k_t = transpose_2d(&k);
    let scores = q.matmul(&k_t);
    let probs = nn_softmax::softmax_last_dim(&scores);
    let out = probs.matmul(&v);
    let expected = nn_linear::linear(&out, &wproj, Some(&bias));

    // Ruta fused full (APX 6.10, solo benchmarking auxiliar)
    let fused = fusions::execute_fused_attention_full(
        &x,
        &wq,
        &wk,
        &wv,
        None,
        None,
        None,
        &wproj,
        Some(&bias),
    );

    let err = max_abs_diff(&expected, &fused);
    assert!(err <= 1e-4, "max abs diff = {} > 1e-4", err);
}
