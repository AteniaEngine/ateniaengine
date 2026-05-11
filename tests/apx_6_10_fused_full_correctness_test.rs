use atenia_engine::amg::fusions;
use atenia_engine::nn::linear as nn_linear;
use atenia_engine::nn::softmax as nn_softmax;
use atenia_engine::tensor::{Device, Layout, Tensor};

fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d expects a 2D tensor");
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut data = vec![0.0; t.numel()];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = t.as_cpu_slice()[r * cols + c];
        }
    }
    let new_shape = vec![cols, rows];
    Tensor::new_cpu_with_layout(new_shape, data, t.device, t.dtype, Layout::Contiguous)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape, "Tensors must have same shape to compare");
    a.as_cpu_slice()
        .iter()
        .zip(b.as_cpu_slice().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, v| acc.max(v))
}

#[test]
fn apx_6_10_fused_full_correctness() {
    // Small 2D dimensions for correctness
    let m = 4;
    let dim = 8;

    let x = Tensor::randn(&[m, dim], Device::CPU);
    let wq = Tensor::randn(&[dim, dim], Device::CPU);
    let wk = Tensor::randn(&[dim, dim], Device::CPU);
    let wv = Tensor::randn(&[dim, dim], Device::CPU);
    let wproj = Tensor::randn(&[dim, dim], Device::CPU);
    let bias = Tensor::randn(&[dim], Device::CPU);

    // Baseline path
    let q = x.matmul(&wq);
    let k = x.matmul(&wk);
    let v = x.matmul(&wv);
    let k_t = transpose_2d(&k);
    let scores = q.matmul(&k_t);
    let probs = nn_softmax::softmax_last_dim(&scores);
    let out = probs.matmul(&v);
    let expected = nn_linear::linear(&out, &wproj, Some(&bias));

    // Fused full path (APX 6.10, benchmarking helper only)
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
