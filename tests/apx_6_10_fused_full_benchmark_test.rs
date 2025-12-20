use atenia_engine::tensor::{Tensor, Device, Layout};
use atenia_engine::amg::fusions;
use atenia_engine::nn::softmax as nn_softmax;
use atenia_engine::nn::linear as nn_linear;
use std::time::Instant;

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
fn apx_6_10_fused_full_benchmark() {
    let size_list = [128, 256, 512];

    for dim in size_list {
        let m = dim; // rows

        let x = Tensor::randn(&[m, dim], Device::CPU);
        let wq = Tensor::randn(&[dim, dim], Device::CPU);
        let wk = Tensor::randn(&[dim, dim], Device::CPU);
        let wv = Tensor::randn(&[dim, dim], Device::CPU);
        let wproj = Tensor::randn(&[dim, dim], Device::CPU);
        let bias = Tensor::randn(&[dim], Device::CPU);

        // Baseline
        let t0 = Instant::now();
        let q = x.matmul(&wq);
        let k = x.matmul(&wk);
        let v = x.matmul(&wv);
        let k_t = transpose_2d(&k);
        let scores = q.matmul(&k_t);
        let probs = nn_softmax::softmax_last_dim(&scores);
        let out = probs.matmul(&v);
        let expected = nn_linear::linear(&out, &wproj, Some(&bias));
        let baseline_us = t0.elapsed().as_micros();

        // Fused 6.10 (benchmarking helper only)
        let t1 = Instant::now();
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
        let fused_us = t1.elapsed().as_micros();

        if std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1") {
            println!(
                "[APX 6.10 FUSED FULL] dim={} -> baseline={}us fused={}us speedup={}x",
                dim,
                baseline_us,
                fused_us,
                (baseline_us as f32) / (fused_us as f32),
            );
        }

        // Mathematical validity (benchmark): we only record the error; strict
        // correctness is covered in apx_6_10_fused_full_correctness_test.
        let err = max_abs_diff(&expected, &fused);
        if std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1") {
            println!("[APX 6.10 FUSED FULL] dim={} max_abs_diff={}", dim, err);
        }
    }
}
