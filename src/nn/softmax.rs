use rayon::prelude::*;

use crate::tensor::{Layout, Tensor};

/// Softmax along the last dimension.
/// For x with shape [N, D], apply softmax to each row.
pub fn softmax_last_dim(x: &Tensor) -> Tensor {
    assert!(
        x.shape.len() >= 1,
        "softmax_last_dim: tensor must have at least 1 dimension"
    );

    let ndim = x.shape.len();
    let last_dim = x.shape[ndim - 1];

    let cols = last_dim;

    let mut out = Tensor::with_layout(
        x.shape.clone(),
        0.0,
        x.device,
        Layout::Contiguous,
        x.dtype,
    );

    let x_data = &x.data;
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, chunk)| {
            let row_start = row * cols;
            let slice = &x_data[row_start..row_start + cols];

            let mut max_val = f32::NEG_INFINITY;
            for &v in slice {
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp = 0.0f32;
            let mut temp = Vec::with_capacity(cols);
            for &v in slice {
                let e = (v - max_val).exp();
                temp.push(e);
                sum_exp += e;
            }

            let inv_sum = 1.0f32 / sum_exp.max(1e-12);
            for i in 0..cols {
                chunk[i] = temp[i] * inv_sum;
            }
        });

    out
}

// ================================================================
// APX 1.5 — Softmax backward paralelo (TAREA 1)
// ================================================================
#[allow(dead_code)]
pub fn softmax_backward_parallel(prob: &Tensor, grad_out: &Tensor) -> Tensor {
    assert!(prob.shape.len() >= 1, "softmax_backward_parallel: tensor must have at least 1 dim");
    let cols = *prob
        .shape
        .last()
        .expect("softmax_backward_parallel requires last dimension");

    let mut grad_in = Tensor::with_layout(
        prob.shape.clone(),
        0.0,
        prob.device,
        Layout::Contiguous,
        prob.dtype,
    );

    grad_in
        .data
        .par_chunks_mut(cols)
        .zip(prob.data.par_chunks(cols))
        .zip(grad_out.data.par_chunks(cols))
        .for_each(|((dst, p), g)| {
            let dot = p.iter().zip(g).map(|(a, b)| a * b).sum::<f32>();
            for i in 0..cols {
                dst[i] = p[i] * (g[i] - dot);
            }
        });

    grad_in
}
