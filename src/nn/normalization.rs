use rayon::prelude::*;

use crate::tensor::{Layout, Tensor};

/// RMSNorm over the last dimension.
/// For x with shape [N, D], normalize each row:
/// y[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + eps)
pub fn rms_norm(x: &Tensor, eps: f32) -> Tensor {
    assert!(
        x.shape.len() >= 1,
        "rms_norm: tensor must have at least 1 dimension"
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

    out
        .data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, chunk)| {
            let start = row * cols;
            let slice = &x.data[start..start + cols];
            let sum_sq: f32 = slice.iter().map(|v| v * v).sum();
            let mean_sq = sum_sq / (cols as f32).max(1.0);
            let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
            for (i, v) in slice.iter().enumerate() {
                chunk[i] = v * inv_rms;
            }
        });

    out
}

// ================================================================
// APX 1.5 — RMSNorm backward paralelo (TAREA 2)
// ================================================================
#[allow(dead_code)]
pub fn rmsnorm_backward_parallel(x: &Tensor, grad_out: &Tensor) -> Tensor {
    assert!(x.shape.len() >= 1, "rmsnorm_backward_parallel: tensor must have at least 1 dim");
    let cols = *x
        .shape
        .last()
        .expect("rmsnorm_backward_parallel requires last dimension");

    let mut grad_in = Tensor::with_layout(
        x.shape.clone(),
        0.0,
        x.device,
        Layout::Contiguous,
        x.dtype,
    );

    grad_in
        .data
        .par_chunks_mut(cols)
        .zip(x.data.par_chunks(cols))
        .zip(grad_out.data.par_chunks(cols))
        .for_each(|((dst, xrow), grow)| {
            let mean_sq = xrow.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let denom = (mean_sq + 1e-5).sqrt();
            for i in 0..cols {
                // simplified derivative (APX 1.5 only)
                dst[i] = grow[i] * (1.0 / denom);
            }
        });

    grad_in
}
