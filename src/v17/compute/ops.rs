#![allow(dead_code)]

use super::compute_errors::ComputeError;
use super::tensor::Tensor;

/// Elementwise addition of two tensors with identical shape.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError> {
    if a.shape != b.shape {
        return Err(ComputeError::ShapeMismatch(
            "add: shapes must match".to_string(),
        ));
    }
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();
    Tensor::new(a.shape.clone(), data).map_err(|e| ComputeError::ShapeMismatch(format!("add: {e}")))
}

/// Matrix multiplication for 2D tensors.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(ComputeError::ShapeMismatch(
            "matmul: both tensors must be 2D".to_string(),
        ));
    }
    let m = a.shape[0];
    let k1 = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    if k1 != k2 {
        return Err(ComputeError::ShapeMismatch(
            "matmul: inner dimensions must match".to_string(),
        ));
    }

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k1 {
                let a_idx = i * k1 + kk;
                let b_idx = kk * n + j;
                acc += a.data[a_idx] * b.data[b_idx];
            }
            out[i * n + j] = acc;
        }
    }

    Tensor::new(vec![m, n], out).map_err(|e| ComputeError::ShapeMismatch(format!("matmul: {e}")))
}

/// ReLU activation applied elementwise.
pub fn relu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data.iter().map(|v| v.max(0.0)).collect();
    Tensor::new(x.shape.clone(), data).expect("shape must remain valid")
}
