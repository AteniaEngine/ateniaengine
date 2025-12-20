use rayon::prelude::*;

use crate::tensor::{Layout, Tensor};
use crate::ops::linear::LinearOp;
use crate::ops::matmul::MatMulOp;

/// 2D matrix multiplication wrapper.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = a.matmul(b);
    // Register MatMulOp as the origin of this tensor for APX 11.2 (GPU backward IR).
    out.op = Some(MatMulOp::new());
    out
}

/// Linear layer:
/// x: [batch, in_features]
/// weight: [in_features, out_features]
/// bias: [out_features] (optional)
///
/// Returns: [batch, out_features]
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    assert!(
        x.shape.len() == 2,
        "linear: x must be 2D [batch, in_features]"
    );
    assert!(
        weight.shape.len() == 2,
        "linear: weight must be 2D [in_features, out_features]"
    );

    let batch = x.shape[0];
    let in_features_x = x.shape[1];
    let in_features_w = weight.shape[0];
    let out_features = weight.shape[1];

    assert_eq!(
        in_features_x, in_features_w,
        "linear: x and weight must agree on in_features"
    );

    if let Some(b) = bias {
        assert!(
            b.shape.len() == 1,
            "linear: bias must be 1D [out_features]"
        );
        assert_eq!(
            b.shape[0], out_features,
            "linear: bias length must match out_features"
        );
    }

    let mut out = Tensor::with_layout(
        vec![batch, out_features],
        0.0,
        x.device,
        Layout::Contiguous,
        x.dtype,
    );

    let weight_data = weight.data.clone();
    let bias_data = bias.map(|b| b.data.clone());

    out
        .data
        .par_chunks_mut(out_features)
        .enumerate()
        .for_each(|(row, chunk)| {
            let x_row = &x.data[row * in_features_x..row * in_features_x + in_features_x];
            for j in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features_x {
                    sum += x_row[k] * weight_data[k * out_features + j];
                }
                if let Some(bias_vals) = &bias_data {
                    chunk[j] = sum + bias_vals[j];
                } else {
                    chunk[j] = sum;
                }
            }
        });

    // Register the Linear op for APX 11.1 (GPU backward IR only, no execution yet).
    out.op = Some(LinearOp::new(weight.clone(), bias.cloned()));

    out
}
