use rayon::prelude::*;

use crate::tensor::Tensor;

/// ReLU activation: max(0, x)
pub fn relu(x: &Tensor) -> Tensor {
    let mut out = x.clone();

    out
        .data
        .par_iter_mut()
        .for_each(|o| {
            if *o < 0.0 {
                *o = 0.0;
            }
        });

    out
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: &Tensor) -> Tensor {
    let mut out = x.clone();

    out
        .data
        .par_iter_mut()
        .zip(x.data.par_iter())
        .for_each(|(o, v)| {
            let s = 1.0f32 / (1.0f32 + (-v).exp());
            *o = v * s;
        });

    out
}

/// GELU activation (approximation): 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
pub fn gelu(x: &Tensor) -> Tensor {
    let mut out = x.clone();

    out
        .data
        .par_iter_mut()
        .zip(x.data.par_iter())
        .for_each(|(o, v)| {
            let x = *v;
            let c = 0.79788456_f32; // sqrt(2/pi)
            let x3 = x * x * x;
            let inner = c * (x + 0.044715_f32 * x3);
            let t = inner.tanh();
            *o = 0.5_f32 * x * (1.0_f32 + t);
        });

    out
}
