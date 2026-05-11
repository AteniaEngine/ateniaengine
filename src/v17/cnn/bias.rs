use crate::v17::cnn::conv2d::AbortFlag;
use crate::v17::compute::tensor::Tensor;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BiasError {
    InvalidInputShape,
    InvalidBiasShape,
    Aborted,
}

/// Add channel-wise bias to an NCHW tensor without mutating the input.
///
/// - input: NCHW [N, C, H, W]
/// - bias:  [C]
pub fn add_bias(
    input: &Tensor,
    bias: &Tensor,
    abort_flag: &AbortFlag,
) -> Result<Tensor, BiasError> {
    if abort_flag.is_aborted() {
        return Err(BiasError::Aborted);
    }

    let shape = &input.shape;
    if shape.len() != 4 {
        return Err(BiasError::InvalidInputShape);
    }
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];

    let b_shape = &bias.shape;
    if !(b_shape.len() == 1 && b_shape[0] == c) {
        return Err(BiasError::InvalidBiasShape);
    }

    let mut out_data = input.data.clone();

    for n_idx in 0..n {
        if abort_flag.is_aborted() {
            return Err(BiasError::Aborted);
        }
        for c_idx in 0..c {
            if abort_flag.is_aborted() {
                return Err(BiasError::Aborted);
            }
            let b = bias.data[c_idx];
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let idx = (((n_idx * c + c_idx) * h + h_idx) * w) + w_idx;
                    out_data[idx] += b;
                }
            }
        }
    }

    Ok(Tensor {
        shape: shape.clone(),
        data: out_data,
    })
}
