use crate::v17::cnn::conv2d::AbortFlag;
use crate::v17::compute::tensor::Tensor;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    InvalidInputShape,
    Aborted,
}

/// Elementwise ReLU activation: max(x, 0.0), without mutating the input.
pub fn relu(input: &Tensor, abort_flag: &AbortFlag) -> Result<Tensor, ActivationError> {
    if abort_flag.is_aborted() {
        return Err(ActivationError::Aborted);
    }

    // Any-rank tensor is accepted; shape is preserved.
    let shape = input.shape.clone();
    let mut out_data = input.data.clone();

    for i in 0..out_data.len() {
        if abort_flag.is_aborted() {
            return Err(ActivationError::Aborted);
        }
        let x = out_data[i];
        out_data[i] = if x < 0.0 { 0.0 } else { x };
    }

    Ok(Tensor {
        shape,
        data: out_data,
    })
}
