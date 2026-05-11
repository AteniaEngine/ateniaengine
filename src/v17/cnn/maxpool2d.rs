use crate::v17::cnn::conv2d::AbortFlag;
use crate::v17::compute::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct MaxPool2DParams {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxPoolError {
    InvalidInputShape,
    InvalidKernel,
    InvalidStride,
    KernelLargerThanInput,
    Aborted,
}

fn idx_nchw(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
) -> usize {
    (((n * c_in + c) * h_in) + h) * w_in + w
}

/// Deterministic, explicit MaxPool2D over NCHW tensors.
pub fn maxpool2d_cpu(
    input: &Tensor,
    params: &MaxPool2DParams,
    abort_flag: &AbortFlag,
) -> Result<Tensor, MaxPoolError> {
    if abort_flag.is_aborted() {
        return Err(MaxPoolError::Aborted);
    }

    let (k_h, k_w) = params.kernel;
    let (stride_h, stride_w) = params.stride;
    let (pad_h, pad_w) = params.padding;

    if k_h == 0 || k_w == 0 {
        return Err(MaxPoolError::InvalidKernel);
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(MaxPoolError::InvalidStride);
    }

    let shape = &input.shape;
    if shape.len() != 4 {
        return Err(MaxPoolError::InvalidInputShape);
    }
    let n = shape[0];
    let c = shape[1];
    let h_in = shape[2];
    let w_in = shape[3];

    // Ensure kernel fits into padded input.
    if h_in + 2 * pad_h < k_h || w_in + 2 * pad_w < k_w {
        return Err(MaxPoolError::KernelLargerThanInput);
    }

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut out_data = vec![0.0_f32; n * c * h_out * w_out];

    for n_idx in 0..n {
        if abort_flag.is_aborted() {
            return Err(MaxPoolError::Aborted);
        }
        for c_idx in 0..c {
            if abort_flag.is_aborted() {
                return Err(MaxPoolError::Aborted);
            }
            for oh in 0..h_out {
                if abort_flag.is_aborted() {
                    return Err(MaxPoolError::Aborted);
                }
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let in_h = oh * stride_h + kh;
                            let in_w = ow * stride_w + kw;

                            let in_h_p = in_h as isize - pad_h as isize;
                            let in_w_p = in_w as isize - pad_w as isize;
                            if in_h_p < 0 || in_w_p < 0 {
                                continue;
                            }
                            let in_h_u = in_h_p as usize;
                            let in_w_u = in_w_p as usize;
                            if in_h_u >= h_in || in_w_u >= w_in {
                                continue;
                            }

                            let idx = idx_nchw(n_idx, c_idx, in_h_u, in_w_u, c, h_in, w_in);
                            let v = input.data[idx];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                    }

                    let out_idx = idx_nchw(n_idx, c_idx, oh, ow, c, h_out, w_out);
                    out_data[out_idx] = max_val;
                }
            }
        }
    }

    Ok(Tensor {
        shape: vec![n, c, h_out, w_out],
        data: out_data,
    })
}
