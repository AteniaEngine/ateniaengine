use crate::v17::compute::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct Conv2DParams {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvError {
    InvalidInputShape,
    InvalidWeightShape,
    KernelLargerThanInput,
    InvalidPadding,
    InvalidStride,
    InvalidBiasShape,
    Aborted,
}

/// Minimal abort flag compatible with existing execution guards.
#[derive(Debug, Default, Clone, Copy)]
pub struct AbortFlag {
    aborted: bool,
}

impl AbortFlag {
    pub fn new() -> Self {
        Self { aborted: false }
    }
    pub fn abort(&mut self) {
        self.aborted = true;
    }
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }
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

fn idx_nchw_out(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
) -> usize {
    (((n * c_out + c) * h_out) + h) * w_out + w
}

fn idx_oihw(
    oc: usize,
    ic: usize,
    kh: usize,
    kw: usize,
    c_in: usize,
    k_h: usize,
    k_w: usize,
) -> usize {
    (((oc * c_in + ic) * k_h) + kh) * k_w + kw
}

/// Deterministic, auditable Conv2D CPU implementation.
///
/// Layouts:
/// - input:  NCHW  (n, c_in, h_in, w_in)
/// - weight: OIHW  (c_out, c_in, k_h, k_w)
/// - bias:   O     (c_out)
pub fn conv2d_cpu(
    input: &Tensor,
    weights: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2DParams,
    abort_flag: &AbortFlag,
) -> Result<Tensor, ConvError> {
    if abort_flag.is_aborted() {
        return Err(ConvError::Aborted);
    }

    let (stride_h, stride_w) = params.stride;
    let (pad_h, pad_w) = params.padding;

    if stride_h == 0 || stride_w == 0 {
        return Err(ConvError::InvalidStride);
    }

    let input_shape = &input.shape;
    if input_shape.len() != 4 {
        return Err(ConvError::InvalidInputShape);
    }
    let n = input_shape[0];
    let c_in = input_shape[1];
    let h_in = input_shape[2];
    let w_in = input_shape[3];

    let weight_shape = &weights.shape;
    if weight_shape.len() != 4 {
        return Err(ConvError::InvalidWeightShape);
    }
    let c_out = weight_shape[0];
    let c_in_w = weight_shape[1];
    let k_h = weight_shape[2];
    let k_w = weight_shape[3];

    if c_in_w != c_in {
        return Err(ConvError::InvalidWeightShape);
    }

    if k_h == 0 || k_w == 0 {
        return Err(ConvError::KernelLargerThanInput);
    }

    // Ensure kernel is not larger than the effective input region with padding.
    if h_in + 2 * pad_h < k_h || w_in + 2 * pad_w < k_w {
        return Err(ConvError::KernelLargerThanInput);
    }

    if let Some(bias_t) = bias {
        let b_shape = &bias_t.shape;
        if !(b_shape.len() == 1 && b_shape[0] == c_out) {
            return Err(ConvError::InvalidBiasShape);
        }
    }

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    // Allocate output tensor locally using public fields.
    let mut out_data = vec![0.0_f32; n * c_out * h_out * w_out];

    for n_idx in 0..n {
        if abort_flag.is_aborted() {
            return Err(ConvError::Aborted);
        }
        for oc in 0..c_out {
            if abort_flag.is_aborted() {
                return Err(ConvError::Aborted);
            }
            for oh in 0..h_out {
                if abort_flag.is_aborted() {
                    return Err(ConvError::Aborted);
                }
                for ow in 0..w_out {
                    let mut acc = 0.0_f32;
                    for ic in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let in_h = oh * stride_h + kh;
                                let in_w = ow * stride_w + kw;

                                // Apply padding by shifting coordinates and checking bounds.
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

                                let idx_in = idx_nchw(n_idx, ic, in_h_u, in_w_u, c_in, h_in, w_in);
                                let idx_w = idx_oihw(oc, ic, kh, kw, c_in, k_h, k_w);
                                let x = input.data[idx_in];
                                let w = weights.data[idx_w];
                                acc += x * w;
                            }
                        }
                    }

                    if let Some(bias_t) = bias {
                        acc += bias_t.data[oc];
                    }

                    let idx_out = idx_nchw_out(n_idx, oc, oh, ow, c_out, h_out, w_out);
                    out_data[idx_out] = acc;
                }
            }
        }
    }

    Ok(Tensor {
        shape: vec![n, c_out, h_out, w_out],
        data: out_data,
    })
}
