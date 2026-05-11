//! CPU implementation of the `NodeType::Conv2D` op (APX v20 M1).
//!
//! Layouts (enforced by assertions):
//! - input:   NCHW  `[n, c_in, h_in, w_in]`
//! - weight:  OIHW  `[c_out, c_in, k_h, k_w]`
//! - bias:    optional, shape `[c_out]`
//! - output:  NCHW  `[n, c_out, h_out, w_out]`
//!
//! Output spatial size:
//! - `h_out = (h_in + 2*pad_h - k_h) / stride_h + 1`
//! - `w_out = (w_in + 2*pad_w - k_w) / stride_w + 1`
//!
//! Panics on invariant violations (shape mismatches, zero stride, kernel
//! larger than padded input). These are programming errors, not runtime
//! conditions — consistent with the rest of `execute_single_inner`.

use crate::amg::nodes::Conv2DConfig;
use crate::tensor::{DType, Layout, Tensor};

/// Gradients produced by [`execute_conv2d_backward`].
///
/// All buffers are flat `Vec<f32>` in row-major contiguous layout,
/// matching the corresponding forward tensors:
/// - `grad_input`  layout NCHW, length == `n * c_in * h_in * w_in`
/// - `grad_weight` layout OIHW, length == `c_out * c_in * k_h * k_w`
/// - `grad_bias`   length == `c_out`, only present when the forward
///                 was called with a bias tensor.
pub struct Conv2DGrads {
    pub grad_input: Vec<f32>,
    pub grad_weight: Vec<f32>,
    pub grad_bias: Option<Vec<f32>>,
}

pub fn execute_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    config: &Conv2DConfig,
) -> Tensor {
    assert_eq!(input.shape.len(), 4, "Conv2D: input must be 4D (NCHW)");
    assert_eq!(weight.shape.len(), 4, "Conv2D: weight must be 4D (OIHW)");

    let n = input.shape[0];
    let c_in = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];

    let c_out = weight.shape[0];
    let c_in_w = weight.shape[1];
    let k_h = weight.shape[2];
    let k_w = weight.shape[3];

    assert_eq!(
        c_in, c_in_w,
        "Conv2D: input channels ({}) must match weight channels ({})",
        c_in, c_in_w
    );
    assert!(
        k_h > 0 && k_w > 0,
        "Conv2D: kernel must have positive spatial dims, got ({}, {})",
        k_h,
        k_w
    );

    let (stride_h, stride_w) = config.stride;
    let (pad_h, pad_w) = config.padding;
    assert!(
        stride_h > 0 && stride_w > 0,
        "Conv2D: stride must be > 0, got ({}, {})",
        stride_h,
        stride_w
    );
    assert!(
        h_in + 2 * pad_h >= k_h && w_in + 2 * pad_w >= k_w,
        "Conv2D: kernel ({}x{}) larger than padded input ({}x{})",
        k_h,
        k_w,
        h_in + 2 * pad_h,
        w_in + 2 * pad_w
    );

    if let Some(b) = bias {
        assert_eq!(
            b.shape,
            vec![c_out],
            "Conv2D: bias must have shape [{}], got {:?}",
            c_out,
            b.shape
        );
    }

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut out = Tensor::with_layout(
        vec![n, c_out, h_out, w_out],
        0.0,
        input.device,
        Layout::Contiguous,
        DType::F32,
    );

    // Row-major contiguous index helpers for NCHW / OIHW tensors.
    let in_idx = |n_i: usize, c: usize, h: usize, w: usize| -> usize {
        (((n_i * c_in + c) * h_in) + h) * w_in + w
    };
    let w_idx = |oc: usize, ic: usize, kh: usize, kw: usize| -> usize {
        (((oc * c_in + ic) * k_h) + kh) * k_w + kw
    };
    let out_idx = |n_i: usize, oc: usize, oh: usize, ow: usize| -> usize {
        (((n_i * c_out + oc) * h_out) + oh) * w_out + ow
    };

    let input_data = input.as_cpu_slice();
    let weight_data = weight.as_cpu_slice();
    let bias_data = bias.map(|b| b.as_cpu_slice());
    let out_data = out.as_cpu_slice_mut();

    for n_i in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0_f32;
                    for ic in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                // Apply padding by shifting coordinates.
                                let ih_signed =
                                    (oh * stride_h) as isize + kh as isize - pad_h as isize;
                                let iw_signed =
                                    (ow * stride_w) as isize + kw as isize - pad_w as isize;
                                if ih_signed < 0 || iw_signed < 0 {
                                    continue;
                                }
                                let ih = ih_signed as usize;
                                let iw = iw_signed as usize;
                                if ih >= h_in || iw >= w_in {
                                    continue;
                                }
                                let x = input_data[in_idx(n_i, ic, ih, iw)];
                                let w_v = weight_data[w_idx(oc, ic, kh, kw)];
                                acc += x * w_v;
                            }
                        }
                    }
                    if let Some(bd) = bias_data {
                        acc += bd[oc];
                    }
                    out_data[out_idx(n_i, oc, oh, ow)] = acc;
                }
            }
        }
    }

    out
}

/// CPU backward pass for `NodeType::Conv2D` (APX v20 M1).
///
/// Given the live forward tensors and the upstream gradient at the
/// output, produces flat gradient buffers for the three conv inputs.
/// The `bias` argument is only used for shape validation — its
/// gradient depends on `out_grad` alone.
///
/// The computation fuses the three gradient accumulations into a
/// single loop over output positions, reusing the exact index
/// arithmetic of the forward pass. Padding is handled identically.
///
/// Panics on shape mismatches; invariants should normally be caught
/// by forward earlier.
pub fn execute_conv2d_backward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out_grad: &Tensor,
    config: &Conv2DConfig,
) -> Conv2DGrads {
    assert_eq!(
        input.shape.len(),
        4,
        "Conv2D backward: input must be 4D (NCHW)"
    );
    assert_eq!(
        weight.shape.len(),
        4,
        "Conv2D backward: weight must be 4D (OIHW)"
    );
    assert_eq!(
        out_grad.shape.len(),
        4,
        "Conv2D backward: out_grad must be 4D (NCHW)"
    );

    let n = input.shape[0];
    let c_in = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];

    let c_out = weight.shape[0];
    let c_in_w = weight.shape[1];
    let k_h = weight.shape[2];
    let k_w = weight.shape[3];

    assert_eq!(
        c_in, c_in_w,
        "Conv2D backward: input channels ({}) must match weight channels ({})",
        c_in, c_in_w
    );

    let (stride_h, stride_w) = config.stride;
    let (pad_h, pad_w) = config.padding;

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(
        out_grad.shape,
        vec![n, c_out, h_out, w_out],
        "Conv2D backward: out_grad shape {:?} does not match expected [{}, {}, {}, {}]",
        out_grad.shape,
        n,
        c_out,
        h_out,
        w_out
    );

    if let Some(b) = bias {
        assert_eq!(
            b.shape,
            vec![c_out],
            "Conv2D backward: bias must have shape [{}], got {:?}",
            c_out,
            b.shape
        );
    }

    let has_bias = bias.is_some();

    let mut grad_input = vec![0.0_f32; n * c_in * h_in * w_in];
    let mut grad_weight = vec![0.0_f32; c_out * c_in * k_h * k_w];
    let mut grad_bias: Option<Vec<f32>> = if has_bias {
        Some(vec![0.0_f32; c_out])
    } else {
        None
    };

    let in_idx = |n_i: usize, c: usize, h: usize, w: usize| -> usize {
        (((n_i * c_in + c) * h_in) + h) * w_in + w
    };
    let w_idx = |oc: usize, ic: usize, kh: usize, kw: usize| -> usize {
        (((oc * c_in + ic) * k_h) + kh) * k_w + kw
    };
    let out_idx = |n_i: usize, oc: usize, oh: usize, ow: usize| -> usize {
        (((n_i * c_out + oc) * h_out) + oh) * w_out + ow
    };

    let input_data = input.as_cpu_slice();
    let weight_data = weight.as_cpu_slice();
    let out_grad_data = out_grad.as_cpu_slice();

    for n_i in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let g_out = out_grad_data[out_idx(n_i, oc, oh, ow)];

                    // grad_bias: Σ_{n, oh, ow} out_grad per output channel
                    if let Some(gb) = grad_bias.as_mut() {
                        gb[oc] += g_out;
                    }

                    for ic in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih_signed =
                                    (oh * stride_h) as isize + kh as isize - pad_h as isize;
                                let iw_signed =
                                    (ow * stride_w) as isize + kw as isize - pad_w as isize;
                                if ih_signed < 0 || iw_signed < 0 {
                                    continue;
                                }
                                let ih = ih_signed as usize;
                                let iw = iw_signed as usize;
                                if ih >= h_in || iw >= w_in {
                                    continue;
                                }

                                let in_i = in_idx(n_i, ic, ih, iw);
                                let w_i = w_idx(oc, ic, kh, kw);

                                let x = input_data[in_i];
                                let w_v = weight_data[w_i];

                                // grad_input[n, ic, ih, iw] += weight * g_out
                                grad_input[in_i] += w_v * g_out;
                                // grad_weight[oc, ic, kh, kw] += input * g_out
                                grad_weight[w_i] += x * g_out;
                            }
                        }
                    }
                }
            }
        }
    }

    Conv2DGrads {
        grad_input,
        grad_weight,
        grad_bias,
    }
}
