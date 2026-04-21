//! CPU implementation of the `NodeType::MaxPool2D` op (APX v20 M1).
//!
//! Layouts:
//! - input:   NCHW  `[n, c, h_in, w_in]`
//! - output:  NCHW  `[n, c, h_out, w_out]`
//!
//! Output spatial size:
//! - `h_out = (h_in + 2*pad_h - k_h) / stride_h + 1`
//! - `w_out = (w_in + 2*pad_w - k_w) / stride_w + 1`
//!
//! Padded positions that fall outside the real input are skipped
//! (never contribute to the max). Windows that land entirely inside
//! padded regions are left at `f32::NEG_INFINITY`; with non-zero
//! padding the caller should size the kernel so this cannot happen.
//!
//! Panics on invariant violations; consistent with the rest of
//! `execute_single_inner`.

use crate::amg::nodes::MaxPool2DConfig;
use crate::tensor::{DType, Layout, Tensor};

/// Tie-breaking: when multiple values in the window are equal to the
/// maximum, both forward and backward use the same deterministic
/// iteration order (`kh` outer loop, then `kw`). This guarantees that
/// the argmax recomputed in backward matches the position selected
/// during forward, ensuring gradient consistency without caching
/// indices.
pub fn execute_maxpool2d(input: &Tensor, config: &MaxPool2DConfig) -> Tensor {
    assert_eq!(input.shape.len(), 4, "MaxPool2D: input must be 4D (NCHW)");

    let n = input.shape[0];
    let c = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];

    let (k_h, k_w) = config.kernel;
    let (stride_h, stride_w) = config.stride;
    let (pad_h, pad_w) = config.padding;

    assert!(
        k_h > 0 && k_w > 0,
        "MaxPool2D: kernel must have positive dims, got ({}, {})",
        k_h,
        k_w
    );
    assert!(
        stride_h > 0 && stride_w > 0,
        "MaxPool2D: stride must be > 0, got ({}, {})",
        stride_h,
        stride_w
    );
    assert!(
        h_in + 2 * pad_h >= k_h && w_in + 2 * pad_w >= k_w,
        "MaxPool2D: kernel ({}x{}) larger than padded input ({}x{})",
        k_h,
        k_w,
        h_in + 2 * pad_h,
        w_in + 2 * pad_w
    );

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut out = Tensor::with_layout(
        vec![n, c, h_out, w_out],
        0.0,
        input.device,
        Layout::Contiguous,
        DType::F32,
    );

    let in_idx = |n_i: usize, c_i: usize, h: usize, w: usize| -> usize {
        (((n_i * c + c_i) * h_in) + h) * w_in + w
    };
    let out_idx = |n_i: usize, c_i: usize, oh: usize, ow: usize| -> usize {
        (((n_i * c + c_i) * h_out) + oh) * w_out + ow
    };

    for n_i in 0..n {
        for c_i in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
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
                            let v = input.data[in_idx(n_i, c_i, ih, iw)];
                            // Strict `>`: first-seen wins in tie-breaking.
                            // Backward replicates this condition exactly.
                            if v > max_val {
                                max_val = v;
                            }
                        }
                    }
                    out.data[out_idx(n_i, c_i, oh, ow)] = max_val;
                }
            }
        }
    }

    out
}

/// CPU backward pass for `NodeType::MaxPool2D` (APX v20 M1).
///
/// Produces a flat `grad_input` buffer (NCHW, length == `input.data.len()`)
/// that routes each element of `out_grad` to the input position that
/// was the argmax of its pooling window during forward.
///
/// Tie-breaking: when multiple values in the window are equal to the
/// maximum, both forward and backward use the same deterministic
/// iteration order (`kh` outer loop, then `kw`) with strict `>`
/// comparison (first-seen wins). This guarantees that the argmax
/// recomputed here matches the position selected during forward,
/// ensuring gradient consistency without caching indices.
///
/// Windows fully outside the real input region (all positions padded)
/// receive no gradient propagation.
pub fn execute_maxpool2d_backward(
    input: &Tensor,
    out_grad: &Tensor,
    config: &MaxPool2DConfig,
) -> Vec<f32> {
    assert_eq!(
        input.shape.len(),
        4,
        "MaxPool2D backward: input must be 4D (NCHW)"
    );
    assert_eq!(
        out_grad.shape.len(),
        4,
        "MaxPool2D backward: out_grad must be 4D (NCHW)"
    );

    let n = input.shape[0];
    let c = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];

    let (k_h, k_w) = config.kernel;
    let (stride_h, stride_w) = config.stride;
    let (pad_h, pad_w) = config.padding;

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(
        out_grad.shape,
        vec![n, c, h_out, w_out],
        "MaxPool2D backward: out_grad shape {:?} does not match expected [{}, {}, {}, {}]",
        out_grad.shape,
        n,
        c,
        h_out,
        w_out
    );

    let mut grad_input = vec![0.0_f32; n * c * h_in * w_in];

    let in_idx = |n_i: usize, c_i: usize, h: usize, w: usize| -> usize {
        (((n_i * c + c_i) * h_in) + h) * w_in + w
    };
    let out_idx = |n_i: usize, c_i: usize, oh: usize, ow: usize| -> usize {
        (((n_i * c + c_i) * h_out) + oh) * w_out + ow
    };

    for n_i in 0..n {
        for c_i in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    // Recompute argmax using the SAME iteration order
                    // and the SAME strict `>` comparison as forward.
                    let mut max_val = f32::NEG_INFINITY;
                    let mut best_pos: Option<usize> = None;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih_signed = (oh * stride_h) as isize + kh as isize
                                - pad_h as isize;
                            let iw_signed = (ow * stride_w) as isize + kw as isize
                                - pad_w as isize;
                            if ih_signed < 0 || iw_signed < 0 {
                                continue;
                            }
                            let ih = ih_signed as usize;
                            let iw = iw_signed as usize;
                            if ih >= h_in || iw >= w_in {
                                continue;
                            }
                            let pos = in_idx(n_i, c_i, ih, iw);
                            let v = input.data[pos];
                            if v > max_val {
                                max_val = v;
                                best_pos = Some(pos);
                            }
                        }
                    }
                    if let Some(p) = best_pos {
                        grad_input[p] += out_grad.data[out_idx(n_i, c_i, oh, ow)];
                    }
                    // best_pos == None → window entirely inside padding;
                    // no gradient to propagate (forward left NEG_INFINITY).
                }
            }
        }
    }

    grad_input
}
