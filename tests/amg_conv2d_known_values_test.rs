//! Absolute-correctness forward tests for AMG's `Conv2D`.
//!
//! Inputs and weights are small integer-valued tensors chosen so the
//! expected output can be computed by hand. Each expected value is
//! accompanied by an inline derivation. Tolerance is `1e-6` because
//! the hand computation is exact and the f32 rounding on sums this
//! small is well below that.

use atenia_engine::amg::nodes::Conv2DConfig;
use atenia_engine::amg::ops::conv2d::execute_conv2d as amg_conv2d;
use atenia_engine::tensor::{DType, Device, Layout, Tensor as AmgTensor};

const TOLERANCE: f32 = 1e-6;

fn amg_tensor(shape: Vec<usize>, data: Vec<f32>) -> AmgTensor {
    let mut t = AmgTensor::with_layout(
        shape,
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    assert_eq!(t.data.len(), data.len());
    t.data = data;
    t
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", ctx);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff < TOLERANCE,
            "{}: idx {}: got={} want={} diff={}",
            ctx, i, g, w, diff
        );
    }
}

/// Input 1x1x3x3 = [1..9], weight 1x1x2x2 = [1,0,0,1] (diagonal).
/// No bias, stride (1,1), padding (0,0). Output 1x1x2x2.
///
/// Hand derivation (weight(0,0)=1, weight(0,1)=0, weight(1,0)=0, weight(1,1)=1):
/// - out(0,0) = 1*1 + 0*2 + 0*4 + 1*5 = 6
/// - out(0,1) = 1*2 + 0*3 + 0*5 + 1*6 = 8
/// - out(1,0) = 1*4 + 0*5 + 0*7 + 1*8 = 12
/// - out(1,1) = 1*5 + 0*6 + 0*8 + 1*9 = 14
#[test]
fn conv2d_diagonal_kernel_no_padding() {
    let input = amg_tensor(
        vec![1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let weight = amg_tensor(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let cfg = Conv2DConfig::new((1, 1), (0, 0));

    let out = amg_conv2d(&input, &weight, None, &cfg);

    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_close(
        &out.data,
        &[6.0, 8.0, 12.0, 14.0],
        "conv2d_diagonal_kernel_no_padding",
    );
}

/// Input 1x1x2x2 = [1,2,3,4], weight 1x1x3x3 = [1..9].
/// Padding (1,1), stride (1,1). Output 1x1x2x2.
///
/// With padding=1 and kernel=3, the kernel center aligns with the
/// output position. For each output cell, some kernel taps fall on
/// padded (virtual zero) positions and are skipped.
///
/// out(0,0): valid (kh,kw) pairs (ih, iw):
///   (1,1)→input(0,0)=1 × weight(1,1)=5 = 5
///   (1,2)→input(0,1)=2 × weight(1,2)=6 = 12
///   (2,1)→input(1,0)=3 × weight(2,1)=8 = 24
///   (2,2)→input(1,1)=4 × weight(2,2)=9 = 36
///   sum = 77
/// out(0,1): valid contributions = 1*4 + 2*5 + 3*7 + 4*8 = 67
/// out(1,0): valid contributions = 1*2 + 2*3 + 3*5 + 4*6 = 47
/// out(1,1): valid contributions = 1*1 + 2*2 + 3*4 + 4*5 = 37
#[test]
fn conv2d_with_padding_3x3_kernel() {
    let input = amg_tensor(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let weight = amg_tensor(
        vec![1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let cfg = Conv2DConfig::new((1, 1), (1, 1));

    let out = amg_conv2d(&input, &weight, None, &cfg);

    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_close(
        &out.data,
        &[77.0, 67.0, 47.0, 37.0],
        "conv2d_with_padding_3x3_kernel",
    );
}

/// Same as diagonal_kernel test but with bias = [10.0].
/// Expected outputs shift by +10: [16, 18, 22, 24].
#[test]
fn conv2d_with_bias_shifts_output() {
    let input = amg_tensor(
        vec![1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let weight = amg_tensor(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let bias = amg_tensor(vec![1], vec![10.0]);
    let cfg = Conv2DConfig::new((1, 1), (0, 0));

    let out = amg_conv2d(&input, &weight, Some(&bias), &cfg);

    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_close(
        &out.data,
        &[16.0, 18.0, 22.0, 24.0],
        "conv2d_with_bias_shifts_output",
    );
}
