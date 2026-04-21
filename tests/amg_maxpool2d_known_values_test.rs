//! Absolute-correctness forward tests for AMG's `MaxPool2D`.
//!
//! Inputs chosen so the expected max per window is obvious and can
//! be derived inline. Tolerance 1e-6 because max selection is exact.

use atenia_engine::amg::nodes::MaxPool2DConfig;
use atenia_engine::amg::ops::maxpool2d::execute_maxpool2d as amg_maxpool2d;
use atenia_engine::tensor::Tensor as AmgTensor;

const TOLERANCE: f32 = 1e-6;

fn amg_tensor(shape: Vec<usize>, data: Vec<f32>) -> AmgTensor {
    AmgTensor::new_cpu(shape, data)
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

/// 4x4 input, 2x2 non-overlapping windows (stride 2, no padding).
/// Output 2x2.
///
/// Input laid out as:
///   [ 1,  8,  2,  7]
///   [ 3,  4,  5,  6]
///   [ 9, 10, 13, 12]
///   [14, 11, 15, 16]
///
/// Windows and max:
///   top-left     (1, 8, 3, 4)       → 8
///   top-right    (2, 7, 5, 6)       → 7
///   bottom-left  (9, 10, 14, 11)    → 14
///   bottom-right (13, 12, 15, 16)   → 16
#[test]
fn maxpool2d_2x2_non_overlapping() {
    let input = amg_tensor(
        vec![1, 1, 4, 4],
        vec![
            1.0, 8.0, 2.0, 7.0, //
            3.0, 4.0, 5.0, 6.0, //
            9.0, 10.0, 13.0, 12.0, //
            14.0, 11.0, 15.0, 16.0,
        ],
    );
    let cfg = MaxPool2DConfig::non_overlapping((2, 2));

    let out = amg_maxpool2d(&input, &cfg);

    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_close(
        out.as_cpu_slice(),
        &[8.0, 7.0, 14.0, 16.0],
        "maxpool2d_2x2_non_overlapping",
    );
}

/// 4x4 input [1..16], 3x3 windows, stride 1, no padding.
/// Output (4-3)/1 + 1 = 2 → 2x2.
///
/// Windows (3x3 each):
///   out(0,0) = max of input[0..3, 0..3] = {1,2,3,5,6,7,9,10,11}    → 11
///   out(0,1) = max of input[0..3, 1..4] = {2,3,4,6,7,8,10,11,12}   → 12
///   out(1,0) = max of input[1..4, 0..3] = {5,6,7,9,10,11,13,14,15} → 15
///   out(1,1) = max of input[1..4, 1..4] = {6,7,8,10,11,12,14,15,16}→ 16
#[test]
fn maxpool2d_3x3_stride1() {
    let input = amg_tensor(
        vec![1, 1, 4, 4],
        (1..=16).map(|v| v as f32).collect(),
    );
    let cfg = MaxPool2DConfig::new((3, 3), (1, 1), (0, 0));

    let out = amg_maxpool2d(&input, &cfg);

    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_close(
        out.as_cpu_slice(),
        &[11.0, 12.0, 15.0, 16.0],
        "maxpool2d_3x3_stride1",
    );
}
