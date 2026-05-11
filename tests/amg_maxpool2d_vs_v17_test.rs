//! Forward-equivalence tests: AMG's `MaxPool2D` vs v17's `maxpool2d_cpu`.
//!
//! Both implementations operate on NCHW layouts with identical
//! iteration order (`kh` outer, `kw` inner) and strict `>` comparison
//! for tie-breaking (first-seen wins). This suite verifies numerical
//! equivalence within 1e-5 tolerance on fixed inputs.
//!
//! These tests assume that AMG and v17 accumulate floating-point
//! operations in the same order. If either implementation is ever
//! modified (e.g. loop reordering for SIMD optimization, parallel
//! accumulation, FMA instructions), these tests may need to use
//! approximate tolerance instead of strict 1e-5. The expected
//! behavior of the op would still be correct; only the numerical
//! match with v17 would loosen.

use atenia_engine::amg::nodes::MaxPool2DConfig;
use atenia_engine::amg::ops::maxpool2d::execute_maxpool2d as amg_maxpool2d;
use atenia_engine::tensor::Tensor as AmgTensor;
use atenia_engine::v17::cnn::conv2d::AbortFlag;
use atenia_engine::v17::cnn::maxpool2d::{MaxPool2DParams, maxpool2d_cpu as v17_maxpool2d};
use atenia_engine::v17::compute::tensor::Tensor as V17Tensor;

const TOLERANCE: f32 = 1e-5;

fn amg_tensor(shape: Vec<usize>, data: Vec<f32>) -> AmgTensor {
    AmgTensor::new_cpu(shape, data)
}

fn v17_tensor(shape: Vec<usize>, data: Vec<f32>) -> V17Tensor {
    V17Tensor::new(shape, data).expect("v17_tensor: shape/data mismatch")
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{}: length mismatch (got {}, want {})",
        ctx,
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff < TOLERANCE,
            "{}: mismatch at idx {}: got={} want={} diff={}",
            ctx,
            i,
            g,
            w,
            diff
        );
    }
}

#[test]
fn maxpool2d_2x2_stride2_no_padding() {
    let input_shape = vec![1, 1, 4, 4];
    let input_data: Vec<f32> = vec![
        1.0, 3.0, 2.0, 4.0, //
        5.0, 6.0, 8.0, 7.0, //
        9.0, 2.0, 1.0, 3.0, //
        4.0, 8.0, 6.0, 5.0,
    ];

    let amg_input = amg_tensor(input_shape.clone(), input_data.clone());
    let amg_cfg = MaxPool2DConfig::non_overlapping((2, 2));
    let amg_out = amg_maxpool2d(&amg_input, &amg_cfg);

    let v17_input = v17_tensor(input_shape, input_data);
    let v17_params = MaxPool2DParams {
        kernel: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };
    let flag = AbortFlag::new();
    let v17_out = v17_maxpool2d(&v17_input, &v17_params, &flag).expect("v17 maxpool2d failed");

    assert_eq!(amg_out.shape, v17_out.shape, "shape mismatch");
    assert_close(
        amg_out.as_cpu_slice(),
        &v17_out.data,
        "maxpool2d_2x2_stride2_no_padding",
    );
}

#[test]
fn maxpool2d_2x2_with_padding_stride1() {
    // 3x3 input + padding (1,1) + kernel (2,2) + stride (1,1)
    // Output spatial = (3 + 2 - 2) / 1 + 1 = 4 -> 4x4 output.
    // All output windows have at least one valid input position, so
    // no NEG_INFINITY cells remain.
    let input_shape = vec![1, 1, 3, 3];
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0,
    ];

    let amg_input = amg_tensor(input_shape.clone(), input_data.clone());
    let amg_cfg = MaxPool2DConfig::new((2, 2), (1, 1), (1, 1));
    let amg_out = amg_maxpool2d(&amg_input, &amg_cfg);

    let v17_input = v17_tensor(input_shape, input_data);
    let v17_params = MaxPool2DParams {
        kernel: (2, 2),
        stride: (1, 1),
        padding: (1, 1),
    };
    let flag = AbortFlag::new();
    let v17_out = v17_maxpool2d(&v17_input, &v17_params, &flag).expect("v17 maxpool2d failed");

    assert_eq!(amg_out.shape, v17_out.shape, "shape mismatch");
    assert_close(
        amg_out.as_cpu_slice(),
        &v17_out.data,
        "maxpool2d_2x2_with_padding_stride1",
    );
}
