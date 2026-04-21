//! Forward-equivalence tests: AMG's `Conv2D` vs v17's `conv2d_cpu`.
//!
//! Both implementations operate on NCHW/OIHW layouts with identical
//! padding and stride semantics. This suite verifies that, for a
//! fixed set of hardcoded inputs, AMG reproduces v17's outputs within
//! 1e-5 tolerance (standard f32 equivalence threshold).
//!
//! These tests assume that AMG and v17 accumulate floating-point
//! operations in the same order. If either implementation is ever
//! modified (e.g. loop reordering for SIMD optimization, parallel
//! accumulation, FMA instructions), these tests may need to use
//! approximate tolerance instead of strict 1e-5. The expected
//! behavior of the op would still be correct; only the numerical
//! match with v17 would loosen.

use atenia_engine::amg::nodes::Conv2DConfig;
use atenia_engine::amg::ops::conv2d::execute_conv2d as amg_conv2d;
use atenia_engine::tensor::Tensor as AmgTensor;
use atenia_engine::v17::cnn::conv2d::{conv2d_cpu as v17_conv2d, AbortFlag, Conv2DParams};
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
fn conv2d_no_bias_no_padding_stride1() {
    let input_shape = vec![1, 1, 4, 4];
    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();

    let weight_shape = vec![1, 1, 3, 3];
    let weight_data: Vec<f32> = (1..=9).map(|v| v as f32 * 0.1).collect();

    let amg_input = amg_tensor(input_shape.clone(), input_data.clone());
    let amg_weight = amg_tensor(weight_shape.clone(), weight_data.clone());
    let amg_cfg = Conv2DConfig::new((1, 1), (0, 0));
    let amg_out = amg_conv2d(&amg_input, &amg_weight, None, &amg_cfg);

    let v17_input = v17_tensor(input_shape, input_data);
    let v17_weight = v17_tensor(weight_shape, weight_data);
    let v17_params = Conv2DParams {
        stride: (1, 1),
        padding: (0, 0),
    };
    let flag = AbortFlag::new();
    let v17_out = v17_conv2d(&v17_input, &v17_weight, None, &v17_params, &flag)
        .expect("v17 conv2d failed");

    assert_eq!(amg_out.shape, v17_out.shape, "shape mismatch");
    assert_close(
        amg_out.as_cpu_slice(),
        &v17_out.data,
        "conv2d_no_bias_no_padding_stride1",
    );
}

#[test]
fn conv2d_with_bias() {
    let input_shape = vec![1, 2, 3, 3];
    let input_data: Vec<f32> = (1..=18).map(|v| v as f32).collect();

    let weight_shape = vec![2, 2, 2, 2];
    let weight_data: Vec<f32> = (1..=16).map(|v| v as f32 * 0.05).collect();

    let bias_data: Vec<f32> = vec![0.7, -0.3];

    let amg_input = amg_tensor(input_shape.clone(), input_data.clone());
    let amg_weight = amg_tensor(weight_shape.clone(), weight_data.clone());
    let amg_bias = amg_tensor(vec![2], bias_data.clone());
    let amg_cfg = Conv2DConfig::new((1, 1), (0, 0));
    let amg_out = amg_conv2d(&amg_input, &amg_weight, Some(&amg_bias), &amg_cfg);

    let v17_input = v17_tensor(input_shape, input_data);
    let v17_weight = v17_tensor(weight_shape, weight_data);
    let v17_bias = v17_tensor(vec![2], bias_data);
    let v17_params = Conv2DParams {
        stride: (1, 1),
        padding: (0, 0),
    };
    let flag = AbortFlag::new();
    let v17_out = v17_conv2d(
        &v17_input,
        &v17_weight,
        Some(&v17_bias),
        &v17_params,
        &flag,
    )
    .expect("v17 conv2d failed");

    assert_eq!(amg_out.shape, v17_out.shape, "shape mismatch");
    assert_close(amg_out.as_cpu_slice(), &v17_out.data, "conv2d_with_bias");
}

#[test]
fn conv2d_with_padding() {
    let input_shape = vec![1, 1, 3, 3];
    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();

    let weight_shape = vec![1, 1, 3, 3];
    let weight_data: Vec<f32> = vec![
        0.1, 0.2, 0.1, //
        0.2, 0.5, 0.2, //
        0.1, 0.2, 0.1,
    ];

    let amg_input = amg_tensor(input_shape.clone(), input_data.clone());
    let amg_weight = amg_tensor(weight_shape.clone(), weight_data.clone());
    let amg_cfg = Conv2DConfig::new((1, 1), (1, 1));
    let amg_out = amg_conv2d(&amg_input, &amg_weight, None, &amg_cfg);

    let v17_input = v17_tensor(input_shape, input_data);
    let v17_weight = v17_tensor(weight_shape, weight_data);
    let v17_params = Conv2DParams {
        stride: (1, 1),
        padding: (1, 1),
    };
    let flag = AbortFlag::new();
    let v17_out = v17_conv2d(&v17_input, &v17_weight, None, &v17_params, &flag)
        .expect("v17 conv2d failed");

    assert_eq!(amg_out.shape, v17_out.shape, "shape mismatch");
    assert_close(amg_out.as_cpu_slice(), &v17_out.data, "conv2d_with_padding");
}

#[test]
fn conv2d_stride_2() {
    let input_shape = vec![1, 1, 5, 5];
    let input_data: Vec<f32> = (1..=25).map(|v| v as f32).collect();

    let weight_shape = vec![1, 1, 3, 3];
    let weight_data: Vec<f32> = vec![0.1; 9];

    let amg_input = amg_tensor(input_shape.clone(), input_data.clone());
    let amg_weight = amg_tensor(weight_shape.clone(), weight_data.clone());
    let amg_cfg = Conv2DConfig::new((2, 2), (0, 0));
    let amg_out = amg_conv2d(&amg_input, &amg_weight, None, &amg_cfg);

    let v17_input = v17_tensor(input_shape, input_data);
    let v17_weight = v17_tensor(weight_shape, weight_data);
    let v17_params = Conv2DParams {
        stride: (2, 2),
        padding: (0, 0),
    };
    let flag = AbortFlag::new();
    let v17_out = v17_conv2d(&v17_input, &v17_weight, None, &v17_params, &flag)
        .expect("v17 conv2d failed");

    assert_eq!(amg_out.shape, v17_out.shape, "shape mismatch");
    assert_close(amg_out.as_cpu_slice(), &v17_out.data, "conv2d_stride_2");
}
