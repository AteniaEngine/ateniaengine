#![allow(dead_code)]

use atenia_engine::v17;

use v17::compute::tensor::Tensor;
use v17::cnn::conv2d::{Conv2DParams, conv2d_cpu, AbortFlag, ConvError};

fn reference_conv2d_naive(
    input: &Tensor,
    weights: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2DParams,
) -> Tensor {
    let abort = AbortFlag::new();
    conv2d_cpu(input, weights, bias, params, &abort).unwrap()
}

#[test]
fn conv2d_simple_output_matches_reference() {
    let input = Tensor::new(vec![1, 1, 4, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0,10.0,11.0,12.0,
        13.0,14.0,15.0,16.0,
    ]).unwrap();

    let weights = Tensor::new(vec![1, 1, 2, 2], vec![
        1.0, 0.0,
        0.0, -1.0,
    ]).unwrap();

    let bias = Tensor::new(vec![1], vec![0.5]).unwrap();

    let params = Conv2DParams { stride: (1, 1), padding: (0, 0) };
    let abort = AbortFlag::new();

    let y = conv2d_cpu(&input, &weights, Some(&bias), &params, &abort).unwrap();
    let y_ref = reference_conv2d_naive(&input, &weights, Some(&bias), &params);

    assert_eq!(y.shape, y_ref.shape);
    for i in 0..y.data.len() {
        assert!((y.data[i] - y_ref.data[i]).abs() < 1e-6);
    }
}

#[test]
fn conv2d_is_deterministic() {
    let input = Tensor::new(vec![1, 1, 3, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]).unwrap();
    let weights = Tensor::new(vec![1, 1, 2, 2], vec![
        1.0, 0.0,
        0.0, 1.0,
    ]).unwrap();

    let params = Conv2DParams { stride: (1, 1), padding: (0, 0) };
    let abort = AbortFlag::new();

    let y1 = conv2d_cpu(&input, &weights, None, &params, &abort).unwrap();
    let y2 = conv2d_cpu(&input, &weights, None, &params, &abort).unwrap();

    assert_eq!(y1.shape, y2.shape);
    assert_eq!(&y1.data, &y2.data);
}

#[test]
fn conv2d_is_abortable() {
    let input = Tensor::new(vec![1, 1, 8, 8], vec![1.0; 64]).unwrap();
    let weights = Tensor::new(vec![1, 1, 3, 3], vec![1.0; 9]).unwrap();
    let params = Conv2DParams { stride: (1, 1), padding: (0, 0) };

    let mut flag = AbortFlag::new();
    flag.abort();

    let result = conv2d_cpu(&input, &weights, None, &params, &flag);
    assert!(matches!(result, Err(ConvError::Aborted)));
}

#[test]
fn invalid_shapes_yield_explicit_error() {
    // input not NCHW (3D)
    let bad_input = Tensor::new(vec![1, 3, 3], vec![0.0; 9]).unwrap();
    let weights = Tensor::new(vec![1, 1, 2, 2], vec![1.0; 4]).unwrap();
    let params = Conv2DParams { stride: (1, 1), padding: (0, 0) };
    let flag = AbortFlag::new();

    let r = conv2d_cpu(&bad_input, &weights, None, &params, &flag);
    assert!(matches!(r, Err(ConvError::InvalidInputShape)));

    // weights not OIHW (3D)
    let input = Tensor::new(vec![1, 1, 4, 4], vec![0.0; 16]).unwrap();
    let bad_weights = Tensor::new(vec![1, 2, 2], vec![0.0; 4]).unwrap();
    let r2 = conv2d_cpu(&input, &bad_weights, None, &params, &flag);
    assert!(matches!(r2, Err(ConvError::InvalidWeightShape)));

    // stride zero
    let bad_params = Conv2DParams { stride: (0, 1), padding: (0, 0) };
    let r3 = conv2d_cpu(&input, &weights, None, &bad_params, &flag);
    assert!(matches!(r3, Err(ConvError::InvalidStride)));

    // kernel larger than input (after padding)
    let big_kernel = Tensor::new(vec![1, 1, 6, 6], vec![0.0; 36]).unwrap();
    let r4 = conv2d_cpu(&input, &big_kernel, None, &params, &flag);
    assert!(matches!(r4, Err(ConvError::KernelLargerThanInput)));

    // bias with wrong shape
    let bad_bias = Tensor::new(vec![2], vec![0.0; 2]).unwrap();
    let r5 = conv2d_cpu(&input, &weights, Some(&bad_bias), &params, &flag);
    assert!(matches!(r5, Err(ConvError::InvalidBiasShape)));
}
