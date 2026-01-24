#![allow(dead_code)]

use atenia_engine::v17;

use v17::compute::tensor::Tensor;
use v17::cnn::conv2d::AbortFlag;
use v17::cnn::activation::{relu, ActivationError};

#[test]
fn relu_zeroes_negative_values() {
    let input = Tensor::new(vec![1, 1, 2, 3], vec![
        -1.0, 0.0, 1.0,
        -2.0, 2.0, 3.0,
    ]).unwrap();
    let flag = AbortFlag::new();

    let out = relu(&input, &flag).unwrap();

    assert_eq!(out.shape, input.shape);
    assert_eq!(out.data, vec![
        0.0, 0.0, 1.0,
        0.0, 2.0, 3.0,
    ]);
}

#[test]
fn relu_is_deterministic() {
    let input = Tensor::new(vec![1, 1, 2, 2], vec![
        -1.0, 1.0,
        -2.0, 2.0,
    ]).unwrap();
    let flag = AbortFlag::new();

    let out1 = relu(&input, &flag).unwrap();
    let out2 = relu(&input, &flag).unwrap();

    assert_eq!(out1.shape, out2.shape);
    assert_eq!(out1.data, out2.data);
}

#[test]
fn relu_is_abortable() {
    let input = Tensor::new(vec![1, 1, 4, 4], vec![
        -1.0, -2.0, 3.0, 4.0,
        5.0, -6.0, 7.0, 8.0,
        9.0, 10.0, -11.0, 12.0,
        -13.0, 14.0, 15.0, -16.0,
    ]).unwrap();

    let mut flag = AbortFlag::new();
    flag.abort();

    let result = relu(&input, &flag);
    assert!(matches!(result, Err(ActivationError::Aborted)));
}

#[test]
fn relu_has_no_side_effects() {
    let input = Tensor::new(vec![1, 1, 2, 2], vec![
        -1.0, 1.0,
        -2.0, 2.0,
    ]).unwrap();
    let flag = AbortFlag::new();

    let before = input.clone();
    let _ = relu(&input, &flag).unwrap();

    // Input tensor must not be modified.
    assert_eq!(input, before);
}
