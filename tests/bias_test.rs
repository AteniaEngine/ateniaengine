#![allow(dead_code)]

use atenia_engine::v17;

use v17::compute::tensor::Tensor;
use v17::cnn::conv2d::AbortFlag;
use v17::cnn::bias::{add_bias, BiasError};

#[test]
fn bias_add_matches_reference() {
    let input = Tensor::new(vec![1, 3, 2, 2], vec![
        1.0, 2.0, 3.0, 4.0, // C0
        5.0, 6.0, 7.0, 8.0, // C1
        9.0,10.0,11.0,12.0, // C2
    ]).unwrap();

    let bias = Tensor::new(vec![3], vec![0.5, -1.0, 2.0]).unwrap();
    let flag = AbortFlag::new();

    let out = add_bias(&input, &bias, &flag).unwrap();

    // Reference: manually add per-channel bias.
    let mut expected = input.data.clone();
    let n = 1usize;
    let c = 3usize;
    let h = 2usize;
    let w = 2usize;
    for n_idx in 0..n {
        for c_idx in 0..c {
            let b = bias.data[c_idx];
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let idx = (((n_idx * c + c_idx) * h + h_idx) * w) + w_idx;
                    expected[idx] += b;
                }
            }
        }
    }

    assert_eq!(out.shape, input.shape);
    assert_eq!(out.data, expected);
}

#[test]
fn bias_is_deterministic() {
    let input = Tensor::new(vec![1, 2, 2, 2], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]).unwrap();
    let bias = Tensor::new(vec![2], vec![0.5, -0.5]).unwrap();
    let flag = AbortFlag::new();

    let out1 = add_bias(&input, &bias, &flag).unwrap();
    let out2 = add_bias(&input, &bias, &flag).unwrap();

    assert_eq!(out1.shape, out2.shape);
    assert_eq!(out1.data, out2.data);
}

#[test]
fn bias_is_abortable() {
    let input = Tensor::new(vec![1, 1, 4, 4], vec![1.0; 16]).unwrap();
    let bias = Tensor::new(vec![1], vec![1.0]).unwrap();

    let mut flag = AbortFlag::new();
    flag.abort();

    let result = add_bias(&input, &bias, &flag);
    assert!(matches!(result, Err(BiasError::Aborted)));
}

#[test]
fn invalid_bias_shape_yields_error() {
    // Input NCHW [1, 2, 2, 2], but bias has wrong channel count.
    let input = Tensor::new(vec![1, 2, 2, 2], vec![0.0; 8]).unwrap();
    let bad_bias = Tensor::new(vec![3], vec![0.0; 3]).unwrap();
    let flag = AbortFlag::new();

    let result = add_bias(&input, &bad_bias, &flag);
    assert!(matches!(result, Err(BiasError::InvalidBiasShape)));
}
