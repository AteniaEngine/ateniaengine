#![allow(dead_code)]

use atenia_engine::v17;

use v17::compute::tensor::Tensor;
use v17::cnn::conv2d::AbortFlag;
use v17::cnn::maxpool2d::{MaxPool2DParams, maxpool2d_cpu, MaxPoolError};

#[test]
fn maxpool2d_matches_reference() {
    // Simple 1x1x4x4 input, 2x2 kernel, stride 2, no padding.
    let input = Tensor::new(vec![1, 1, 4, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0,10.0,11.0,12.0,
        13.0,14.0,15.0,16.0,
    ]).unwrap();

    let params = MaxPool2DParams {
        kernel: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };

    let flag = AbortFlag::new();

    let out = maxpool2d_cpu(&input, &params, &flag).unwrap();

    // Reference result:
    // [[6, 8],
    //  [14,16]]
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_eq!(out.data, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn maxpool2d_is_deterministic() {
    let input = Tensor::new(vec![1, 1, 3, 3], vec![
        1.0, 3.0, 2.0,
        4.0, 6.0, 5.0,
        7.0, 9.0, 8.0,
    ]).unwrap();

    let params = MaxPool2DParams {
        kernel: (2, 2),
        stride: (1, 1),
        padding: (0, 0),
    };

    let flag = AbortFlag::new();

    let out1 = maxpool2d_cpu(&input, &params, &flag).unwrap();
    let out2 = maxpool2d_cpu(&input, &params, &flag).unwrap();

    assert_eq!(out1.shape, out2.shape);
    assert_eq!(out1.data, out2.data);
}

#[test]
fn maxpool2d_is_abortable() {
    let input = Tensor::new(vec![1, 1, 4, 4], vec![1.0; 16]).unwrap();

    let params = MaxPool2DParams {
        kernel: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };

    let mut flag = AbortFlag::new();
    flag.abort();

    let result = maxpool2d_cpu(&input, &params, &flag);
    assert!(matches!(result, Err(MaxPoolError::Aborted)));
}

#[test]
fn invalid_params_yield_explicit_error() {
    // Input not NCHW (3D)
    let bad_input = Tensor::new(vec![1, 3, 3], vec![0.0; 9]).unwrap();
    let params = MaxPool2DParams {
        kernel: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };
    let flag = AbortFlag::new();

    let r = maxpool2d_cpu(&bad_input, &params, &flag);
    assert!(matches!(r, Err(MaxPoolError::InvalidInputShape)));

    // Zero kernel size
    let input = Tensor::new(vec![1, 1, 4, 4], vec![0.0; 16]).unwrap();
    let bad_kernel = MaxPool2DParams {
        kernel: (0, 2),
        stride: (2, 2),
        padding: (0, 0),
    };
    let r2 = maxpool2d_cpu(&input, &bad_kernel, &flag);
    assert!(matches!(r2, Err(MaxPoolError::InvalidKernel)));

    // Zero stride
    let bad_stride = MaxPool2DParams {
        kernel: (2, 2),
        stride: (0, 2),
        padding: (0, 0),
    };
    let r3 = maxpool2d_cpu(&input, &bad_stride, &flag);
    assert!(matches!(r3, Err(MaxPoolError::InvalidStride)));

    // Kernel larger than padded input
    let bad_params = MaxPool2DParams {
        kernel: (6, 6),
        stride: (1, 1),
        padding: (0, 0),
    };
    let r4 = maxpool2d_cpu(&input, &bad_params, &flag);
    assert!(matches!(r4, Err(MaxPoolError::KernelLargerThanInput)));
}
