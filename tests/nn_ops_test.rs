use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::nn::linear::{linear, matmul};
use atenia_engine::nn::activations::silu;
use atenia_engine::nn::normalization::rms_norm;
use atenia_engine::nn::softmax::softmax_last_dim;

fn make_tensor_2d(shape: (usize, usize), fill_fn: impl Fn(usize, usize) -> f32) -> Tensor {
    let (rows, cols) = shape;
    let mut t = Tensor::with_layout(
        vec![rows, cols],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for i in 0..rows {
        for j in 0..cols {
            t.data[i * cols + j] = fill_fn(i, j);
        }
    }
    t
}

#[test]
fn matmul_basic() {
    let a = make_tensor_2d((2, 3), |i, j| (i * 3 + j) as f32 + 1.0);
    let b = make_tensor_2d((3, 2), |i, j| (i * 2 + j) as f32 + 1.0);

    let out = matmul(&a, &b);

    assert_eq!(out.shape, vec![2, 2]);
    assert!((out.data[0] - 22.0).abs() < 1e-5);
}

#[test]
fn linear_without_bias() {
    let x = make_tensor_2d((2, 3), |i, j| (i * 3 + j) as f32);
    let w = make_tensor_2d((3, 4), |i, j| ((i + j) as f32) * 0.5);

    let out = linear(&x, &w, None);

    assert_eq!(out.shape, vec![2, 4]);
}

#[test]
fn linear_with_bias() {
    let x = make_tensor_2d((1, 3), |_, j| j as f32);
    let w = make_tensor_2d((3, 2), |i, j| (i + j) as f32);
    let mut b = Tensor::with_layout(
        vec![2],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    b.data[0] = 1.0;
    b.data[1] = -1.0;

    let out = linear(&x, &w, Some(&b));

    assert_eq!(out.shape, vec![1, 2]);
    assert_ne!(out.data[0], out.data[1]);
}

#[test]
fn rms_norm_keeps_constant_vector_same_scale() {
    let mut t = Tensor::with_layout(
        vec![2, 4],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for v in t.data.iter_mut() {
        *v = 1.0;
    }

    let out = rms_norm(&t, 1e-5);

    assert_eq!(out.shape, t.shape);
    for v in &out.data {
        assert!((*v - 1.0).abs() < 1e-3);
    }
}

#[test]
fn silu_is_reasonable() {
    let mut t = Tensor::with_layout(
        vec![4],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.data = vec![-2.0, -1.0, 0.0, 2.0];

    let out = silu(&t);

    assert_eq!(out.shape, t.shape);
    assert!(out.data[0] < 0.0);
    assert!(out.data[1] < 0.0);
    assert!((out.data[2]).abs() < 1e-6);
    assert!(out.data[3] > 0.0);
    assert!(out.data[3] < t.data[3]);
}

#[test]
fn softmax_rows_sum_to_one() {
    let t = make_tensor_2d((3, 5), |i, j| (i * 5 + j) as f32);

    let out = softmax_last_dim(&t);

    assert_eq!(out.shape, t.shape);

    let rows = 3;
    let cols = 5;

    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        let row = &out.data[start..end];

        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax row {} sums to {}", i, sum);
    }
}
