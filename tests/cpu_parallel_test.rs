use atenia_engine::nn::linear::linear;
use atenia_engine::nn::softmax::softmax_last_dim;
use atenia_engine::tensor::ops::batch_matmul::batch_matmul_parallel;
use atenia_engine::tensor::ops::matmul_cpu::matmul_parallel;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn tensor_from_fn(shape: (usize, usize), f: impl Fn(usize, usize) -> f32) -> Tensor {
    let (rows, cols) = shape;
    let mut t = Tensor::with_layout(vec![rows, cols], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    {
        let slice = t.as_cpu_slice_mut();
        for r in 0..rows {
            for c in 0..cols {
                slice[r * cols + c] = f(r, c);
            }
        }
    }
    t
}

fn serial_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    let mut out = Tensor::with_layout(vec![m, n], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    let a_slice = a.as_cpu_slice();
    let b_slice = b.as_cpu_slice();
    let out_slice = out.as_cpu_slice_mut();
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a_slice[i * k + kk] * b_slice[kk * n + j];
            }
            out_slice[i * n + j] = sum;
        }
    }
    out
}

fn serial_softmax(x: &Tensor) -> Tensor {
    let ndim = x.shape.len();
    let cols = x.shape[ndim - 1];
    let rows = if ndim == 1 { 1 } else { x.shape[..ndim - 1].iter().product() };
    let mut out = Tensor::with_layout(x.shape.clone(), 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    let x_slice = x.as_cpu_slice();
    let out_slice = out.as_cpu_slice_mut();
    for row in 0..rows {
        let start = row * cols;
        let slice = &x_slice[start..start + cols];
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            max_val = max_val.max(v);
        }
        let mut sum_exp = 0.0f32;
        let mut temp = Vec::with_capacity(cols);
        for &v in slice {
            let e = (v - max_val).exp();
            temp.push(e);
            sum_exp += e;
        }
        let inv = 1.0f32 / sum_exp.max(1e-12);
        for i in 0..cols {
            out_slice[start + i] = temp[i] * inv;
        }
    }
    out
}

#[test]
fn test_parallel_matmul_matches_serial() {
    let a = tensor_from_fn((4, 6), |i, j| (i * 6 + j) as f32);
    let b = tensor_from_fn((6, 5), |i, j| ((i + j) as f32) * 0.25);

    let parallel = matmul_parallel(&a, &b);
    let serial = serial_matmul(&a, &b);

    assert_eq!(parallel.shape, serial.shape);
    for (p, s) in parallel.as_cpu_slice().iter().zip(serial.as_cpu_slice().iter()) {
        assert!((p - s).abs() < 1e-5);
    }
}

#[test]
fn test_parallel_softmax_identical() {
    let x = tensor_from_fn((3, 7), |i, j| (i * 7 + j) as f32 - 3.0);

    let parallel = softmax_last_dim(&x);
    let serial = serial_softmax(&x);

    for (p, s) in parallel.as_cpu_slice().iter().zip(serial.as_cpu_slice().iter()) {
        assert!((p - s).abs() < 1e-6);
    }
}

#[test]
fn test_parallel_linear_matches_serial() {
    let x = tensor_from_fn((5, 8), |i, j| ((i + j) as f32) * 0.1);
    let w = tensor_from_fn((8, 4), |i, j| (i * 4 + j) as f32 * 0.05);
    let bias = Tensor::with_layout(vec![4], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    let mut bias = bias;
    {
        let b_slice = bias.as_cpu_slice_mut();
        for i in 0..4 {
            b_slice[i] = i as f32 * 0.01;
        }
    }

    let parallel = linear(&x, &w, Some(&bias));
    let serial_mat = serial_matmul(&x, &w);
    let mut serial = serial_mat.clone();
    let shape = serial.shape.clone();
    {
        let serial_slice = serial.as_cpu_slice_mut();
        let bias_slice = bias.as_cpu_slice();
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                serial_slice[row * shape[1] + col] += bias_slice[col];
            }
        }
    }

    for (p, s) in parallel.as_cpu_slice().iter().zip(serial.as_cpu_slice().iter()) {
        assert!((p - s).abs() < 1e-5);
    }
}

#[test]
fn test_batch_matmul_3d() {
    let mut a = Tensor::with_layout(vec![3, 2, 4], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    let mut b = Tensor::with_layout(vec![3, 4, 3], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    for (idx, v) in a.as_cpu_slice_mut().iter_mut().enumerate() {
        *v = (idx as f32) * 0.01;
    }
    for (idx, v) in b.as_cpu_slice_mut().iter_mut().enumerate() {
        *v = ((idx as f32) * 0.02) - 1.0;
    }

    let parallel = batch_matmul_parallel(&a, &b);

    // serial baseline
    let mut serial = Tensor::with_layout(vec![3, 2, 3], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    {
        let a_slice = a.as_cpu_slice();
        let b_slice = b.as_cpu_slice();
        let serial_slice = serial.as_cpu_slice_mut();
        for batch in 0..3 {
            for i in 0..2 {
                for j in 0..3 {
                    let mut sum = 0.0f32;
                    for kk in 0..4 {
                        let a_idx = batch * 2 * 4 + i * 4 + kk;
                        let b_idx = batch * 4 * 3 + kk * 3 + j;
                        sum += a_slice[a_idx] * b_slice[b_idx];
                    }
                    serial_slice[batch * 2 * 3 + i * 3 + j] = sum;
                }
            }
        }
    }

    assert_eq!(parallel.shape, serial.shape);
    for (p, s) in parallel.as_cpu_slice().iter().zip(serial.as_cpu_slice().iter()) {
        assert!((p - s).abs() < 1e-5);
    }
}
