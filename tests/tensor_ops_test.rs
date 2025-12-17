use atenia_engine::tensor::tensor::{Device, DType, Layout, Tensor};

fn sample_tensor(values: &[f32], shape: Vec<usize>, device: Device, dtype: DType) -> Tensor {
    let layout = Layout::Contiguous;
    let strides = Tensor::compute_strides(&shape, &layout);
    Tensor {
        shape,
        data: values.to_vec(),
        device,
        dtype,
        layout,
        strides,
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    }
}

#[test]
fn add_two_tensors() {
    let a = Tensor {
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
        device: Device::CPU,
        dtype: DType::F32,
        layout: Layout::Contiguous,
        strides: vec![1],
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    };
    let b = Tensor {
        shape: vec![3],
        data: vec![4.0, 5.0, 6.0],
        device: Device::CPU,
        dtype: DType::F32,
        layout: Layout::Contiguous,
        strides: vec![1],
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    };

    let result = a.add(&b);
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    assert_eq!(result.device, Device::CPU);
}

#[test]
fn mul_two_tensors() {
    let a = sample_tensor(&[2.0, 3.0], vec![2], Device::GPU, DType::F32);
    let b = sample_tensor(&[4.0, 5.0], vec![2], Device::GPU, DType::F32);

    let result = a.mul(&b);
    assert_eq!(result.data, vec![8.0, 15.0]);
    assert_eq!(result.device, Device::GPU);
}

#[test]
fn constructors_work() {
    let zeros = Tensor::zeros(vec![2, 2], Device::CPU, DType::F32);
    assert_eq!(zeros.data, vec![0.0; 4]);

    let ones = Tensor::ones(vec![3], Device::GPU, DType::F32);
    assert_eq!(ones.data, vec![1.0; 3]);

    let random = Tensor::random(vec![2], Device::CPU, DType::F32);
    assert_eq!(random.shape, vec![2]);
    assert_eq!(random.data.len(), 2);
    assert!(random.data.iter().all(|v| v.is_finite()));
}
