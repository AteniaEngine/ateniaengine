use atenia_engine::tensor::tensor::{DType, Device, Layout, Tensor};

fn sample_tensor(values: &[f32], shape: Vec<usize>, device: Device, dtype: DType) -> Tensor {
    Tensor::new_cpu_with_layout(shape, values.to_vec(), device, dtype, Layout::Contiguous)
}

#[test]
fn add_two_tensors() {
    let a = Tensor::new_cpu_with_layout(
        vec![3],
        vec![1.0, 2.0, 3.0],
        Device::CPU,
        DType::F32,
        Layout::Contiguous,
    );
    let b = Tensor::new_cpu_with_layout(
        vec![3],
        vec![4.0, 5.0, 6.0],
        Device::CPU,
        DType::F32,
        Layout::Contiguous,
    );

    let result = a.add(&b);
    assert_eq!(result.copy_to_cpu_vec(), vec![5.0, 7.0, 9.0]);
    assert_eq!(result.device, Device::CPU);
}

#[test]
fn mul_two_tensors() {
    let a = sample_tensor(&[2.0, 3.0], vec![2], Device::GPU, DType::F32);
    let b = sample_tensor(&[4.0, 5.0], vec![2], Device::GPU, DType::F32);

    let result = a.mul(&b);
    assert_eq!(result.copy_to_cpu_vec(), vec![8.0, 15.0]);
    assert_eq!(result.device, Device::GPU);
}

#[test]
fn constructors_work() {
    let zeros = Tensor::zeros(vec![2, 2], Device::CPU, DType::F32);
    assert_eq!(zeros.copy_to_cpu_vec(), vec![0.0; 4]);

    let ones = Tensor::ones(vec![3], Device::GPU, DType::F32);
    assert_eq!(ones.copy_to_cpu_vec(), vec![1.0; 3]);

    let random = Tensor::random(vec![2], Device::CPU, DType::F32);
    assert_eq!(random.shape, vec![2]);
    assert_eq!(random.numel(), 2);
    assert!(random.as_cpu_slice().iter().all(|v| v.is_finite()));
}
