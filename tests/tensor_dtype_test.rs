use atenia_engine::tensor::tensor::{Device, DType, Tensor};

#[test]
fn dtype_size_in_bytes_matches_spec() {
    assert_eq!(DType::F32.size_in_bytes(), 4);
    assert_eq!(DType::F16.size_in_bytes(), 2);
    assert_eq!(DType::BF16.size_in_bytes(), 2);
    assert_eq!(DType::FP8.size_in_bytes(), 1);
}

#[test]
fn tensor_constructors_capture_dtype() {
    let tensor = Tensor::new(vec![2, 2], 0.0, Device::CPU, DType::FP8);
    assert_eq!(tensor.dtype, DType::FP8);

    let ones = Tensor::ones(vec![3], Device::GPU, DType::BF16);
    assert_eq!(ones.dtype, DType::BF16);

    let random = Tensor::random(vec![1], Device::CPU, DType::F16);
    assert_eq!(random.dtype, DType::F16);
}

#[test]
fn cast_changes_dtype_but_not_shape_or_device() {
    let base = Tensor::ones(vec![2, 2], Device::GPU, DType::F32);
    let casted = base.cast_to(DType::BF16);

    assert_eq!(casted.dtype, DType::BF16);
    assert_eq!(casted.shape, base.shape);
    assert_eq!(casted.device, base.device);
    assert_eq!(casted.data, base.data);
}

#[test]
fn estimated_bytes_matches_elements_times_dtype_size() {
    let tensor = Tensor::new(vec![2, 3], 1.0, Device::CPU, DType::F16);
    assert_eq!(tensor.num_elements(), 6);
    assert_eq!(tensor.estimated_bytes(), 6 * DType::F16.size_in_bytes());
}
