use atenia_engine::tensor::tensor::{Device, DType, Tensor};

#[test]
fn creates_cpu_tensor() {
    let tensor = Tensor::new(vec![2, 3], 1.0, Device::CPU, DType::F32);
    assert_eq!(tensor.device, Device::CPU);
    assert_eq!(tensor.num_elements(), 6);
    assert!(tensor.data.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
}

#[test]
fn moves_to_gpu() {
    let tensor = Tensor::new(vec![4], 0.0, Device::CPU, DType::F32);
    let gpu_tensor = tensor.to_gpu();
    assert_eq!(gpu_tensor.device, Device::GPU);
    assert_eq!(gpu_tensor.data, tensor.data);
}

#[test]
fn counts_elements_correctly() {
    let tensor = Tensor::new(vec![2, 2, 3], 0.5, Device::GPU, DType::F32);
    assert_eq!(tensor.num_elements(), 12);
}
