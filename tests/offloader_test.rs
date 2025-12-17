use atenia_engine::amm::offloading::{Offloader, StorageLocation};
use atenia_engine::tensor::tensor::{Device, DType, Layout, Tensor};
use std::fs;
use uuid::Uuid;

fn make_tensor(values: Vec<f32>, device: Device) -> Tensor {
    let shape = vec![values.len()];
    let layout = Layout::Contiguous;
    let strides = Tensor::compute_strides(&shape, &layout);

    Tensor {
        shape,
        data: values,
        device,
        dtype: DType::F32,
        layout,
        strides,
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    }
}

fn temp_dir() -> String {
    let dir = std::env::temp_dir()
        .join(format!("atenia_offload_test_{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().to_string()
}

#[test]
fn gpu_to_ram_sets_device_cpu() {
    let dir = temp_dir();
    let offloader = Offloader::new(dir);
    let mut tensor = make_tensor(vec![1.0, 2.0], Device::GPU);
    offloader.to_ram(&mut tensor);
    assert_eq!(tensor.device, Device::CPU);
}

#[test]
fn offloading_to_disk_creates_file() {
    let dir = temp_dir();
    let offloader = Offloader::new(dir.clone());
    let tensor = make_tensor(vec![1.0, 2.0, 3.0], Device::CPU);
    let handle = offloader.to_disk(&tensor);
    assert!(handle.path.is_some());
    assert!(matches!(handle.location, StorageLocation::DISK));
    assert!(fs::metadata(handle.path.as_ref().unwrap()).is_ok());
}

#[test]
fn loading_from_disk_restores_data() {
    let dir = temp_dir();
    let offloader = Offloader::new(dir);
    let tensor = make_tensor(vec![3.14, 2.71], Device::CPU);
    let handle = offloader.to_disk(&tensor);
    let restored = offloader.from_disk(&handle, tensor.shape.clone());
    assert_eq!(restored.data, tensor.data);
    assert_eq!(restored.shape, tensor.shape);
    assert_eq!(restored.device, Device::CPU);
}

#[test]
fn directory_auto_created() {
    let dir = std::env::temp_dir()
        .join(format!("atenia_offload_test_auto_{}", Uuid::new_v4()));
    let dir_str = dir.to_string_lossy().to_string();
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    let _offloader = Offloader::new(dir_str.clone());
    assert!(std::path::Path::new(&dir_str).exists());
}
