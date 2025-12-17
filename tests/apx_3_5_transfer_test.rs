use atenia_engine::tensor::{Device, Tensor};
use atenia_engine::apx3_5::memory_manager::MemoryManager;

#[test]
fn apx_3_5_cpu_to_cpu_roundtrip_is_identical() {
    let t = Tensor::randn(&[4, 4], Device::CPU);
    let moved = MemoryManager::move_tensor(&t, Device::CPU);

    assert_eq!(t.shape, moved.shape);
    assert_eq!(t.strides, moved.strides);
    assert_eq!(t.data.len(), moved.data.len());

    for (a, b) in t.data.iter().zip(&moved.data) {
        let diff = (a - b).abs();
        assert!(diff < 1e-6, "values differ: a={} b={} diff={}", a, b, diff);
    }
}

#[test]
fn apx_3_5_gpu_transfer_placeholder() {
    let t = Tensor::randn(&[2, 2], Device::CPU);
    let moved = MemoryManager::move_tensor(&t, Device::GPU);
    assert_eq!(moved.device, Device::GPU);
}
