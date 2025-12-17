use atenia_engine::amm::batch_manager::BatchManager;
use atenia_engine::tensor::tensor::{Device, DType, Tensor};

fn sample_tensor(elements: usize) -> Tensor {
    Tensor::new(vec![elements], 1.0, Device::CPU, DType::F32)
}

#[test]
fn returns_zero_when_limit_is_too_small() {
    let manager = BatchManager::new(0, 0);
    let tensor = sample_tensor(4);
    assert_eq!(manager.estimate_max_batch_size(&tensor), 0);
}

#[test]
fn reasonable_limit_produces_positive_batch() {
    let tensor = sample_tensor(10); // 10 * 4 bytes per sample = 40 bytes.
    let per_sample_bytes = tensor.estimated_bytes();
    let limit = per_sample_bytes * 10 + 1_024;
    let manager = BatchManager::new(limit, 512);

    let batch = manager.estimate_max_batch_size(&tensor);
    assert!(batch > 0);
    let available = limit - 512;
    assert!(batch * per_sample_bytes <= available);
}

#[test]
fn safety_margin_reduces_batch_size() {
    let tensor = sample_tensor(8);
    let per_sample = tensor.estimated_bytes();
    let limit = per_sample * 20;

    let manager_high_margin = BatchManager::new(limit, per_sample * 5);
    let manager_low_margin = BatchManager::new(limit, per_sample);

    let batch_high = manager_high_margin.estimate_max_batch_size(&tensor);
    let batch_low = manager_low_margin.estimate_max_batch_size(&tensor);

    assert!(batch_high <= batch_low);
}

#[test]
fn large_limit_allows_large_batch() {
    let tensor = sample_tensor(16);
    let per_sample = tensor.estimated_bytes();
    let limit = per_sample * 1_000;
    let manager = BatchManager::new(limit, per_sample * 10);

    let batch = manager.estimate_max_batch_size(&tensor);
    assert!(batch >= 900); // allow for margin to reduce slightly.
}
