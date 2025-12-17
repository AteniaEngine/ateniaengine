use atenia_engine::amm::forecaster::MemoryForecaster;
use atenia_engine::tensor::tensor::{Device, DType, Tensor};

fn sample_tensor(num_elements: usize, dtype: DType) -> Tensor {
    Tensor::new(vec![num_elements], 1.0, Device::CPU, dtype)
}

#[test]
fn register_tensor_accumulates_bytes() {
    let mut forecaster = MemoryForecaster::new();
    let t1 = sample_tensor(4, DType::F32);
    let t2 = sample_tensor(2, DType::F16);

    forecaster.register_tensor(&t1);
    forecaster.register_tensor(&t2);

    assert_eq!(forecaster.current_bytes, t1.estimated_bytes() + t2.estimated_bytes());
    assert_eq!(forecaster.predicted_next_bytes, 0);
}

#[test]
fn predict_add_operation_sets_prediction() {
    let mut forecaster = MemoryForecaster::new();
    let a = sample_tensor(3, DType::F32);
    let b = sample_tensor(3, DType::F32);

    forecaster.predict_add_operation(&a, &b);

    let expected = a.estimated_bytes() + b.estimated_bytes() + a.estimated_bytes();
    assert_eq!(forecaster.predicted_next_bytes, expected);
}

#[test]
fn limit_check_behaves_as_expected() {
    let mut forecaster = MemoryForecaster::new();
    let a = sample_tensor(2, DType::F32);
    let b = sample_tensor(2, DType::F32);
    forecaster.predict_add_operation(&a, &b);

    let limit_high = forecaster.predicted_next_bytes + 1024;
    let limit_low = forecaster.predicted_next_bytes - 1;

    assert!(!forecaster.is_over_limit(limit_high));
    assert!(forecaster.is_over_limit(limit_low));
}

#[test]
fn new_forecaster_starts_with_zeroed_state() {
    let forecaster = MemoryForecaster::new();
    assert_eq!(forecaster.current_bytes, 0);
    assert_eq!(forecaster.predicted_next_bytes, 0);
}
