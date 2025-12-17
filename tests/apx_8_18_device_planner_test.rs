use atenia_engine::apx8::device_planner::*;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_18_detect_devices() {
    let gpus = detect_simulated_gpus();
    assert!(gpus.len() >= 2);
}

#[test]
fn apx_8_18_plans_cuda() {
    let p = plan_for_ir("matmul_cuda");
    assert!(p.target_gpu.is_some());
    assert!(p.target_gpu.unwrap().name.contains("CUDA"));
}

#[test]
fn apx_8_18_plans_hip() {
    let p = plan_for_ir("vecadd_hip");
    assert!(p.target_gpu.is_some());
    assert!(p.target_gpu.unwrap().name.contains("AMD"));
}

#[test]
fn apx_8_18_plans_default() {
    let p = plan_for_ir("softmax_cpu");
    assert!(p.target_gpu.is_none());
}

#[test]
fn apx_8_18_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}
