use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::apx8::dualgraph::DevicePlacement;
use atenia_engine::apx8::gpu_transfer_estimator::GPUTransferEstimator;
use atenia_engine::apx8::hybrid_dispatcher::HybridDispatcher;
use atenia_engine::apx8::hybrid_dispatcher::ExecDevice;

#[test]
fn apx_8_3_estimator_basic_values() {
    let t = Tensor::randn(&[1024, 1024], Device::CPU);
    let est = GPUTransferEstimator::estimate(&t, DevicePlacement::CPU);

    assert!(est.h2d_ms > 0.0);
    assert!(est.d2h_ms > est.h2d_ms * 0.9);
    assert!(est.stay_gpu_ms >= 0.0);
}

#[test]
fn apx_8_3_dispatch_prefers_cpu_for_small() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.3"); }

    let t = Tensor::randn(&[64, 64], Device::CPU);
    let dev = HybridDispatcher::choose_device_for(&t);

    assert_eq!(dev, ExecDevice::CPU);
}

#[test]
fn apx_8_3_dispatch_prefers_gpu_for_large() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.3"); }

    let t = Tensor::randn(&[4096, 4096], Device::CPU);
    let dev = HybridDispatcher::choose_device_for(&t);

    assert_eq!(dev, ExecDevice::GPU);
}
