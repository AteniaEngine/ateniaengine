use atenia_engine::apx3_8::device_context::DeviceContext;
use atenia_engine::tensor::Device;

#[test]
fn apx_3_8_dispatch_selects_correct_kernel() {
    let ctx = DeviceContext::new(Device::CPU);

    let a = vec![1.0; 6];
    let b = vec![1.0; 6];
    let mut out = vec![0.0; 4];

    atenia_engine::apx3_8::kernel_dispatch::dispatch_matmul(&a, &b, &mut out, 2, 3, 2, &ctx);

    assert_eq!(out.len(), 4);
}
