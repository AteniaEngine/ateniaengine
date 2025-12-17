use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::apx8::multiarch_router::{route_kernel, TargetArch};
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_16_structure() {
    let ir = KernelIR::new_mock("matmul");
    let arch = route_kernel(&ir);
    assert_eq!(arch, TargetArch::CPU);
}

#[test]
fn apx_8_16_routes_cuda() {
    let ir = KernelIR::new_mock("matmul_cuda");
    assert_eq!(route_kernel(&ir), TargetArch::CUDA);
}

#[test]
fn apx_8_16_routes_hip() {
    let ir = KernelIR::new_mock("vecadd_hip");
    assert_eq!(route_kernel(&ir), TargetArch::HIP);
}

#[test]
fn apx_8_16_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}
