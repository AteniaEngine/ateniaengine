use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::apx8::gpu_finalizer::gpu_finalize;
use atenia_engine::apx8::codegen::gpu_codegen_v1::GPUCodegenV1;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_17_structure() {
    let ir = KernelIR::new_mock("matmul_cpu");
    let out = gpu_finalize(&ir);
    assert!(out.contains("CPU fallback"));
}

#[test]
fn apx_8_17_cuda() {
    let ir = KernelIR::new_mock("matmul_cuda");
    assert!(gpu_finalize(&ir).contains("CUDA"));
}

#[test]
fn apx_8_17_hip() {
    let ir = KernelIR::new_mock("vecadd_hip");
    assert!(gpu_finalize(&ir).contains("HIP"));
}

#[test]
fn apx_8_17_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}

#[test]
fn apx_8_17_codegen_pipeline() {
    let ir = KernelIR::new_mock("matmul_cuda");
    let out = GPUCodegenV1::codegen_with_finalizer(&ir);
    assert!(out.contains("FINALIZED CUDA KERNEL"));
}
