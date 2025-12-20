use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::apx8::codegen::gpu_codegen_v1::GPUCodegenV1;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_13_codegen_produces_strings() {
    let cg = GPUCodegenV1 { target: "cuda".into() };
    let ir = KernelIR::new_mock("vecadd");
    let code = cg.generate_kernel(&ir);
    assert!(code.contains("vecadd"));
}

#[test]
fn apx_8_13_codegen_does_not_change_numerics() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let res_before = a.add(&b);

    // Generate synthetic GPU code from a mock IR.
    let cg = GPUCodegenV1 { target: "cuda".into() };
    let ir = KernelIR::new_mock("vecadd");
    let _code = cg.generate_kernel(&ir);

    let res_after = a.add(&b);

    for (x, y) in res_before.data.iter().zip(res_after.data.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn apx_8_13_targets_independent() {
    let ir = KernelIR::new_mock("vecadd");

    let cuda = GPUCodegenV1 { target: "cuda".into() };
    let hip = GPUCodegenV1 { target: "hip".into() };
    let metal = GPUCodegenV1 { target: "metal".into() };

    let c_cuda = cuda.generate_kernel(&ir);
    let c_hip = hip.generate_kernel(&ir);
    let c_metal = metal.generate_kernel(&ir);

    assert!(c_cuda.contains("synthetic cuda"));
    assert!(c_hip.contains("synthetic hip"));
    assert!(c_metal.contains("synthetic metal"));

    assert_ne!(c_cuda, c_hip);
    assert_ne!(c_cuda, c_metal);
    assert_ne!(c_hip, c_metal);
}
