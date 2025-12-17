use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::apx8::hxo::{build_hxo_plan, hybrid_dispatch, HybridDispatchResult};
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_20_hxo_structure() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.20"); }

    let ir = KernelIR::new_mock("my_add_cuda");
    let shape = vec![1024, 1024];
    let p = build_hxo_plan(&ir, &shape);

    assert!(!p.device.is_empty());
    assert!(!p.backend.is_empty());
    assert!(!p.partition.is_empty());
    assert!(!p.codegen.is_empty());
}

#[test]
fn apx_8_20_hxo_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}

#[test]
fn apx_8_20_hxo_logic_combined() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.20"); }

    let ir = KernelIR::new_mock("matmul_cuda");
    let shape = vec![2048, 2048];
    let p = build_hxo_plan(&ir, &shape);

    assert!(p.partition.contains("Split2D") || p.partition.contains("Split1D"));
}

#[test]
fn apx_8_20_dispatcher_integration() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.20"); }

    let ir = KernelIR::new_mock("add_cuda");
    let shape = vec![8];
    let r = hybrid_dispatch(&ir, &shape);
    match r {
        HybridDispatchResult::Pseudo { device, backend, .. } => {
            assert!(!device.is_empty());
            assert!(!backend.is_empty());
        }
    }
}
