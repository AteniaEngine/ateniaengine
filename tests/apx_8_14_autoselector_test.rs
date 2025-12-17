use atenia_engine::apx8::gpu_autoselector::GPUAutoSelector;
use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_14_selector_structure() {
    let sel = GPUAutoSelector::detect();
    assert!(sel.vendors.len() >= 3);
}

#[test]
fn apx_8_14_selects_based_on_ir_name() {
    let sel = GPUAutoSelector::detect();

    let ir_vec = KernelIR::new_mock("vecadd");
    assert_eq!(sel.choose_backend(&ir_vec), "hip");

    let ir_mm = KernelIR::new_mock("matmul");
    assert_eq!(sel.choose_backend(&ir_mm), "cuda");
}

#[test]
fn apx_8_14_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}
