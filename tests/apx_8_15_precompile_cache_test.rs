use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::apx8::precompile_cache::PrecompileCache;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_15_basic_structure() {
    let cache = PrecompileCache::new();
    assert_eq!(cache.len(), 0);
}

#[test]
fn apx_8_15_compile_once() {
    let mut cache = PrecompileCache::new();
    let ir = KernelIR::new_mock("matmul");

    let c1 = cache.compile_if_missing(&ir);
    let c2 = cache.compile_if_missing(&ir);

    assert_eq!(c1, c2);
    assert!(cache.contains(&ir));
    assert_eq!(cache.len(), 1);
}

#[test]
fn apx_8_15_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}
