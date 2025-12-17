use atenia_engine::apx9::gpu_codegen_real::*;
use atenia_engine::apx8::kernel_registry::register_real_kernel;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_9_10_ptx_structure() {
    let k = RealKernelBuilder::new_ptx("vec_add", &["a","b","out"]);
    assert!(k.code.contains(".entry vec_add"));
    assert!(k.code.contains(".version"));
    assert!(k.code.contains("ret;"));
}

#[test]
fn apx_9_10_cl_structure() {
    let k = RealKernelBuilder::new_cl("vec_add", &["a","b","out"]);
    assert!(k.code.contains("__kernel void vec_add"));
}

#[test]
fn apx_9_10_register_kernel() {
    let k = RealKernelBuilder::new_ptx("vec_add", &["a","b","out"]);
    register_real_kernel(k);
}

#[test]
fn apx_9_10_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);

    for v in c.data {
        assert!((v - 2.0).abs() < 1e-6);
    }
}
