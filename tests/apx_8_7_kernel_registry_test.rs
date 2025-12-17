use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::apx8::kernel_registry::*;
use atenia_engine::gpu_vec_add;

#[test]
fn apx_8_7_register_and_retrieve() {
    KERNEL_REGISTRY.register(KernelKey::VecAdd, gpu_vec_add);
    let k = KERNEL_REGISTRY.get(&KernelKey::VecAdd);
    assert!(k.is_some());
}

#[test]
fn apx_8_7_dispatcher_uses_registered_kernel_stub() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.7"); }

    // En esta fase no integramos con HybridDispatcher real; usamos el registry directamente
    let mut a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let mut b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    for v in b.data.iter_mut() { *v = 1.0; }

    KERNEL_REGISTRY.register(KernelKey::VecAdd, gpu_vec_add);
    let kernel = KERNEL_REGISTRY.get(&KernelKey::VecAdd).unwrap();
    kernel(&mut a, &b);

    for v in a.data.iter() {
        assert!((*v - 2.0).abs() < 1e-6);
    }
}

#[test]
fn apx_8_7_equivalence_cpu_vs_registry_gpu_stub() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let cpu_res = a.add(&b);

    let mut gpu_like = Tensor::ones(vec![4], Device::CPU, DType::F32);
    KERNEL_REGISTRY.register(KernelKey::VecAdd, gpu_vec_add);
    let kernel = KERNEL_REGISTRY.get(&KernelKey::VecAdd).unwrap();
    kernel(&mut gpu_like, &b);

    for (x, y) in cpu_res.data.iter().zip(gpu_like.data.iter()) {
        assert!((*x - *y).abs() < 1e-6);
    }
}
