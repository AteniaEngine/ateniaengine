use atenia_engine::apx8::gpu_kernel_signature::*;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_8_register_and_get() {
    let sig = GpuKernelSignature {
        key: GpuKernelType::VecAdd,
        min_dims: (1, 1, 1),
        max_dims: (1024, 1, 1),
        workspace_bytes: 0,
        launcher_name: "vec_add",
    };

    register_signature(sig.clone());

    let retrieved = get_signature(&GpuKernelType::VecAdd).unwrap();
    assert_eq!(retrieved.launcher_name, "vec_add");
}

#[test]
fn apx_8_8_dispatcher_checks_signature_stub() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.8"); }

    // Llamamos directamente a get_signature como si el dispatcher lo hubiera consultado.
    let _ = get_signature(&GpuKernelType::VecAdd);
}

#[test]
fn apx_8_8_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let res = a.add(&b);

    for v in res.data.iter() {
        assert!((*v - 2.0).abs() < 1e-6);
    }
}
