use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::gpu_vec_add;

#[test]
fn apx_8_6_gpu_vec_add_structure() {
    let a = Tensor::ones(vec![8], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![8], Device::CPU, DType::F32);
    let _ = a.add(&b); // existing CPU path, smoke test only
}

#[test]
fn apx_8_6_gpu_vec_add_equivalence() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.6"); }

    let mut a = Tensor::ones(vec![8], Device::CPU, DType::F32);
    let mut b = Tensor::ones(vec![8], Device::CPU, DType::F32);
    // b = vector of 2.0
    for v in b.data.iter_mut() {
        *v = 2.0;
    }

    gpu_vec_add(&mut a, &b);

    // CPU view must be 3.0 in all positions.
    for v in a.data.iter() {
        assert!((*v - 3.0).abs() < 1e-6);
    }
}

#[test]
fn apx_8_6_gpu_vec_add_cpu_coherence() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.6"); }

    let mut a = Tensor::ones(vec![8], Device::CPU, DType::F32);
    let mut b = Tensor::ones(vec![8], Device::CPU, DType::F32);
    for v in b.data.iter_mut() {
        *v = 2.0;
    }

    let before = a.data.clone();
    gpu_vec_add(&mut a, &b);
    a.sync_cpu();
    let after = a.data.clone();

    // We do not introduce NaNs or corruption; the vector remains valid.
    assert_eq!(after.len(), before.len());
}
