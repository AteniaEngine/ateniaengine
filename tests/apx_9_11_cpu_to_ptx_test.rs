use atenia_engine::apx9::cpu_to_ptx::*;
use atenia_engine::apx8::kernel_generator::{KernelIR, KernelOp};

#[test]
fn apx_9_11_structure() {
    let ir = KernelIR {
        ops: vec![
            KernelOp::LoadTensor("A".into()),
            KernelOp::Compute("Add".into()),
            KernelOp::StoreTensor("A".into()),
        ],
        name: "vec_add".to_string(),
        params: vec!["a".into(), "b".into(), "out".into()],
    };

    let r = CPUToPTX::translate(&ir);

    assert!(r.ptx.contains(".entry vec_add"));
    assert!(r.ptx.contains("ld.global.f32"));
    assert!(r.ptx.contains("st.global.f32"));
}

#[test]
fn apx_9_11_thread_indexing() {
    let ir = KernelIR {
        ops: vec![KernelOp::Compute("Add".into())],
        name: "k".to_string(),
        params: vec!["x".into()],
    };

    let r = CPUToPTX::translate(&ir);
    assert!(r.ptx.contains("mov.u32 %r0, %tid.x"));
    assert!(r.ptx.contains("mad.lo.s32"));
}

#[test]
fn apx_9_11_no_numeric_change() {
    use atenia_engine::tensor::{Tensor, Device, DType};

    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);

    for v in c.data {
        assert!((v - 2.0).abs() < 1e-6);
    }
}
