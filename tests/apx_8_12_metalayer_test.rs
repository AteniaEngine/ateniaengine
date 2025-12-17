use atenia_engine::apx8::gpu_metalayer::*;
use atenia_engine::apx8::kernel_generator::*;
use atenia_engine::apx8::kernel_generator::KernelOp;

#[test]
fn apx_8_12_structure() {
    let ir = KernelIR {
        ops: vec![
            KernelOp::LoadTensor("A".into()),
            KernelOp::Compute("Add".into()),
            KernelOp::StoreTensor("A".into()),
        ],
        name: "apx_8_12_structure".into(),
        params: vec![],
    };

    let opt = optimize_ir(&ir);

    assert_eq!(opt.ops.len(), 3);
    assert!(opt.meta.contains_key("fusion"));
    assert!(opt.meta.contains_key("tiling"));
    assert!(opt.meta.contains_key("vectorization"));
}

#[test]
fn apx_8_12_removes_nops() {
    let ir = KernelIR {
        ops: vec![
            KernelOp::Nop,
            KernelOp::LoadTensor("A".into()),
        ],
        name: "apx_8_12_removes_nops".into(),
        params: vec![],
    };

    let opt = optimize_ir(&ir);
    assert_eq!(opt.ops.len(), 1);
}

#[test]
fn apx_8_12_no_math_change() {
    let ir = KernelIR {
        ops: vec![
            KernelOp::LoadTensor("X".into()),
            KernelOp::Compute("Mul".into()),
            KernelOp::StoreTensor("Y".into()),
        ],
        name: "apx_8_12_no_math_change".into(),
        params: vec![],
    };

    let opt = optimize_ir(&ir);

    for (a, b) in ir.ops.iter().zip(opt.ops.iter()) {
        match (a, b) {
            (KernelOp::LoadTensor(x), KernelOp::LoadTensor(y)) => assert_eq!(x, y),
            (KernelOp::Compute(x), KernelOp::Compute(y)) => assert_eq!(x, y),
            (KernelOp::StoreTensor(x), KernelOp::StoreTensor(y)) => assert_eq!(x, y),
            _ => panic!("IR mismatch"),
        }
    }
}
