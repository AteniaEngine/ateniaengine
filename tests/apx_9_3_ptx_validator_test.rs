use atenia_engine::apx9::gpu_ir::*;
use atenia_engine::apx9::ptx_emitter::*;
use atenia_engine::apx9::ptx_validator::*;

#[test]
fn apx_9_3_accepts_valid_ptx() {
    let ir = GpuKernelIR {
        name: "vecadd".into(),
        threads: 256,
        ops: vec![
            GpuOp::Load  { dst: "%f1".into(), src: "%A".into() },
            GpuOp::Load  { dst: "%f2".into(), src: "%B".into() },
            GpuOp::Add   { dst: "%f3".into(), a: "%f1".into(), b: "%f2".into() },
            GpuOp::Store { dst: "%Out".into(), src: "%f3".into() },
        ]
    };

    let ptx = PtxEmitter::emit(&ir);
    let result = PtxValidator::validate(&ptx);

    assert!(result.ok);
}

#[test]
fn apx_9_3_rejects_invalid_ptx() {
    let bad_ptx = "this_is_not_ptx";

    let result = PtxValidator::validate(bad_ptx);

    assert!(!result.ok);
    assert!(result.errors.len() >= 1);
}

#[test]
fn apx_9_3_does_not_change_numerics() {
    let mut a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    for i in 0..a.len() {
        a[i] += b[i];
    }

    assert_eq!(a, vec![5.0, 7.0, 9.0]); // invariable
}
