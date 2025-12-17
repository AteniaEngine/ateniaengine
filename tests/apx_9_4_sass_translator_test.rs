use atenia_engine::apx9::gpu_ir::*;
use atenia_engine::apx9::ptx_emitter::*;
use atenia_engine::apx9::sass_translator::*;

#[test]
fn apx_9_4_translates_valid_ptx() {
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
    let sass = SassTranslator::translate(&ptx);

    assert!(sass.sass.contains("FADD"));
    assert!(sass.sass.contains("LDG"));
    assert!(sass.sass.contains("STG"));
}

#[test]
fn apx_9_4_invalid_ptx_returns_empty() {
    let bad_ptx = "this is not ptx";

    let sass = SassTranslator::translate(bad_ptx);

    assert!(sass.sass.contains("invalid"));
}

#[test]
fn apx_9_4_no_numeric_change() {
    let mut v = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    for i in 0..v.len() {
        v[i] += b[i];
    }

    assert_eq!(v, vec![5.0, 7.0, 9.0]);
}
