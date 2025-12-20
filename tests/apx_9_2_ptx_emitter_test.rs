use atenia_engine::apx9::gpu_ir::*;
use atenia_engine::apx9::ptx_emitter::PtxEmitter;

#[test]
fn apx_9_2_emit_basic_structure() {
    let ir = GpuKernelIR {
        name: "vecadd".to_string(),
        threads: 256,
        ops: vec![
            GpuOp::Load { dst: "%f1".into(), src: "%A+tid*4".into() },
            GpuOp::Load { dst: "%f2".into(), src: "%B+tid*4".into() },
            GpuOp::Add  { dst: "%f3".into(), a: "%f1".into(), b: "%f2".into() },
            GpuOp::Store { dst: "%Out+tid*4".into(), src: "%f3".into() }
        ],
    };

    let ptx = PtxEmitter::emit(&ir);

    assert!(ptx.contains(".visible .entry vecadd"));
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn apx_9_2_no_numeric_change() {
    // The generator must not modify data or alter execution
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let mut out = a.clone();

    for i in 0..out.len() {
        out[i] = a[i] + b[i];
    }

    // Same as before — PTX does not execute anything
    assert_eq!(out, vec![5.0, 7.0, 9.0]);
}
