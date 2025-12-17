use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::{VGpuMemory, VGpuRunner};
use atenia_engine::apx9::gpu_ir::*;

#[test]
fn apx_9_17_structure() {
    let warp = VGPUWarp::new(32, 0);
    assert_eq!(warp.lanes.len(), 32);
}

#[test]
fn apx_9_17_predicate() {
    let mut warp = VGPUWarp::new(4, 0);
    warp.apply_predicate(|tid| tid % 2 == 0);
    assert!(warp.lanes[0].active);
    assert!(!warp.lanes[1].active);
}

#[test]
fn apx_9_17_reconverge() {
    let mut warp = VGPUWarp::new(4, 0);
    warp.apply_predicate(|_| false);
    warp.reconverge();
    assert!(warp.lanes.iter().all(|l| l.active));
}

#[test]
fn apx_9_17_ir_integration_predicate_noop() {
    // IR: out = in1 + in2, con un Predicate simbólico en medio.
    let ir = GpuKernelIR {
        name: "vk_predicate_add".into(),
        threads: 4,
        ops: vec![
            GpuOp::Load { dst: "a".into(),   src: "in".into() },
            GpuOp::Load { dst: "b".into(),   src: "in2".into() },
            GpuOp::Predicate { lane_mod: 2, value: 0 },
            GpuOp::Add  { dst: "c".into(),   a: "a".into(), b: "b".into() },
            GpuOp::Store{ dst: "out".into(), src: "c".into() },
        ],
    };

    let mut mem = VGpuMemory::new(32, 16, 1, 4);

    let in_idx  = VGpuRunner::hash_slot("in");
    let in2_idx = VGpuRunner::hash_slot("in2");
    let out_idx = VGpuRunner::hash_slot("out");

    mem.store_global(in_idx, 3.0);
    mem.store_global(in2_idx, 4.0);

    // En esta fase, Predicate es un no-op simbólico: la matemática debe mantenerse.
    VGpuRunner::run_kernel(&ir, &mut mem, 0, 0);

    assert_eq!(mem.load_global(out_idx), 7.0);
}
