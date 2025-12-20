use atenia_engine::apx9::vgpu_sync::*;
use atenia_engine::{VGpuMemory, VGpuRunner};
use atenia_engine::apx9::gpu_ir::*;

#[test]
fn apx_9_16_sync_structure() {
    let barrier = VGPUBarrier::new(8);
    assert_eq!(barrier.threads, 8);
}

#[test]
fn apx_9_16_sync_basic() {
    let mut b = VGPUBarrier::new(4);
    b.arrive(); b.arrive(); b.arrive(); b.arrive();
    assert!(b.is_complete());
}

#[test]
fn apx_9_16_sync_ir_integration() {
    // IR: out = in1 + in2, with a Sync in the middle to verify it does not break execution.
    let ir = GpuKernelIR {
        name: "vk_sync_add".into(),
        threads: 4,
        ops: vec![
            GpuOp::Load { dst: "a".into(), src: "in".into() },
            GpuOp::Load { dst: "b".into(), src: "in2".into() },
            GpuOp::Add  { dst: "c".into(), a: "a".into(), b: "b".into() },
            GpuOp::Sync,
            GpuOp::Store{ dst: "out".into(), src: "c".into() },
        ],
    };

    let mut mem = VGpuMemory::new(32, 16, 1, 4);

    let in_idx  = VGpuRunner::hash_slot("in");
    let in2_idx = VGpuRunner::hash_slot("in2");
    let out_idx = VGpuRunner::hash_slot("out");

    mem.store_global(in_idx, 1.5);
    mem.store_global(in2_idx, 2.5);

    VGpuRunner::run_kernel(&ir, &mut mem, 0, 0);

    // The barrier does not change the math: 1.5 + 2.5 = 4.0
    assert_eq!(mem.load_global(out_idx), 4.0);
}
