use atenia_engine::{VGpuMemory, VGpuRunner};
use atenia_engine::apx9::gpu_ir::*;

#[test]
fn apx_9_14_executes_add_correctly() {
    let ir = GpuKernelIR {
        name: "vk_add".into(),
        threads: 4,
        ops: vec![
            GpuOp::Load { dst: "a".into(), src: "in".into() },
            GpuOp::Load { dst: "b".into(), src: "in2".into() },
            GpuOp::Add { dst: "c".into(), a: "a".into(), b: "b".into() },
            GpuOp::Store { dst: "out".into(), src: "c".into() },
        ],
    };

    let mut mem = VGpuMemory::new(32, 8, 1, 4);

    let in_idx = VGpuRunner::hash_slot("in");
    let in2_idx = VGpuRunner::hash_slot("in2");
    let out_idx = VGpuRunner::hash_slot("out");

    mem.store_global(in_idx, 10.0);
    mem.store_global(in2_idx, 1.0);

    VGpuRunner::run_kernel(&ir, &mut mem, 0, 0);

    assert_eq!(mem.load_global(out_idx), 11.0);
}

#[test]
fn apx_9_14_structure_is_valid() {
    let ir = GpuKernelIR { name: "empty".into(), threads: 1, ops: vec![] };
    let mut mem = VGpuMemory::new(32, 16, 2, 4);

    VGpuRunner::run_kernel(&ir, &mut mem, 1, 2);
}
