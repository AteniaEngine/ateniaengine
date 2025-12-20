use atenia_engine::{VGpuMemory, VGpuBlockLauncher};
use atenia_engine::apx9::gpu_ir::*;

#[test]
fn apx_9_15_launches_threads_and_blocks() {
    // Simple IR: out = in1 + in2, executed over several virtual blocks/threads.
    let ir = GpuKernelIR {
        name: "vk_add_grid".into(),
        threads: 4,
        ops: vec![
            GpuOp::Load { dst: "a".into(), src: "in".into() },
            GpuOp::Load { dst: "b".into(), src: "in2".into() },
            GpuOp::Add { dst: "c".into(), a: "a".into(), b: "b".into() },
            GpuOp::Store { dst: "out".into(), src: "c".into() },
        ],
    };

    let mut mem = VGpuMemory::new(32, 16, 2, 4);

    let in_idx = atenia_engine::VGpuRunner::hash_slot("in");
    let in2_idx = atenia_engine::VGpuRunner::hash_slot("in2");
    let out_idx = atenia_engine::VGpuRunner::hash_slot("out");

    mem.store_global(in_idx, 2.0);
    mem.store_global(in2_idx, 3.0);

    // Launch a virtual grid of 2 blocks, 4 threads each.
    VGpuBlockLauncher::launch(&ir, &mut mem, 2, 4);

    // Since the IR does not depend on block_id/thread_id yet, the final result
    // matches a single kernel execution.
    assert_eq!(mem.load_global(out_idx), 5.0);
}

#[test]
fn apx_9_15_structure() {
    let ir = GpuKernelIR { name: "empty".into(), threads: 1, ops: vec![] };
    let mut mem = VGpuMemory::new(64, 16, 3, 8);

    // It should be able to launch an empty grid without panics or side effects.
    VGpuBlockLauncher::launch(&ir, &mut mem, 3, 8);
}
