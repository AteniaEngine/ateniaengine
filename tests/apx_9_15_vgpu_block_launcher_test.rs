use atenia_engine::{VGpuMemory, VGpuBlockLauncher};
use atenia_engine::apx9::gpu_ir::*;

#[test]
fn apx_9_15_launches_threads_and_blocks() {
    // IR simple: out = in1 + in2, ejecutado sobre varios bloques/threads virtuales.
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

    // Lanzar grid virtual de 2 bloques, 4 threads c/u.
    VGpuBlockLauncher::launch(&ir, &mut mem, 2, 4);

    // Como el IR no depende aún de block_id/thread_id, el resultado final
    // coincide con una sola ejecución del kernel.
    assert_eq!(mem.load_global(out_idx), 5.0);
}

#[test]
fn apx_9_15_structure() {
    let ir = GpuKernelIR { name: "empty".into(), threads: 1, ops: vec![] };
    let mut mem = VGpuMemory::new(64, 16, 3, 8);

    // Debe poder lanzar un grid vacío sin panics ni efectos secundarios.
    VGpuBlockLauncher::launch(&ir, &mut mem, 3, 8);
}
