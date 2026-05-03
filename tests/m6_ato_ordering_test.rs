//! M6 ATO ordering fix — empirical test that GPU dispatch is
//! attempted **before** the Auto-Tiling Optimizer in
//! `Graph::execute_single_inner`'s MatMul arm.
//!
//! Pre-fix the order was: ATO → GPU → legacy CPU. Under APX
//! mode ≥ 6.6 with all-Cpu F32 operands the ATO block returned
//! before `try_gpu_matmul` could fire, leaving the GPU surface
//! unreachable in production. Post-fix the order is: GPU → ATO
//! → legacy CPU, so an oversize MatMul (single buffer > 64 MiB)
//! reaches the M6 step 2b non-pooled path even under APX 6.6+.
//!
//! Pre-fix expected counter delta: 0.
//! Post-fix expected counter delta: 1.
//!
//! Skipped on non-CUDA hosts (the same `cuda_available()` skip
//! pattern as `tests/cuda_matmul_residency_test.rs`).

use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::cuda::cuda_available;
use atenia_engine::gpu::dispatch::hooks::gpu_matmul_non_pooled_count;
use atenia_engine::init_apx;
use atenia_engine::tensor::Tensor;

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn gpu_dispatch_runs_before_ato_under_apx_66() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    init_apx();
    // Force APX mode to 6.6 — the mode that triggers the ATO
    // capture pre-fix. Set via the documented env-var contract
    // `ATENIA_APX_MODE`. Rust 2024 marks `set_var`/`remove_var`
    // as unsafe (the C `setenv` is not thread-safe); wrapped in
    // `unsafe` since the test runs single-threaded.
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.6");
    }

    // Oversize shape: any single buffer must exceed
    // `DEFAULT_BLOCK_SIZE = 64 MiB`. 4097² × 4 bytes = 67.14 MB,
    // matches the post-step-2b unit tests in
    // `src/gpu/dispatch/hooks.rs::m6_step_2b_routing_tests`.
    let dim = 4097_usize;
    let a = Tensor::new_cpu(vec![dim, dim], vec![0.0_f32; dim * dim]);
    let b = Tensor::new_cpu(vec![dim, dim], vec![0.0_f32; dim * dim]);

    // Minimal graph: Input A, Input B, MatMul, Output.
    let mut nodes = Vec::new();
    let a_id = 0_usize;
    nodes.push(Node::new(a_id, NodeType::Input, vec![]));
    let b_id = 1_usize;
    nodes.push(Node::new(b_id, NodeType::Input, vec![]));
    let mm_id = 2_usize;
    nodes.push(Node::new(mm_id, NodeType::MatMul, vec![a_id, b_id]));
    let out_id = 3_usize;
    nodes.push(Node::new(out_id, NodeType::Output, vec![mm_id]));

    let mut g = Graph::new(nodes);

    let np_before = gpu_matmul_non_pooled_count();

    // Run the graph. Under the post-fix ordering with APX 6.6 +
    // oversize CPU operands, the MatMul arm:
    //   1. `gpu_can_run_matmul` returns true
    //      (cuda_available, ops > 256, no pool size gate).
    //   2. `try_gpu_matmul` routes to `cuda_matmul_non_pooled`
    //      (max_per_alloc > 64 MiB) and increments the counter.
    //   3. ATO never sees the node; legacy fallback never sees it.
    //
    // Pre-fix the ATO block at line 3155 of `graph.rs` returned
    // before step 1 could fire, so the counter would not move.
    let _outputs = g.execute(vec![a, b]);

    let np_after = gpu_matmul_non_pooled_count();

    // Restore environment so this test does not leak state into
    // subsequent tests in the same process.
    unsafe {
        std::env::remove_var("ATENIA_APX_MODE");
    }

    assert_eq!(
        np_after - np_before,
        1,
        "M6 ATO ordering fix regression: gpu_matmul_non_pooled_count \
         did not increment by 1 under APX 6.6 + oversize CPU MatMul. \
         Expected post-fix order is GPU → ATO → legacy; observed \
         delta = {}.",
        np_after - np_before
    );
}
