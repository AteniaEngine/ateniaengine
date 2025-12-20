use atenia_engine::gpu::planning::{AutoPlanner, LaunchConfig};

/// APX 12.1: basic AutoPlanner test for square matmul.
#[test]
fn apx_12_1_autoplanner_square_matmul() {
    let n = 512usize;
    let cfg: LaunchConfig = AutoPlanner::plan_square_matmul(n);

    // For N=512, our heuristic should choose 16x16.
    assert_eq!(cfg.block, (16, 16, 1));
    assert_eq!(cfg.grid, (32, 32, 1));

    // shared_mem must be > 0 for tiled kernels.
    assert!(cfg.shared_mem > 0, "shared_mem should be greater than zero");
}
