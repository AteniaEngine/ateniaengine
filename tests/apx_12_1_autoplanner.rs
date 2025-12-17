use atenia_engine::gpu::planning::{AutoPlanner, LaunchConfig};

/// APX 12.1: test básico del AutoPlanner para matmul cuadrado.
#[test]
fn apx_12_1_autoplanner_square_matmul() {
    let n = 512usize;
    let cfg: LaunchConfig = AutoPlanner::plan_square_matmul(n);

    // Para N=512, nuestra heurística debería elegir 16x16.
    assert_eq!(cfg.block, (16, 16, 1));
    assert_eq!(cfg.grid, (32, 32, 1));

    // shared_mem debe ser > 0 para kernels tileados.
    assert!(cfg.shared_mem > 0, "shared_mem should be greater than zero");
}
