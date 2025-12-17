#[test]
fn apx_12_3_autotuner_basic() {
    use atenia_engine::gpu::autotuner::*;

    // Simula GPU runner
    fn mock_runner(layout: (u32, u32, u32, u32)) -> f32 {
        let (bx, by, _, _) = layout;
        // Simulamos "más rápido cuanto mayor el bloque"
        50.0 / ((bx * by) as f32)
    }

    let r = autotune_matmul(128, 89, &mock_runner, true);
    assert!(r.block_x >= 16);
    assert!(r.block_y >= 16);
}

#[test]
fn apx_12_3_cpu_fallback_mode() {
    use atenia_engine::gpu::autotuner::*;

    fn runner(_: (u32, u32, u32, u32)) -> f32 { 1.0 }

    let r = autotune_matmul(128, 89, &runner, false);
    assert_eq!(r.block_x, 16);
    assert_eq!(r.block_y, 16);
}
