use atenia_engine::tensor::{Tensor, Device};

#[test]
fn apx_7_3_adaptive_learns_best_strategy() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.3");
    }

    let sizes = [128_usize, 256];

    for _ in 0..8 {
        for &n in &sizes {
            let a = Tensor::randn(&[n, n], Device::CPU);
            let b = Tensor::randn(&[n, n], Device::CPU);
            let _ = a.matmul_adaptive(&b);
        }
    }

    // No validamos rendimiento exacto, pero sí que aprendió algo en el
    // bucket de tamaños pequeños.
    let buckets = atenia_engine::apx7::adaptive_pgl::ADAPTIVE_BUCKETS.read().unwrap();
    assert!(buckets[0].count >= 5);
}
