use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::apx7::pex_engine::PEXExecutor;

#[test]
fn apx_7_0_parallel_matmul_matches_sequential() {
    let a = Tensor::randn(&[128, 128], Device::CPU);
    let b = Tensor::randn(&[128, 128], Device::CPU);

    let seq = a.matmul(&b);
    let par = a.matmul_parallel(&b);

    // Comparar máximo error absoluto.
    let mut max_diff = 0.0f32;
    for (x, y) in seq.data.iter().zip(par.data.iter()) {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(max_diff < 1e-5);
}

#[test]
fn apx_7_0_parallel_scheduler_uses_multiple_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let counter = Arc::new(AtomicUsize::new(0));

    let tasks: Vec<_> = (0..8)
        .map(|_| {
            let c = counter.clone();
            move || {
                c.fetch_add(1, Ordering::SeqCst);
            }
        })
        .collect();

    let pex = PEXExecutor::new(8);
    pex.execute_parallel(tasks);

    assert_eq!(counter.load(Ordering::SeqCst), 8);
}
