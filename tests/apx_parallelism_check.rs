use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn apx_parallelism_detected() {
    let counter = AtomicUsize::new(0);

    (0..1000).into_par_iter().for_each(|_| {
        counter.fetch_add(1, Ordering::SeqCst);
    });

    let count = counter.load(Ordering::SeqCst);
    assert_eq!(count, 1000, "Parallel forEach failed");

    let threads = rayon::current_num_threads();
    assert!(threads > 1, "Rayon did not spawn multiple threads");
}
