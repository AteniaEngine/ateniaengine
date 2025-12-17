use atenia_engine::apx4_12::{pool_alloc, pool_free};
use std::time::Instant;

fn main() {
    println!("[APX 4.12 BENCH] Starting pool alloc/free...");

    let sizes: [usize; 3] = [4096, 65_536, 131_072];

    for size in sizes {
        let bytes = size;
        let t0 = Instant::now();
        for _ in 0..500 {
            let ptr = pool_alloc();
            pool_free(ptr);
        }
        let dt = t0.elapsed().as_millis();

        println!(
            "[APX 4.12] size={} bytes alloc+free 500x d {} ms",
            bytes, dt
        );
    }
}
