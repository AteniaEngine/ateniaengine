use std::time::Instant;

use atenia_engine::apx6_8::BlockSizePredictor;
use atenia_engine::matmul::matmul_tiled_flex::matmul_tiled_flex;

#[test]
fn apx_6_8_block_prediction() {
    let sizes = BlockSizePredictor::candidate_block_sizes();
    let mut pred = BlockSizePredictor::new();

    let m = 256usize;
    let k = 256usize;
    let n = 256usize;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut out = vec![0.0f32; m * n];

    for (bm, bn, bk) in sizes {
        out.fill(0.0);
        let t0 = Instant::now();
        matmul_tiled_flex(&a, &b, &mut out, m, k, n, bm, bn, bk);
        let us = t0.elapsed().as_micros();
        pred.record_result(bm, bn, bk, us);
        eprintln!("[APX 6.8 PRED] bm={bm} bn={bn} bk={bk} -> {us} us");
    }

    let best = pred.best_block().unwrap();
    eprintln!("[APX 6.8 BEST] {:?}", best);
}
