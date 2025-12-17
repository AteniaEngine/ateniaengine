use std::time::Instant;

use atenia_engine::apx6_9::fusion_profiler::FusionProfiler;

fn fake_fused_op_work(amount: usize) {
    // Trabajo dummy proporcional a amount para simular costo fused.
    let mut acc = 0.0f32;
    for i in 0..amount {
        acc += (i as f32 * 0.13).sin();
    }
    std::hint::black_box(acc);
}

fn fake_unfused_op_work(amount: usize) {
    // Trabajo dummy proporcional a amount para simular costo unfused.
    let mut acc = 0.0f32;
    for i in 0..amount {
        acc += (i as f32 * 0.17).cos();
    }
    std::hint::black_box(acc);
}

#[test]
fn apx_6_9_fusion_benchmark() {
    let mut fp = FusionProfiler::new();

    let sizes = [128usize, 256, 512, 1024];

    for &s in &sizes {
        let work = s * s;

        let t0 = Instant::now();
        fake_unfused_op_work(work);
        let unfused_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        fake_fused_op_work(work);
        let fused_us = t1.elapsed().as_micros() as u64;

        fp.record("FusedQKV", unfused_us, fused_us);
        let decision = fp.should_use_fused("FusedQKV");

        println!(
            "[APX 6.9] op=FusedQKV baseline={}us fused={}us selected={:?}",
            unfused_us,
            fused_us,
            decision
        );
    }
}
