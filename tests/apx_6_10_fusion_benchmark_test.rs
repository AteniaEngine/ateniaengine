use std::time::Instant;

use atenia_engine::apx6_10::{FusionSelector, FusionProfile};

fn baseline_op(work: usize) {
    let mut acc = 0.0f32;
    for i in 0..work {
        acc += (i as f32 * 0.11).sin();
    }
    std::hint::black_box(acc);
}

fn fused_qkv_placeholder(work: usize) {
    let mut acc = 0.0f32;
    for i in 0..work {
        acc += (i as f32 * 0.13).cos();
    }
    std::hint::black_box(acc);
}

fn fused_full_placeholder(work: usize) {
    // Simular QKV + softmax + proj + bias con trabajo algo mayor.
    let mut acc = 0.0f32;
    for i in 0..(work * 2) {
        let x = i as f32 * 0.07;
        acc += x.tanh() + x.exp().ln();
    }
    std::hint::black_box(acc);
}

#[test]
fn apx_6_10_fusion_benchmark() {
    let mut selector = FusionSelector::new();

    let sizes = [128usize, 256, 512, 1024];

    for &s in &sizes {
        let work = s * s;

        let t0 = Instant::now();
        baseline_op(work);
        let baseline = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        fused_qkv_placeholder(work);
        let fused_qkv = t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        fused_full_placeholder(work);
        let fused_full = t2.elapsed().as_micros() as u64;

        selector.record_profile(FusionProfile {
            op_name: "FusedAttention".to_string(),
            baseline_us: baseline,
            fused_qkv_us: fused_qkv,
            fused_full_us: fused_full,
        });

        let decision = selector.decide();

        println!(
            "[APX 6.10 FUSION] baseline={}us qkv={}us full={}us decision={:?}",
            baseline,
            fused_qkv,
            fused_full,
            decision.use_full_fusion,
        );
    }
}
