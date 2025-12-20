use atenia_engine::{softmax3, sample_decision};

fn run_one(temp: f32, iters: usize) {
    // Synthetic scores: full clearly better than qkv/base.
    let td = softmax3(10.0, 3.0, 1.0, temp);

    let mut full = 0;
    let mut qkv = 0;
    let mut base = 0;

    for _ in 0..iters {
        match sample_decision(&td) {
            "full" => full += 1,
            "qkv" => qkv += 1,
            _ => base += 1,
        }
    }

    println!(
        "[APX 6.13] T={temp:.2} | p_full={:.4} p_qkv={:.4} p_base={:.4} | counts: full={} qkv={} base={}",
        td.p_full,
        td.p_qkv,
        td.p_base,
        full,
        qkv,
        base,
    );
}

#[test]
fn apx_6_13_tempered_behavior_bench() {
    // Light benchmark, logs only: no hard assertions to avoid
    // flakiness due to randomness. Used to manually inspect how
    // the distribution changes as temperature varies.
    let iters = 10_000;

    for &temp in &[0.3f32, 0.8f32, 2.0f32] {
        run_one(temp, iters);
    }
}
