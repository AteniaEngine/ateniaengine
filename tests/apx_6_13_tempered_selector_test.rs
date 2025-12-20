use atenia_engine::{softmax3, sample_decision};

#[test]
fn apx_6_13_softmax_sums_to_one() {
    let td = softmax3(1.0, 2.0, 3.0, 1.0);
    let s = td.p_full + td.p_qkv + td.p_base;
    assert!((s - 1.0).abs() < 1e-5);
}

#[test]
fn apx_6_13_sampling_varies() {
    let td = softmax3(10.0, 1.0, 1.0, 1.0);

    let mut full = 0;
    let mut qkv = 0;
    let mut base = 0;

    for _ in 0..5000 {
        match sample_decision(&td) {
            "full" => full += 1,
            "qkv" => qkv += 1,
            _ => base += 1,
        }
    }

    // full should be larger because it has the highest score
    assert!(full > qkv && full > base);
}
