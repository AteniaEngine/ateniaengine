//! Phase A.2 (M4.6) — RmsNorm eps configurability tests.
//!
//! Two tests:
//!   1. Custom eps (1e-6) produces output that diverges from default
//!      eps (1e-5) on inputs where eps has measurable relative
//!      impact.
//!   2. Default eps (1e-5) produces well-formed RmsNorm output:
//!      finite values, expected shape, post-norm variance close to
//!      1.0 (sanity baseline).

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;

#[test]
fn rms_norm_with_custom_eps_diverges_from_default() {
    let mut gb1 = GraphBuilder::new();
    let x1 = gb1.input();
    let out1 = gb1.rms_norm(x1, 1e-5);
    let _ = gb1.output(out1);
    let mut graph1 = gb1.build();

    let mut gb2 = GraphBuilder::new();
    let x2 = gb2.input();
    let out2 = gb2.rms_norm(x2, 1e-6);
    let _ = gb2.output(out2);
    let mut graph2 = gb2.build();

    // Small-magnitude input: eps shifts the inv_rms denominator
    // measurably. With ||x|| ~ 1e-3, eps=1e-5 vs 1e-6 changes the
    // sqrt term by a factor large enough to register in F32.
    let input_data = vec![1e-3_f32, 2e-3, 3e-3, 4e-3];
    let input1 = Tensor::new_cpu(vec![1, 4], input_data.clone());
    let input2 = Tensor::new_cpu(vec![1, 4], input_data);

    let out1 = graph1.execute(vec![input1]);
    let out2 = graph2.execute(vec![input2]);

    let v1 = out1[0].as_cpu_slice();
    let v2 = out2[0].as_cpu_slice();

    let max_diff = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff > 1e-6,
        "Custom eps should produce different output than default; max diff = {}",
        max_diff
    );

    println!("eps=1e-5 vs eps=1e-6 max diff: {:.6e}", max_diff);
}

#[test]
fn rms_norm_with_default_eps_produces_unit_variance() {
    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let out = gb.rms_norm(x, 1e-5);
    let _ = gb.output(out);
    let mut graph = gb.build();

    // Hidden-state-like input: 2048 dims, gentle ramp.
    let input_data: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.001).collect();
    let input = Tensor::new_cpu(vec![1, 2048], input_data);

    let out = graph.execute(vec![input]);
    let v = out[0].as_cpu_slice();

    assert_eq!(v.len(), 2048, "output length");

    let finite = v.iter().filter(|x| x.is_finite()).count();
    assert_eq!(finite, 2048, "all outputs must be finite");

    // Post-RmsNorm property: y_i = x_i / sqrt(mean(x²) + eps).
    // Therefore mean(y²) = mean(x²) / (mean(x²) + eps) ≈ 1 when
    // mean(x²) >> eps. With this input, mean(x²) ≈ 1.4 >> 1e-5,
    // so mean(y²) should be very close to 1.
    let mean_sq: f32 = v.iter().map(|x| x * x).sum::<f32>() / 2048.0;
    let mean: f32 = v.iter().sum::<f32>() / 2048.0;

    println!("Output mean: {:.6}, mean_sq: {:.6}", mean, mean_sq);
    assert!(
        (mean_sq - 1.0).abs() < 0.01,
        "mean(y²) should be ≈1.0 (RmsNorm invariant), got {}",
        mean_sq
    );
}
