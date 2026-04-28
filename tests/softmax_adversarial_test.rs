//! Adversarial softmax tests — investigation of SmolLM2 pos-0 drift
//! (M4.6 numerical validation diagnosis).
//!
//! Hypothesis under test: `softmax_last_dim` may produce non-canonical
//! output when given inputs of the shape `[s, -inf, -inf, -inf]` —
//! exactly the row that the causal mask produces at position 0 of
//! attention. If softmax does not collapse cleanly to `[1, 0, 0, 0]`,
//! the position-0 attention output is corrupted, which would manifest
//! downstream as drift concentrated only in pos 0 (which is what we
//! observed).
//!
//! These tests do NOT modify any production code. They only exercise
//! the existing `Softmax` NodeType through the public builder API.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;

fn run_softmax(input_data: Vec<f32>, shape: Vec<usize>) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let input_id = gb.input();
    let output_id = gb.softmax(input_id);
    let _ = gb.output(output_id);
    let mut graph = gb.build();
    let outputs = graph.execute(vec![Tensor::new_cpu(shape, input_data)]);
    outputs[0].as_cpu_slice().to_vec()
}

fn assert_pos0_canonical(output: &[f32], label: &str) {
    println!(
        "{}: output = [{:.10}, {:.10}, {:.10}, {:.10}]   sum = {:.10}",
        label,
        output[0],
        output[1],
        output[2],
        output[3],
        output.iter().sum::<f32>()
    );
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "{}: expected output[0] ≈ 1.0, got {}",
        label,
        output[0]
    );
    for i in 1..4 {
        assert!(
            output[i] == 0.0 || output[i].abs() < 1e-30,
            "{}: expected output[{}] == 0.0, got {}",
            label,
            i,
            output[i]
        );
    }
}

#[test]
fn softmax_pos0_pattern_zero_score() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![0.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=0");
}

#[test]
fn softmax_pos0_pattern_small_positive_score() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![1.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=1");
}

#[test]
fn softmax_pos0_pattern_large_positive_score() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![10.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=10");
}

#[test]
fn softmax_pos0_pattern_negative_score() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![-10.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=-10");
}

#[test]
fn softmax_pos0_pattern_very_large_positive() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![100.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=100");
}

#[test]
fn softmax_pos0_pattern_very_large_negative() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![-100.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=-100");
}

#[test]
fn softmax_pos0_pattern_extreme_positive() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![1000.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=1000");
}

#[test]
fn softmax_pos0_pattern_extreme_negative() {
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![-1000.0, inf, inf, inf], vec![1, 4]);
    assert_pos0_canonical(&output, "s=-1000");
}

#[test]
fn softmax_pos1_pattern() {
    // Causal mask row at position 1: [s0, s1, -inf, -inf]
    // Expected: shifted-exp softmax over [1.0, 2.0]
    //   max = 2.0 → shifted = [-1.0, 0.0]
    //   exp = [e^-1, 1.0] = [0.36788, 1.0]
    //   sum = 1.36788
    //   output = [0.26894, 0.73106]
    let inf = f32::NEG_INFINITY;
    let output = run_softmax(vec![1.0, 2.0, inf, inf], vec![1, 4]);

    println!(
        "pos1 [1.0, 2.0, -inf, -inf]: {:?}   sum = {}",
        output,
        output.iter().sum::<f32>()
    );

    let expected_0 = 0.26894142_f32;
    let expected_1 = 0.7310586_f32;
    assert!(
        (output[0] - expected_0).abs() < 1e-5,
        "pos1[0]: expected {}, got {}",
        expected_0,
        output[0]
    );
    assert!(
        (output[1] - expected_1).abs() < 1e-5,
        "pos1[1]: expected {}, got {}",
        expected_1,
        output[1]
    );
    assert!(output[2].abs() < 1e-30, "pos1[2] should be ~0, got {}", output[2]);
    assert!(output[3].abs() < 1e-30, "pos1[3] should be ~0, got {}", output[3]);
}

/// Sweep across the realistic range of attention scores in SmolLM2.
/// With head_dim=64, the scale factor is 1/sqrt(64) = 0.125. Raw
/// Q@K.T values can land roughly in [-100, +100]; after scaling, in
/// [-12.5, +12.5]. We test this band plus the boundaries.
#[test]
fn softmax_realistic_attention_scores_range() {
    let inf = f32::NEG_INFINITY;
    for &s in &[-12.5_f32, -5.0, -1.0, 0.0, 1.0, 5.0, 12.5] {
        let output = run_softmax(vec![s, inf, inf, inf], vec![1, 4]);
        println!(
            "s={:>6.2}: output[0]={:.10}  sum={:.10}",
            s,
            output[0],
            output.iter().sum::<f32>()
        );
        assert!(
            (output[0] - 1.0).abs() < 1e-6,
            "s={}: output[0]={} (expected 1.0)",
            s,
            output[0]
        );
        for i in 1..4 {
            assert!(
                output[i] == 0.0 || output[i].abs() < 1e-30,
                "s={}: output[{}]={} (expected 0)",
                s,
                i,
                output[i]
            );
        }
    }
}

/// Same shape as the actual attention scores tensor in the SmolLM2 graph:
/// [batch=1, n_heads=32, seq=4, seq=4]. Reproduces the four causal-mask
/// rows pattern in a single softmax call.
#[test]
fn softmax_full_causal_mask_pattern_4d() {
    let inf = f32::NEG_INFINITY;
    let s = 1.5_f32;
    let row0 = [s, inf, inf, inf];
    let row1 = [s, s, inf, inf];
    let row2 = [s, s, s, inf];
    let row3 = [s, s, s, s];
    let mut data = Vec::with_capacity(1 * 32 * 4 * 4);
    for _h in 0..32 {
        // 32 heads, all rows the same content for simplicity
        data.extend_from_slice(&row0);
        data.extend_from_slice(&row1);
        data.extend_from_slice(&row2);
        data.extend_from_slice(&row3);
    }
    let output = run_softmax(data, vec![1, 32, 4, 4]);

    // Each head should have identical 4×4 output. Pull head 0.
    let head0 = &output[0..16];
    println!("4D causal-mask head 0:");
    for r in 0..4 {
        let row = &head0[r * 4..(r + 1) * 4];
        let sum: f32 = row.iter().sum();
        println!(
            "  row {}: [{:.6}, {:.6}, {:.6}, {:.6}]  sum={:.10}",
            r, row[0], row[1], row[2], row[3], sum
        );
    }

    // Row 0 of the head 0 output is the canonical [1, 0, 0, 0]
    assert!((head0[0] - 1.0).abs() < 1e-6, "row0[0]: {}", head0[0]);
    assert!(head0[1].abs() < 1e-30, "row0[1]: {}", head0[1]);
    assert!(head0[2].abs() < 1e-30, "row0[2]: {}", head0[2]);
    assert!(head0[3].abs() < 1e-30, "row0[3]: {}", head0[3]);

    // Rows must each sum to 1.0
    for r in 0..4 {
        let sum: f32 = head0[r * 4..(r + 1) * 4].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {} sum should be 1.0, got {}",
            r,
            sum
        );
    }
}
