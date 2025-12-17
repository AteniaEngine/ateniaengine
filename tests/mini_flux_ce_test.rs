use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn manual_cross_entropy(logits: &Tensor, targets: &[usize]) -> f32 {
    let shape = &logits.shape;
    assert_eq!(shape.len(), 3, "logits must be [batch, seq, vocab]");
    let batch = shape[0];
    let seq = shape[1];
    let vocab = shape[2];
    assert_eq!(targets.len(), batch * seq);

    let mut total = 0.0f32;
    for row in 0..(batch * seq) {
        let start = row * vocab;
        let end = start + vocab;
        let slice = &logits.data[start..end];
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = 0.0f32;
        for &v in slice {
            sum_exp += (v - max_val).exp();
        }
        let log_denom = max_val + sum_exp.ln();
        let target_idx = targets[row];
        total += slice[target_idx] - log_denom;
    }

    -total / (batch * seq) as f32
}

#[test]
fn cross_entropy_matches_manual() {
    let mut logits = Tensor::with_layout(
        vec![2, 2, 3],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    logits.data = vec![
        1.0, 0.5, -0.5,
        0.2, 0.1, -0.1,
        -0.3, 0.7, 0.1,
        0.9, -0.4, 0.3,
    ];

    let mut targets = Tensor::with_layout(
        vec![2, 2],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    targets.data = vec![0.0, 1.0, 2.0, 1.0];

    let mut gb = GraphBuilder::new();
    let logits_id = gb.input();
    let targets_id = gb.input();
    let log_probs = gb.log_softmax(logits_id);
    let loss = gb.cross_entropy_loss(log_probs, targets_id);
    gb.output(loss);
    let mut graph = gb.build();

    let outputs = graph.execute(vec![logits.clone(), targets.clone()]);
    let graph_loss = outputs[0].data[0];

    let manual_targets: Vec<usize> = targets
        .data
        .iter()
        .map(|v| v.round() as usize)
        .collect();
    let manual_loss = manual_cross_entropy(&logits, &manual_targets);

    assert!(
        (graph_loss - manual_loss).abs() < 1e-5,
        "graph loss {} vs manual {}",
        graph_loss,
        manual_loss
    );
}
