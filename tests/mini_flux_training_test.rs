mod mini_flux_common;

use atenia_engine::nn::mini_flux::MiniFluxConfig;
use mini_flux_common::{build_trainer, default_cfg, sample_tokens, tokens_to_one_hot};

#[test]
fn mini_flux_learns_identity() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "4.13");
    }

    let cfg = MiniFluxConfig {
        vocab_size: 16,
        seq_len: 6,
        d_model: 32,
        d_hidden: 64,
        num_layers: 2,
        batch_size: 8,
    };

    let mut trainer = build_trainer(&cfg);
    let mut first_loss = None;
    let mut best_loss = f32::INFINITY;

    for step in 0..100 {
        let tokens = sample_tokens(&cfg, step);
        let targets = tokens_to_one_hot(&tokens, cfg.vocab_size);
        let outputs = trainer.train_step(vec![tokens.clone(), targets]);
        let loss = outputs[0].data[0];
        if step == 0 {
            first_loss = Some(loss);
        }
        if loss < best_loss {
            best_loss = loss;
        }
    }

    let first = first_loss.expect("loss missing at step 0");
    // Requerimos una mejora moderada (~10%) respecto de la loss inicial.
    let target = first * 0.9;
    assert!(
        best_loss <= target,
        "loss did not drop sufficiently: first={first}, best={best_loss}, target={target}"
    );
}

#[test]
fn logits_shape_matches_expectation() {
    let cfg = default_cfg(4);
    let logits = mini_flux_common::run_logits_forward(&cfg, sample_tokens(&cfg, 1));
    assert_eq!(logits.shape, vec![cfg.batch_size, cfg.seq_len, cfg.vocab_size]);
}

#[test]
fn parameters_receive_gradients() {
    let cfg = default_cfg(2);
    let (mut graph, param_ids) = mini_flux_common::build_training_graph(&cfg);
    let tokens = sample_tokens(&cfg, 0);
    let targets = tokens_to_one_hot(&tokens, cfg.vocab_size);
    let outputs = graph.execute(vec![tokens.clone(), targets.clone()]);
    assert_eq!(outputs.len(), 1);
    let loss_id = graph.last_output_id();
    graph.backward(loss_id);

    for pid in param_ids {
        let grad_present = graph.nodes[pid]
            .output
            .as_ref()
            .and_then(|t| t.grad.as_ref())
            .map(|g| g.iter().any(|v| v.abs() > 1e-9))
            .unwrap_or(false);
        assert!(grad_present, "parameter {pid} missing gradient");
    }
}
