mod mini_flux_common;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::mini_flux::{build_mini_flux_language_model, MiniFluxConfig};
use mini_flux_common::{build_language_trainer, default_cfg, next_token_targets, sample_tokens};

#[test]
fn language_logits_shape_is_correct() {
    let cfg = MiniFluxConfig {
        vocab_size: 32,
        seq_len: 10,
        d_model: 48,
        d_hidden: 96,
        num_layers: 1,
        batch_size: 2,
    };

    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let (logits_id, _) = build_mini_flux_language_model(&mut gb, &cfg, tokens_id);
    gb.output(logits_id);
    let mut graph = gb.build();

    let tokens = sample_tokens(&cfg, 0);
    let outputs = graph.execute(vec![tokens]);
    let logits = outputs.into_iter().next().expect("missing logits");
    assert_eq!(logits.shape, vec![cfg.batch_size, cfg.seq_len, cfg.vocab_size]);
}

#[test]
fn language_loss_reduces_over_time() {
    let cfg = default_cfg(2);
    let mut trainer = build_language_trainer(&cfg);

    let tokens = sample_tokens(&cfg, 0);
    let targets = next_token_targets(&tokens, cfg.vocab_size);

    let mut initial_loss = None;
    let mut best_loss = f32::MAX;

    // Hacemos este test razonablemente rápido y menos estricto para CI:
    // - 200 pasos de entrenamiento.
    // - Solo requerimos una mejora moderada en la loss.
    for step in 0..200 {
        let outputs = trainer.train_step(vec![tokens.clone(), targets.clone()]);
        let loss = outputs[0].data[0];
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        if loss < best_loss {
            best_loss = loss;
        }
        if atenia_engine::apx_debug_enabled() && step % 100 == 0 {
            eprintln!("step {step}: loss={loss:.4}");
        }
    }

    let start_loss = initial_loss.expect("training steps produced no loss");
    // Requerimos que la mejor loss mejore al menos ~10% respecto del inicio.
    let target = start_loss * 0.9;
    assert!(
        best_loss <= target,
        "loss didn't reduce enough: start {start_loss}, best {best_loss}, target {target}"
    );
}
