mod mini_flux_common;

use atenia_engine::nn::mini_flux::MiniFluxConfig;
use mini_flux_common::{run_logits_forward, sample_tokens};

#[test]
fn forward_shapes_are_correct() {
    let cfg = MiniFluxConfig {
        vocab_size: 32,
        seq_len: 8,
        d_model: 32,
        d_hidden: 64,
        num_layers: 2,
        batch_size: 4,
    };

    let tokens = sample_tokens(&cfg, 0);
    let logits = run_logits_forward(&cfg, tokens);

    assert_eq!(logits.shape, vec![cfg.batch_size, cfg.seq_len, cfg.vocab_size]);
}
