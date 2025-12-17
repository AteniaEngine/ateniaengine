#[path = "../tests/mini_flux_common.rs"]
mod mini_flux_common;

use atenia_engine::nn::mini_flux::MiniFluxConfig;
use mini_flux_common::{build_trainer, sample_tokens, tokens_to_one_hot};

fn main() {
    let cfg = MiniFluxConfig {
        vocab_size: 16,
        seq_len: 8,
        d_model: 32,
        d_hidden: 64,
        num_layers: 2,
        batch_size: 16,
    };

    let mut trainer = build_trainer(&cfg);

    for step in 0..200 {
        let tokens = sample_tokens(&cfg, step);
        let targets = tokens_to_one_hot(&tokens, cfg.vocab_size);
        let outputs = trainer.train_step(vec![tokens.clone(), targets]);
        let loss = outputs[0].data[0];
        if step % 20 == 0 {
            println!("Step {:3}: loss = {:.6}", step, loss);
        }
    }
}
