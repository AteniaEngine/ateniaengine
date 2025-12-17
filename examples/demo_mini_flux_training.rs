use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::cpu_features::cpu_features;
use atenia_engine::data::tinydata::{
    build_vocab,
    encode_text,
    load_text_dataset,
    make_batches,
    Vocab,
};
use atenia_engine::nn::mini_flux::{
    build_language_training_graph,
    build_mini_flux_language_model,
    MiniFluxConfig,
};
use atenia_engine::optim::adamw::AdamW;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer_v2::TrainerV2;

const DATA_PATH: &str = "data/tiny_shakespeare.txt";
const TRAIN_STEPS: usize = 600;
const SAMPLE_STEPS: usize = 200;

fn main() {
    let features = cpu_features();
    println!(
        "[CPU] SIMD features -> AVX512F: {}, AVX2: {}, FMA: {}",
        features.avx512f, features.avx2, features.fma
    );

    let text = match load_text_dataset(DATA_PATH) {
        Ok(t) => t,
        Err(err) => {
            eprintln!(
                "Could not open {DATA_PATH}: {err}.\nDownload Tiny Shakespeare (tinyshakespeare.txt) and place it at that path before running this example."
            );
            return;
        }
    };
    let vocab = build_vocab(&text);
    let encoded = encode_text(&text, &vocab);

    let cfg = MiniFluxConfig {
        vocab_size: vocab.size(),
        seq_len: 64,
        d_model: 128,
        d_hidden: 256,
        num_layers: 2,
        batch_size: 4,
    };

    let batches = make_batches(encoded.clone(), cfg.seq_len, cfg.batch_size);
    assert!(!batches.is_empty(), "dataset too small for requested configuration");

    let (graph, param_ids) = build_language_training_graph(&cfg);
    let optim = AdamW::new(param_ids.len(), 0.008, 0.9, 0.999, 1e-8, 0.0);
    let mut trainer = TrainerV2::new(graph, param_ids, optim);

    for step in 0..TRAIN_STEPS {
        let (inputs, targets) = &batches[step % batches.len()];
        let outputs = trainer.train_step(vec![inputs.clone(), targets.clone()]);
        let loss = outputs[0].data[0];
        if step % 20 == 0 {
            println!("Step {:3}: loss = {:.6}", step, loss);
        }
    }

    let mut infer_builder = GraphBuilder::new();
    let tokens_id = infer_builder.input();
    let (logits_id, infer_param_ids) = build_mini_flux_language_model(&mut infer_builder, &cfg, tokens_id);
    infer_builder.output(logits_id);
    let mut infer_graph = infer_builder.build();

    copy_parameters(
        &trainer.graph,
        &trainer.param_ids,
        &mut infer_graph,
        &infer_param_ids,
    );

    let prompt = "H";
    let generated = generate_text(&mut infer_graph, &cfg, &vocab, prompt, SAMPLE_STEPS);
    println!("Generated text:\n{}", generated);
}

fn copy_parameters(src: &Graph, src_ids: &[usize], dst: &mut Graph, dst_ids: &[usize]) {
    assert_eq!(src_ids.len(), dst_ids.len());
    for (s, d) in src_ids.iter().zip(dst_ids.iter()) {
        let src_tensor = src.nodes[*s]
            .output
            .as_ref()
            .expect("source parameter missing output");
        let dst_tensor = dst.nodes[*d]
            .output
            .as_mut()
            .expect("destination parameter missing output");
        dst_tensor.data.copy_from_slice(&src_tensor.data);
    }
}

fn generate_text(
    graph: &mut Graph,
    cfg: &MiniFluxConfig,
    vocab: &Vocab,
    prompt: &str,
    steps: usize,
) -> String {
    let mut row_tokens = prompt_to_tokens(prompt, vocab, cfg.seq_len);
    let mut text = prompt.to_string();
    for _ in 0..steps {
        let input = build_inference_input(&row_tokens, cfg);
        let outputs = graph.execute(vec![input]);
        let logits = outputs.into_iter().next().expect("missing logits output");
        let vocab_slice = extract_last_position(&logits, cfg);
        let next_token = argmax(vocab_slice);
        row_tokens.remove(0);
        row_tokens.push(next_token);
        let ch = vocab.itos[next_token];
        text.push(ch);
    }
    text
}

fn prompt_to_tokens(prompt: &str, vocab: &Vocab, seq_len: usize) -> Vec<usize> {
    let mut tokens = vec![0usize; seq_len];
    let prompt_tokens: Vec<usize> = prompt
        .chars()
        .map(|c| *vocab.stoi.get(&c).unwrap_or(&0))
        .collect();
    let take = prompt_tokens.len().min(seq_len);
    tokens[seq_len - take..].copy_from_slice(&prompt_tokens[prompt_tokens.len() - take..]);
    tokens
}

fn build_inference_input(row_tokens: &[usize], cfg: &MiniFluxConfig) -> Tensor {
    let mut tensor = Tensor::with_layout(
        vec![cfg.batch_size, cfg.seq_len],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for row in 0..cfg.batch_size {
        for col in 0..cfg.seq_len {
            let idx = row * cfg.seq_len + col;
            tensor.data[idx] = row_tokens[col] as f32;
        }
    }
    tensor
}

fn extract_last_position<'a>(logits: &'a Tensor, cfg: &MiniFluxConfig) -> &'a [f32] {
    let vocab = cfg.vocab_size;
    let row = cfg.seq_len - 1;
    let start = row * vocab;
    let end = start + vocab;
    &logits.data[start..end]
}

fn argmax(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}
