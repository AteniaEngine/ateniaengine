use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::NodeType;
use atenia_engine::nn::mini_flux::{build_language_training_graph, build_mini_flux, MiniFluxConfig};
use atenia_engine::optim::adamw::AdamW;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer_v2::TrainerV2;

#[allow(dead_code)]
pub fn default_cfg(batch_size: usize) -> MiniFluxConfig {
    MiniFluxConfig {
        vocab_size: 16,
        seq_len: 6,
        d_model: 24,
        d_hidden: 48,
        num_layers: 1,
        batch_size,
    }
}

#[allow(dead_code)]
pub fn next_token_targets(tokens: &Tensor, vocab: usize) -> Tensor {
    assert_eq!(tokens.shape.len(), 2, "tokens must be [batch, seq]");
    let batch = tokens.shape[0];
    let seq = tokens.shape[1];
    let mut t = Tensor::with_layout(
        vec![batch, seq],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for b in 0..batch {
        for s in 0..seq {
            let idx = b * seq + s;
            let next_col = (s + 1) % seq;
            let next_idx = b * seq + next_col;
            let target = tokens.data[next_idx].round() as usize % vocab;
            t.data[idx] = target as f32;
        }
    }
    t
}

pub fn sample_tokens(cfg: &MiniFluxConfig, seed: usize) -> Tensor {
    let mut t = Tensor::with_layout(
        vec![cfg.batch_size, cfg.seq_len],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for b in 0..cfg.batch_size {
        for s in 0..cfg.seq_len {
            let idx = b * cfg.seq_len + s;
            let val = ((b + s + seed) % cfg.vocab_size) as f32;
            t.data[idx] = val;
        }
    }
    t
}

#[allow(dead_code)]
pub fn tokens_to_one_hot(tokens: &Tensor, vocab: usize) -> Tensor {
    assert_eq!(tokens.shape.len(), 2, "tokens must be [batch, seq]");
    let batch = tokens.shape[0];
    let seq = tokens.shape[1];
    let mut t = Tensor::with_layout(
        vec![batch, seq, vocab],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for b in 0..batch {
        for s in 0..seq {
            let token = tokens.data[b * seq + s].round() as usize % vocab;
            let offset = b * seq * vocab + s * vocab + token;
            t.data[offset] = 1.0;
        }
    }
    t
}

#[allow(dead_code)]
pub fn run_logits_forward(cfg: &MiniFluxConfig, tokens: Tensor) -> Tensor {
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, cfg, tokens_id);
    graph.add_node_of_type(NodeType::Output, vec![handles.logits_id]);
    let outputs = graph.execute(vec![tokens]);
    outputs.into_iter().next().expect("missing logits output")
}

#[allow(dead_code)]
pub fn build_training_graph(cfg: &MiniFluxConfig) -> (Graph, Vec<usize>) {
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let target_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, cfg, tokens_id);
    let flat_shape = vec![
        (cfg.batch_size * cfg.seq_len) as isize,
        cfg.vocab_size as isize,
    ];
    let logits_flat = graph.add_node_of_type(NodeType::Reshape { target: flat_shape.clone() }, vec![handles.logits_id]);
    let target_flat = graph.add_node_of_type(NodeType::Reshape { target: flat_shape }, vec![target_id]);

    let diff = graph.add_node_of_type(NodeType::Sub, vec![logits_flat, target_flat]);
    let sq = graph.add_node_of_type(NodeType::Mul, vec![diff, diff]);

    let ones_feat = Tensor::with_layout(
        vec![cfg.vocab_size, 1],
        1.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let ones_feat_id = graph.add_parameter(ones_feat);
    let sum_features = graph.add_node_of_type(NodeType::MatMul, vec![sq, ones_feat_id]);

    let ones_seq = Tensor::with_layout(
        vec![1, cfg.batch_size * cfg.seq_len],
        1.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let ones_seq_id = graph.add_parameter(ones_seq);
    let total = graph.add_node_of_type(NodeType::MatMul, vec![ones_seq_id, sum_features]);

    let mean_scale = 1.0f32 / (cfg.batch_size * cfg.seq_len * cfg.vocab_size) as f32;
    let scale = Tensor::with_layout(vec![1, 1], mean_scale, Device::CPU, Layout::Contiguous, DType::F32);
    let scale_id = graph.add_parameter(scale);
    let loss = graph.add_node_of_type(NodeType::Mul, vec![total, scale_id]);
    graph.add_node_of_type(NodeType::Output, vec![loss]);

    (graph, handles.param_ids)
}

#[allow(dead_code)]
pub fn build_trainer(cfg: &MiniFluxConfig) -> TrainerV2 {
    let (graph, param_ids) = build_training_graph(cfg);
    let optim = AdamW::new(param_ids.len(), 0.02, 0.9, 0.999, 1e-8, 0.0);
    TrainerV2::new(graph, param_ids, optim)
}

#[allow(dead_code)]
pub fn build_language_trainer(cfg: &MiniFluxConfig) -> TrainerV2 {
    let (graph, param_ids) = build_language_training_graph(cfg);
    let optim = AdamW::new(param_ids.len(), 0.015, 0.9, 0.999, 1e-8, 0.0);
    TrainerV2::new(graph, param_ids, optim)
}
