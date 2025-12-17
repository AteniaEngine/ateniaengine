use crate::amg::builder::GraphBuilder;
use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;
use crate::tensor::{Device, DType, Layout, Tensor};

pub trait GraphLike {
    fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize;
    fn add_parameter(&mut self, tensor: Tensor) -> usize;
}

impl GraphLike for Graph {
    fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        Graph::add_node_of_type(self, node_type, inputs)
    }

    fn add_parameter(&mut self, tensor: Tensor) -> usize {
        Graph::add_parameter(self, tensor)
    }
}

impl GraphLike for GraphBuilder {
    fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        GraphBuilder::add_node_of_type(self, node_type, inputs)
    }

    fn add_parameter(&mut self, tensor: Tensor) -> usize {
        self.parameter(tensor)
    }
}

#[derive(Clone)]
pub struct MiniFluxConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub batch_size: usize,
}

pub struct MiniFluxHandles {
    pub token_input_id: usize,
    pub logits_id: usize,
    pub param_ids: Vec<usize>,
}

pub fn build_mini_flux(
    graph: &mut Graph,
    cfg: &MiniFluxConfig,
    token_input_id: usize,
) -> MiniFluxHandles {
    let (logits_id, param_ids) = build_mini_flux_internal(graph, cfg, token_input_id);
    MiniFluxHandles {
        token_input_id,
        logits_id,
        param_ids,
    }
}

pub fn build_mini_flux_language_model(
    graph: &mut GraphBuilder,
    cfg: &MiniFluxConfig,
    token_input_id: usize,
) -> (usize, Vec<usize>) {
    build_mini_flux_internal(graph, cfg, token_input_id)
}

pub fn build_language_training_graph(cfg: &MiniFluxConfig) -> (Graph, Vec<usize>) {
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let targets_id = gb.input();
    let (logits_id, param_ids) = build_mini_flux_language_model(&mut gb, cfg, tokens_id);
    let log_probs = gb.log_softmax(logits_id);
    let flat_targets = gb.reshape(
        targets_id,
        vec![(cfg.batch_size * cfg.seq_len) as isize],
    );
    let loss_id = gb.cross_entropy_loss(log_probs, flat_targets);
    gb.output(loss_id);
    (gb.build(), param_ids)
}

fn build_mini_flux_internal<G: GraphLike>(
    graph: &mut G,
    cfg: &MiniFluxConfig,
    token_input_id: usize,
) -> (usize, Vec<usize>) {
    let mut param_ids = Vec::new();

    let embedding_id = register_weight(
        graph,
        "embedding",
        cfg.vocab_size,
        cfg.d_model,
        &mut param_ids,
    );

    let embed_id = graph.add_node_of_type(NodeType::IndexSelect, vec![embedding_id, token_input_id]);

    let pos_param = graph.add_parameter(build_positional_table(cfg.seq_len, cfg.d_model));
    let with_pos = graph.add_node_of_type(NodeType::BroadcastAdd, vec![embed_id, pos_param]);

    let mut current_id = with_pos;
    for layer_idx in 0..cfg.num_layers {
        let prefix = format!("layer{}", layer_idx);
        current_id = build_block(graph, &prefix, current_id, cfg, &mut param_ids);
    }

    let w_out = register_weight(graph, "w_out", cfg.d_model, cfg.vocab_size, &mut param_ids);
    let logits_id = linear_3d(
        graph,
        current_id,
        w_out,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_model,
        cfg.vocab_size,
    );

    (logits_id, param_ids)
}

fn build_block<G: GraphLike>(
    graph: &mut G,
    prefix: &str,
    input_id: usize,
    cfg: &MiniFluxConfig,
    param_ids: &mut Vec<usize>,
) -> usize {
    let norm_in = graph.add_node_of_type(NodeType::RmsNorm, vec![input_id]);

    let w_q = register_weight(graph, &format!("{}_wq", prefix), cfg.d_model, cfg.d_model, param_ids);
    let w_k = register_weight(graph, &format!("{}_wk", prefix), cfg.d_model, cfg.d_model, param_ids);
    let w_v = register_weight(graph, &format!("{}_wv", prefix), cfg.d_model, cfg.d_model, param_ids);
    let w_o = register_weight(graph, &format!("{}_wo", prefix), cfg.d_model, cfg.d_model, param_ids);

    let q = linear_3d(
        graph,
        norm_in,
        w_q,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_model,
        cfg.d_model,
    );
    let k = linear_3d(
        graph,
        norm_in,
        w_k,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_model,
        cfg.d_model,
    );
    let v = linear_3d(
        graph,
        norm_in,
        w_v,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_model,
        cfg.d_model,
    );

    let k_t = graph.add_node_of_type(NodeType::TransposeLastTwo, vec![k]);
    let scores = graph.add_node_of_type(NodeType::BatchMatMul, vec![q, k_t]);

    let scale_val = 1.0f32 / (cfg.d_model as f32).sqrt();
    let scale_tensor = Tensor::with_layout(
        vec![cfg.batch_size, cfg.seq_len, cfg.seq_len],
        scale_val,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let scale_param = graph.add_parameter(scale_tensor);
    let scaled_scores = graph.add_node_of_type(NodeType::Mul, vec![scores, scale_param]);

    let attn_weights = graph.add_node_of_type(NodeType::Softmax, vec![scaled_scores]);
    let attn_out = graph.add_node_of_type(NodeType::BatchMatMul, vec![attn_weights, v]);
    let attn_proj = linear_3d(
        graph,
        attn_out,
        w_o,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_model,
        cfg.d_model,
    );

    let attn_res = graph.add_node_of_type(NodeType::Add, vec![input_id, attn_proj]);

    let norm_mlp_in = graph.add_node_of_type(NodeType::RmsNorm, vec![attn_res]);
    let w1 = register_weight(graph, &format!("{}_w1", prefix), cfg.d_model, cfg.d_hidden, param_ids);
    let w2 = register_weight(graph, &format!("{}_w2", prefix), cfg.d_hidden, cfg.d_model, param_ids);

    let hidden = linear_3d(
        graph,
        norm_mlp_in,
        w1,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_model,
        cfg.d_hidden,
    );
    let activated = graph.add_node_of_type(NodeType::SiLU, vec![hidden]);
    let mlp_out = linear_3d(
        graph,
        activated,
        w2,
        cfg.batch_size,
        cfg.seq_len,
        cfg.d_hidden,
        cfg.d_model,
    );

    graph.add_node_of_type(NodeType::Add, vec![attn_res, mlp_out])
}

fn linear_3d<G: GraphLike>(
    graph: &mut G,
    input_id: usize,
    weight_id: usize,
    batch: usize,
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) -> usize {
    let flat = graph.add_node_of_type(
        NodeType::Reshape {
            target: vec![(batch * seq) as isize, in_dim as isize],
        },
        vec![input_id],
    );
    let lin = graph.add_node_of_type(NodeType::Linear, vec![flat, weight_id]);
    graph.add_node_of_type(
        NodeType::Reshape {
            target: vec![batch as isize, seq as isize, out_dim as isize],
        },
        vec![lin],
    )
}

fn register_weight<G: GraphLike>(
    graph: &mut G,
    name_prefix: &str,
    rows: usize,
    cols: usize,
    param_ids: &mut Vec<usize>,
) -> usize {
    let mut tensor = Tensor::with_layout(
        vec![rows, cols],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let mut hash = 0u32;
    for ch in name_prefix.chars() {
        hash = hash.wrapping_mul(31).wrapping_add(ch as u32);
    }
    let seed = (hash % 4096) as f32 / 4096.0;
    let scale = 0.1f32;
    for (idx, value) in tensor.data.iter_mut().enumerate() {
        let angle = (idx as f32 + seed) * 0.37;
        *value = angle.sin() * scale;
    }
    tensor.strides = Tensor::compute_strides(&tensor.shape, &tensor.layout);
    let id = graph.add_parameter(tensor);
    param_ids.push(id);
    id
}

fn build_positional_table(seq_len: usize, dim: usize) -> Tensor {
    let mut tensor = Tensor::with_layout(
        vec![1, seq_len, dim],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for i in 0..seq_len {
        for j in 0..dim {
            let idx = i * dim + j;
            tensor.data[idx] = (i as f32) / ((j + 1) as f32);
        }
    }
    tensor.strides = Tensor::compute_strides(&tensor.shape, &tensor.layout);
    tensor
}
