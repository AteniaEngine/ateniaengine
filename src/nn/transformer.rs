use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;
use crate::tensor::{Device, DType, Layout, Tensor};

pub struct TransformerConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub seq_len: usize,
}

pub struct TransformerBlockHandles {
    pub input_id: usize,
    pub output_id: usize,
    pub param_ids: Vec<usize>,
}

pub fn build_transformer_block(
    graph: &mut Graph,
    cfg: &TransformerConfig,
    prefix: &str,
    input_id: usize,
) -> TransformerBlockHandles {
    let mut param_ids = Vec::new();

    let w_q = register_weight(graph, prefix, "w_q", cfg.d_model, cfg.d_model, &mut param_ids);
    let w_k = register_weight(graph, prefix, "w_k", cfg.d_model, cfg.d_model, &mut param_ids);
    let w_v = register_weight(graph, prefix, "w_v", cfg.d_model, cfg.d_model, &mut param_ids);
    let w_o = register_weight(graph, prefix, "w_o", cfg.d_model, cfg.d_model, &mut param_ids);
    let w1 = register_weight(graph, prefix, "w1", cfg.d_model, cfg.d_ff, &mut param_ids);
    let w2 = register_weight(graph, prefix, "w2", cfg.d_ff, cfg.d_model, &mut param_ids);

    let input_norm_id = graph.add_node_of_type(NodeType::RmsNorm, vec![input_id]);

    let q_id = graph.add_node_of_type(NodeType::Linear, vec![input_norm_id, w_q]);
    let k_id = graph.add_node_of_type(NodeType::Linear, vec![input_norm_id, w_k]);
    let v_id = graph.add_node_of_type(NodeType::Linear, vec![input_norm_id, w_v]);

    let k_t_id = graph.add_node_of_type(NodeType::Transpose2D, vec![k_id]);
    let scores_id = graph.add_node_of_type(NodeType::MatMul, vec![q_id, k_t_id]);

    let scale = 1.0f32 / (cfg.d_model as f32).sqrt();
    let scale_tensor = Tensor::with_layout(
        vec![cfg.seq_len, cfg.seq_len],
        scale,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let scale_param = graph.add_parameter(scale_tensor);
    let scaled_scores_id = graph.add_node_of_type(NodeType::Mul, vec![scores_id, scale_param]);

    let attn_weights_id = graph.add_node_of_type(NodeType::Softmax, vec![scaled_scores_id]);
    let attn_out_id = graph.add_node_of_type(NodeType::MatMul, vec![attn_weights_id, v_id]);
    let attn_proj_id = graph.add_node_of_type(NodeType::Linear, vec![attn_out_id, w_o]);

    let x1_id = graph.add_node_of_type(NodeType::Add, vec![input_id, attn_proj_id]);
    let x1_norm_id = graph.add_node_of_type(NodeType::RmsNorm, vec![x1_id]);

    let hidden_id = graph.add_node_of_type(NodeType::Linear, vec![x1_norm_id, w1]);
    let hidden_silu_id = graph.add_node_of_type(NodeType::SiLU, vec![hidden_id]);
    let mlp_out_id = graph.add_node_of_type(NodeType::Linear, vec![hidden_silu_id, w2]);

    let output_id = graph.add_node_of_type(NodeType::Add, vec![x1_id, mlp_out_id]);

    TransformerBlockHandles {
        input_id,
        output_id,
        param_ids,
    }
}

fn register_weight(
    graph: &mut Graph,
    prefix: &str,
    name: &str,
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
    let seed = deterministic_seed(prefix, name);
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

fn deterministic_seed(prefix: &str, name: &str) -> f32 {
    let mut hash = 0u32;
    for ch in prefix.chars().chain(name.chars()) {
        hash = hash.wrapping_mul(31).wrapping_add(ch as u32);
    }
    (hash % 4096) as f32 / 4096.0
}
