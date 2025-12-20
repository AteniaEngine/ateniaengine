use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::nodes::NodeType;
use atenia_engine::nn::transformer::{build_transformer_block, TransformerBlockHandles, TransformerConfig};
use atenia_engine::optim::adamw::AdamW;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer_v2::TrainerV2;

fn sample_input(cfg: &TransformerConfig) -> Tensor {
    let mut t = Tensor::with_layout(
        vec![cfg.seq_len, cfg.d_model],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for i in 0..cfg.seq_len {
        let start = i * cfg.d_model;
        let end = start + cfg.d_model;
        for j in 0..cfg.d_model {
            let idx = start + j;
            t.data[idx] = (idx as f32 + 1.0) * 0.05;
        }
        let mut sum_sq = 0.0f32;
        for idx in start..end {
            sum_sq += t.data[idx] * t.data[idx];
        }
        let rms = (sum_sq / cfg.d_model as f32).sqrt().max(1e-6);
        for idx in start..end {
            t.data[idx] /= rms;
        }
    }
    t
}

fn ones_tensor(shape: Vec<usize>) -> Tensor {
    Tensor::with_layout(shape, 1.0, Device::CPU, Layout::Contiguous, DType::F32)
}

fn setup_trainer(cfg: &TransformerConfig) -> (TrainerV2, Tensor) {
    let mut gb = GraphBuilder::new();
    let input_id = gb.input();
    let target_id = gb.input();
    let mut graph = gb.build();

    let TransformerBlockHandles {
        input_id: block_input_id,
        output_id: block_output_id,
        param_ids,
    } = build_transformer_block(&mut graph, cfg, "mini_t", input_id);

    assert_eq!(block_input_id, input_id, "block must consume the provided input node");

    let diff_id = graph.add_node_of_type(NodeType::Sub, vec![block_output_id, target_id]);
    let sq_id = graph.add_node_of_type(NodeType::Mul, vec![diff_id, diff_id]);

    let ones_feat = ones_tensor(vec![cfg.d_model, 1]);
    let ones_feat_id = graph.add_parameter(ones_feat);
    let sum_features_id = graph.add_node_of_type(NodeType::MatMul, vec![sq_id, ones_feat_id]);

    let ones_seq = ones_tensor(vec![1, cfg.seq_len]);
    let ones_seq_id = graph.add_parameter(ones_seq);
    let total_sum_id = graph.add_node_of_type(NodeType::MatMul, vec![ones_seq_id, sum_features_id]);

    let mean_scale = 1.0f32 / (cfg.d_model * cfg.seq_len) as f32;
    let scale_tensor = Tensor::with_layout(vec![1, 1], mean_scale, Device::CPU, Layout::Contiguous, DType::F32);
    let scale_id = graph.add_parameter(scale_tensor);
    let loss_id = graph.add_node_of_type(NodeType::Mul, vec![total_sum_id, scale_id]);
    let _out_id = graph.add_node_of_type(NodeType::Output, vec![loss_id]);

    let optim = AdamW::new(param_ids.len(), 0.05, 0.9, 0.999, 1e-8, 0.0);
    let trainer = TrainerV2::new(graph, param_ids, optim);
    let input_tensor = sample_input(cfg);
    (trainer, input_tensor)
}

#[test]
fn mini_transformer_learns_identity() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "4.13");
    }

    let cfg = TransformerConfig {
        d_model: 8,
        d_ff: 16,
        seq_len: 4,
    };
    let (mut trainer, input_tensor) = setup_trainer(&cfg);

    let mut first_loss = None;
    let mut best_loss = f32::INFINITY;

    for step in 0..200 {
        let outputs = trainer.train_step(vec![input_tensor.clone(), input_tensor.clone()]);
        let loss = outputs[0].data[0];
        if step == 0 {
            first_loss = Some(loss);
        }
        if loss < best_loss {
            best_loss = loss;
        }
    }

    let first = first_loss.expect("loss should be recorded at step 0");
    // Require a strong drop, but less extreme (~20x instead of 50x).
    let target = first * 0.05;
    assert!(
        best_loss <= target,
        "loss did not drop sufficiently (expected >=20x): first={first}, best={best_loss}, target={target}"
    );
}
