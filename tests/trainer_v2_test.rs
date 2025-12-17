use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::optim::adamw::AdamW;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer_v2::TrainerV2;

fn scalar(value: f32) -> Tensor {
    let mut t = Tensor::with_layout(vec![1, 1], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    t.data[0] = value;
    t
}

#[test]
fn trainer_v2_reduces_loss_on_linear_problem() {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let target_id = gb.input();
    let w_id = gb.parameter(scalar(0.5));
    let neg_one_id = gb.parameter(scalar(-1.0));

    let pred_id = gb.linear(x_id, w_id, None);
    let neg_target_id = gb.mul(target_id, neg_one_id);
    let diff_id = gb.add(pred_id, neg_target_id);
    let loss_id = gb.mul(diff_id, diff_id);
    let _out_id = gb.output(loss_id);

    let graph = gb.build();
    let param_ids = vec![w_id];
    let optim = AdamW::new(param_ids.len(), 0.1, 0.9, 0.999, 1e-8, 0.0);
    let mut trainer = TrainerV2::new(graph, param_ids, optim);

    let x = scalar(2.0);
    let target = scalar(6.0);

    let mut losses = Vec::new();
    for _ in 0..40 {
        let outputs = trainer.train_step(vec![x.clone(), target.clone()]);
        losses.push(outputs[0].data[0]);
    }

    let first = *losses.first().unwrap();
    let last = *losses.last().unwrap();
    assert!(last < first, "loss did not decrease: first={first}, last={last}");
}
