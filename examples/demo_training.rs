use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::optim::adamw::AdamW;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer_v2::TrainerV2;

fn tensor_from_vec(values: &[f32]) -> Tensor {
    let rows = values.len();
    let mut t = Tensor::with_layout(
        vec![rows, 1],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for (i, v) in values.iter().enumerate() {
        t.data[i] = *v;
    }
    t
}

fn filled_tensor(shape: &[usize], value: f32) -> Tensor {
    let mut t = Tensor::with_layout(
        shape.to_vec(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for v in t.data.iter_mut() {
        *v = value;
    }
    t
}

fn build_graph(x: &Tensor) -> (GraphBuilder, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.parameter(filled_tensor(&[1, 1], 0.5));
    let b_id = gb.parameter(filled_tensor(&[1], -0.5));

    let y_true_values: Vec<f32> = x.data.iter().map(|v| 2.0 * v + 3.0).collect();
    let y_true_id = gb.parameter(tensor_from_vec(&y_true_values));

    let pred_id = gb.linear(x_id, w_id, Some(b_id));
    let diff_id = gb.sub(pred_id, y_true_id);
    let loss_id = gb.mul(diff_id, diff_id);
    gb.output(loss_id);

    (gb, w_id, b_id)
}

fn main() {
    let x_values = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = tensor_from_vec(&x_values);
    let (gb, w_id, b_id) = build_graph(&x);

    let graph = gb.build();
    let param_ids = vec![w_id, b_id];
    let optim = AdamW::new(param_ids.len(), 0.1, 0.9, 0.999, 1e-8, 0.0);
    let mut trainer = TrainerV2::new(graph, param_ids.clone(), optim);

    for step in 0..200 {
        let outputs = trainer.train_step(vec![x.clone()]);
        let loss = outputs[0].data[0];
        if step % 10 == 0 {
            println!("Paso {:3}: pérdida = {:.6}", step, loss);
        }
    }

    let final_w = trainer.graph.nodes[param_ids[0]]
        .output
        .as_ref()
        .expect("falta tensor w")
        .data[0];
    let final_b = trainer.graph.nodes[param_ids[1]]
        .output
        .as_ref()
        .expect("falta tensor b")
        .data[0];

    println!("Entrenamiento terminado. w ≈ {:.4}, b ≈ {:.4}", final_w, final_b);
}
