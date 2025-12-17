use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn make_tensor(shape: Vec<usize>, value: f32) -> Tensor {
    let mut t = Tensor::with_layout(
        shape,
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

#[test]
fn linear_backward_matches_manual_gradients() {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let b_id = gb.input();

    let lin_id = gb.linear(x_id, w_id, Some(b_id));
    let sq_id = gb.mul(lin_id, lin_id);
    let out_id = gb.output(sq_id);

    let mut graph = gb.build();

    if atenia_engine::apx_debug_enabled() {
        for (id, node) in graph.nodes.iter().enumerate() {
            eprintln!(
                "[DUMP] NODE {}: {:?} | inputs={:?}",
                id,
                node.node_type,
                node.inputs
            );
        }
    }

    let x = make_tensor(vec![1, 1], 1.0);
    let w = make_tensor(vec![1, 1], 2.0);
    let b = make_tensor(vec![1], 0.5);

    let outputs = graph.execute(vec![x.clone(), w.clone(), b.clone()]);
    assert_eq!(outputs.len(), 1);
    let loss_value = outputs[0].data[0];
    assert!((loss_value - 6.25).abs() < 1e-5);

    graph.backward(out_id);

    let x_grad = graph.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("x grad missing")[0];
    let w_grad = graph.nodes[w_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("w grad missing")[0];
    let b_grad = graph.nodes[b_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("b grad missing")[0];

    assert!((x_grad - 10.0).abs() < 1e-4);
    assert!((w_grad - 5.0).abs() < 1e-4);
    assert!((b_grad - 5.0).abs() < 1e-4);
}
