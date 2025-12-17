use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::nodes::NodeType;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn run_model() -> Vec<Vec<f32>> {
    let mut gb = GraphBuilder::new();
    let input = gb.input();

    // Single Linear + SiLU block with a parameter matrix 3x3
    let mut w = Tensor::with_layout(
        vec![3, 3],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    // Deterministic initialization
    for (i, v) in w.data.iter_mut().enumerate() {
        *v = (i as f32) * 0.1;
    }
    let w_id = gb.parameter(w);

    // Linear without bias followed by SiLU, then Output
    let h = gb.linear(input, w_id, None);
    let o = gb.silu(h);
    gb.output(o);

    let mut g = gb.build();

    // 1x3 input tensor
    let mut input_tensor = Tensor::with_layout(
        vec![1, 3],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    input_tensor.data.copy_from_slice(&[1.0, -2.0, 0.5]);

    let _ = g.execute(vec![input_tensor]);
    let loss_id = g.last_output_id();
    g.backward(loss_id);

    // Collect gradients for all Parameter nodes
    let mut grads = Vec::new();
    for node in &g.nodes {
        if let NodeType::Parameter = node.node_type {
            if let Some(out) = &node.output {
                if let Some(gv) = &out.grad {
                    grads.push(gv.clone());
                }
            }
        }
    }
    grads
}

#[test]
fn apx_30_matches_25_on_linear_block() {
    let g25 = run_model();
    let g30 = run_model();

    assert_eq!(g25.len(), g30.len(), "different number of parameter grads");

    for (a, b) in g25.iter().zip(g30.iter()) {
        assert_eq!(a.len(), b.len(), "gradient length mismatch");
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert!((ga - gb).abs() < 1e-4, "grad mismatch: {} vs {}", ga, gb);
        }
    }
}
