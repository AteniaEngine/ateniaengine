use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::NodeType;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn build_simple_mlp_graph() -> (Graph, Vec<usize>) {
    // x -> Linear(w1, b1) -> SiLU -> Linear(w2, b2) -> Output (mean squared loss vs target)
    let batch = 2usize;
    let in_dim = 3usize;
    let hidden = 4usize;
    let out_dim = 2usize;

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let y_id = gb.input();
    let mut graph = gb.build();

    // Parameters
    let w1 = Tensor::with_layout(
        vec![in_dim, hidden],
        0.1,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    // Bias must be 1D [out_features] for nn::linear
    let b1 = Tensor::with_layout(
        vec![hidden],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let w2 = Tensor::with_layout(
        vec![hidden, out_dim],
        0.1,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let b2 = Tensor::with_layout(
        vec![out_dim],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );

    let w1_id = graph.add_parameter(w1);
    let b1_id = graph.add_parameter(b1);
    let w2_id = graph.add_parameter(w2);
    let b2_id = graph.add_parameter(b2);

    // Forward
    let lin1 = graph.add_node_of_type(NodeType::Linear, vec![x_id, w1_id, b1_id]);
    let act = graph.add_node_of_type(NodeType::SiLU, vec![lin1]);
    let lin2 = graph.add_node_of_type(NodeType::Linear, vec![act, w2_id, b2_id]);

    // Mean squared error: loss = mean((lin2 - y)^2)
    let diff = graph.add_node_of_type(NodeType::Sub, vec![lin2, y_id]);
    let sq = graph.add_node_of_type(NodeType::Mul, vec![diff, diff]);

    let ones = Tensor::with_layout(
        vec![1, batch * out_dim],
        1.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let ones_id = graph.add_parameter(ones);
    // Flatten sq to [batch*out_dim, 1] so that [1, batch*out_dim] x [batch*out_dim, 1] -> [1,1]
    let sq_flat = graph.add_node_of_type(
        NodeType::Reshape { target: vec![(batch * out_dim) as isize, 1] },
        vec![sq],
    );
    let total = graph.add_node_of_type(NodeType::MatMul, vec![ones_id, sq_flat]);

    let scale_val = 1.0f32 / (batch * out_dim) as f32;
    let scale = Tensor::with_layout(vec![1, 1], scale_val, Device::CPU, Layout::Contiguous, DType::F32);
    let scale_id = graph.add_parameter(scale);
    let loss = graph.add_node_of_type(NodeType::Mul, vec![total, scale_id]);
    graph.add_node_of_type(NodeType::Output, vec![loss]);

    let params = vec![w1_id, b1_id, w2_id, b2_id];
    (graph, params)
}

fn build_inputs() -> (Tensor, Tensor) {
    let batch = 2usize;
    let in_dim = 3usize;
    let out_dim = 2usize;

    let mut x = Tensor::with_layout(
        vec![batch, in_dim],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let mut y = Tensor::with_layout(
        vec![batch, out_dim],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );

    for b in 0..batch {
        for i in 0..in_dim {
            x.data[b * in_dim + i] = (b as f32) + (i as f32) * 0.1;
        }
        for o in 0..out_dim {
            y.data[b * out_dim + o] = ((b + o) as f32) * 0.2;
        }
    }

    (x, y)
}

fn collect_param_grads(graph: &Graph, param_ids: &[usize]) -> Vec<Vec<f32>> {
    param_ids
        .iter()
        .map(|&pid| {
            graph.nodes[pid]
                .output
                .as_ref()
                .and_then(|t| t.grad.as_ref())
                .map(|g| g.clone())
                .unwrap_or_else(|| vec![0.0; graph.nodes[pid].output.as_ref().unwrap().data.len()])
        })
        .collect()
}

fn approx_equal(a: &[f32], b: &[f32], atol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > atol {
            return false;
        }
    }
    true
}

#[test]
fn apx_parallel_backward_matches_sequential() {
    let (mut g_seq, params) = build_simple_mlp_graph();
    let mut g_par = g_seq.clone();

    let (x, y) = build_inputs();

    // Forward
    let outputs_seq = g_seq.execute(vec![x.clone(), y.clone()]);
    let outputs_par = g_par.execute(vec![x, y]);
    assert_eq!(outputs_seq.len(), 1);
    assert_eq!(outputs_par.len(), 1);

    let loss_id_seq = g_seq.last_output_id();
    let loss_id_par = g_par.last_output_id();
    assert_eq!(loss_id_seq, loss_id_par);

    // Backward sequential
    g_seq.backward_sequential(loss_id_seq);

    // Backward parallel (default APX 2.0)
    g_par.backward(loss_id_par);

    let grads_seq = collect_param_grads(&g_seq, &params);
    let grads_par = collect_param_grads(&g_par, &params);

    for (i, (gs, gp)) in grads_seq.iter().zip(grads_par.iter()).enumerate() {
        assert!(
            approx_equal(gs, gp, 1e-6),
            "gradient mismatch for param {}: seq={:?} par={:?}",
            i,
            gs,
            gp
        );
    }
}
