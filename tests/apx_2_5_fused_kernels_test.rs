use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn rand_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 31) as f32).sin()).collect()
}

#[test]
fn apx_2_5_gradients_match_reference() {
    let batch = 4usize;
    let in_dim = 16usize;
    let hidden = 32usize;
    let classes = 24usize;

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let t_id = gb.input();

    // Parameters as Tensors
    let w1 = Tensor::with_layout(
        vec![in_dim, hidden],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let w2 = Tensor::with_layout(
        vec![hidden, classes],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );

    let w1_id = gb.parameter(w1);
    let w2_id = gb.parameter(w2);

    // x -> Linear(w1) -> SiLU -> Linear(w2) -> LogSoftmax -> CrossEntropyLoss
    let l1 = gb.linear(x_id, w1_id, None);
    let a1 = gb.silu(l1);
    let l2 = gb.linear(a1, w2_id, None);
    let logp = gb.log_softmax(l2);
    let loss = gb.cross_entropy_loss(logp, t_id);
    gb.output(loss);

    let mut g_seq: Graph = gb.build();
    let mut g_par: Graph = g_seq.clone();

    // Inputs: x as Tensor [batch, in_dim], t as Tensor of integer targets [batch]
    let mut x = Tensor::with_layout(
        vec![batch, in_dim],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for (i, v) in rand_vec(batch * in_dim).into_iter().enumerate() {
        x.data[i] = v;
    }

    let mut t = Tensor::with_layout(
        vec![batch],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for b in 0..batch {
        t.data[b] = ((b * 7) % classes) as f32;
    }

    let out_seq = g_seq.execute(vec![x.clone(), t.clone()]);
    let out_par = g_par.execute(vec![x, t]);

    assert!((out_seq[0].data[0] - out_par[0].data[0]).abs() < 1e-6);

    let loss_id_seq = g_seq.last_output_id();
    let loss_id_par = g_par.last_output_id();

    g_seq.backward_sequential(loss_id_seq);
    g_par.backward(loss_id_par);

    for (n1, n2) in g_seq.nodes.iter().zip(g_par.nodes.iter()) {
        if let (Some(o1), Some(o2)) = (&n1.output, &n2.output) {
            match (&o1.grad, &o2.grad) {
                (Some(g1), Some(g2)) => {
                    if g1.len() == g2.len() {
                        let maxd = g1
                            .iter()
                            .zip(g2.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f32::max);
                        assert!(maxd < 1e-3, "gradient mismatch: {}", maxd);
                    }
                }
                _ => {}
            }
        }
    }
}
