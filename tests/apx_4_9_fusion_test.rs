use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::nodes::{NodeType, ActType};
use atenia_engine::tensor::{Tensor, Device};

#[test]
fn test_apx_4_9_detecta_y_fusiona_cadena() {
    let mut gb = GraphBuilder::new();

    let x = gb.input();
    let w1 = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b1 = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let w2 = gb.parameter(Tensor::randn(&[32, 16], Device::CPU));
    let b2 = gb.parameter(Tensor::randn(&[16], Device::CPU));

    let l1 = gb.linear(x, w1, Some(b1));
    let a1 = gb.silu(l1);
    let a2 = gb.silu(a1);
    let l2 = gb.linear(a2, w2, Some(b2));

    gb.output(l2);
    let g = gb.build();

    // There must be exactly one FusedLinearActivationChain node.
    let mut chain_node_id = None;
    for (i, n) in g.nodes.iter().enumerate() {
        if let NodeType::FusedLinearActivationChain(acts) = &n.node_type {
            chain_node_id = Some(i);
            // It must capture exactly two SiLU activations.
            assert_eq!(acts.len(), 2);
            assert!(acts.iter().all(|a| *a == ActType::SiLU));
        }
    }

    let chain_id = chain_node_id.expect("no FusedLinearActivationChain node found");

    // Intermediate nodes must be NoOp.
    assert!(matches!(g.nodes[l1].node_type, NodeType::NoOp));
    assert!(matches!(g.nodes[a1].node_type, NodeType::NoOp));
    assert!(matches!(g.nodes[a2].node_type, NodeType::NoOp));

    // The fused node must have the expected inputs: [x, w1, b1, w2, b2].
    let chain_inputs = &g.nodes[chain_id].inputs;

    assert_eq!(chain_inputs.len(), 5);
    assert_eq!(chain_inputs[0], x);
    assert_eq!(chain_inputs[1], w1);
    assert_eq!(chain_inputs[2], b1);
    assert_eq!(chain_inputs[3], w2);
    assert_eq!(chain_inputs[4], b2);
}

#[test]
fn test_apx_4_9_equivalencia_numerica() {
    use atenia_engine::nn::linear::linear as linear_op;
    use atenia_engine::nn::activations::silu;

    let x = Tensor::randn(&[1, 32], Device::CPU);
    let w1 = Tensor::randn(&[32, 32], Device::CPU);
    let b1 = Tensor::randn(&[32], Device::CPU);
    let w2 = Tensor::randn(&[32, 16], Device::CPU);
    let b2 = Tensor::randn(&[16], Device::CPU);

    // Ruta normal: linear -> silu -> silu -> linear.
    let h1 = linear_op(&x, &w1, Some(&b1));
    let h2 = silu(&h1);
    let h3 = silu(&h2);
    let ref_out = linear_op(&h3, &w2, Some(&b2));

    // Graph path with APX 4.9 fusion.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w1_id = gb.parameter(w1.clone());
    let b1_id = gb.parameter(b1.clone());
    let w2_id = gb.parameter(w2.clone());
    let b2_id = gb.parameter(b2.clone());

    let l1 = gb.linear(x_id, w1_id, Some(b1_id));
    let a1 = gb.silu(l1);
    let a2 = gb.silu(a1);
    let l2 = gb.linear(a2, w2_id, Some(b2_id));

    gb.output(l2);
    let mut g = gb.build();

    let out = g.execute(vec![x.clone()]);
    assert_eq!(out.len(), 1);
    let fused_out = &out[0];

    assert_eq!(ref_out.shape, fused_out.shape);
    assert_eq!(ref_out.data.len(), fused_out.data.len());

    let mut max_diff = 0.0f32;
    for i in 0..ref_out.data.len() {
        let d = (ref_out.data[i] - fused_out.data[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("[APX 4.9] max_diff = {}", max_diff);
    assert!(max_diff < 1e-3);
}
