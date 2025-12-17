use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, Layout};

fn make_branch_graph() -> Graph {
    // Grafo con dos ramas que se juntan en un Output.
    // 0: Input X
    // 1: Pesos A (rama barata)
    // 2: Pesos B (rama cara)
    // 3: Linear(X, A)   -- barato
    // 4: Linear(X, B)   -- caro
    // 5: Add(3, 4)
    // 6: Output(5)
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::Parameter, vec![]));
    nodes.push(Node::new(2, NodeType::Parameter, vec![]));
    nodes.push(Node::new(3, NodeType::Linear, vec![0, 1]));
    nodes.push(Node::new(4, NodeType::Linear, vec![0, 2]));
    nodes.push(Node::new(5, NodeType::Add, vec![3, 4]));
    nodes.push(Node::new(6, NodeType::Output, vec![5]));

    Graph::build(nodes)
}

fn make_branch_inputs() -> Vec<Tensor> {
    let x = Tensor::with_layout(vec![8, 8], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    let wa = Tensor::with_layout(vec![8, 8], 0.5, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    let wb = Tensor::with_layout(vec![8, 8], 0.5, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![x, wa, wb]
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| if d > acc { d } else { acc })
}

#[test]
fn apx_7_6_hpge_equivalence_with_sequential() {
    let mut inputs = make_branch_inputs();
    let x = inputs.remove(0);
    let wa = inputs.remove(0);
    let wb = inputs.remove(0);

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.4"); }
    let mut g_seq = make_branch_graph();
    // 1 y 2 son parámetros en make_branch_graph.
    g_seq.nodes[1].set_output(wa.clone());
    g_seq.nodes[2].set_output(wb.clone());
    let out_seq = g_seq.execute(vec![x.clone()]);

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.6"); }
    let mut g_prio = make_branch_graph();
    g_prio.nodes[1].set_output(wa);
    g_prio.nodes[2].set_output(wb);
    let out_prio = g_prio.execute(vec![x]);

    assert_eq!(out_seq.len(), out_prio.len());
    assert!(max_abs_diff(&out_seq[0], &out_prio[0]) < 1e-5);
}

#[test]
fn apx_7_6_hpge_performance_sanity() {
    let mut inputs = make_branch_inputs();
    let x = inputs.remove(0);
    let wa = inputs.remove(0);
    let wb = inputs.remove(0);

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.4"); }
    let mut g_seq = make_branch_graph();
    g_seq.nodes[1].set_output(wa.clone());
    g_seq.nodes[2].set_output(wb.clone());
    let t_seq = now_ms(|| g_seq.execute(vec![x.clone()]));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.6"); }
    let mut g_prio = make_branch_graph();
    g_prio.nodes[1].set_output(wa);
    g_prio.nodes[2].set_output(wb);
    let t_prio = now_ms(|| g_prio.execute(vec![x]));

    // En debug dejamos margen amplio: no debe ser mucho peor.
    assert!(t_prio <= t_seq * 1.2);
}
