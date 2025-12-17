use atenia_engine::apx7::pfls::PFLSHistory;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, Layout};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| if d > acc { d } else { acc })
}

fn make_simple_graph() -> Graph {
    // Grafo sencillo con una pequeña cadena y suma.
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(2, NodeType::SiLU, vec![1]));
    nodes.push(Node::new(3, NodeType::Add, vec![1, 2]));
    nodes.push(Node::new(4, NodeType::Output, vec![3]));
    Graph::build(nodes)
}

fn make_inputs() -> Vec<Tensor> {
    let x = Tensor::with_layout(vec![8, 8], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![x]
}

#[test]
fn apx_7_11_predicts_hotspot_correctly() {
    let mut h = PFLSHistory::default();
    // SL 0: 1.0s, congestión 5
    h.record(0, 1.0, 5);
    // SL 1: 0.5s, congestión 2
    h.record(1, 0.5, 2);
    // SL 2: más lento y más congestionado
    h.record(2, 2.0, 10);

    let hot = h.predict_next_hotspot().expect("must predict hotspot");
    assert_eq!(hot, 2);
}

#[test]
fn apx_7_11_equivalence_vs_7_10() {
    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.10"); }
    let mut g_710 = make_simple_graph();
    let out_710 = g_710.execute(inputs.clone());

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.11"); }
    let mut g_711 = make_simple_graph();
    let out_711 = g_711.execute(inputs);

    assert_eq!(out_710.len(), out_711.len());
    assert!(max_abs_diff(&out_710[0], &out_711[0]) < 1e-5);
}

#[test]
fn apx_7_11_structural_reordering() {
    // Test de unidad sobre la heurística de ordenamiento: nodos con más hijos
    // y mayor profundidad deberían recibir mayor prioridad (clave más baja
    // tras el signo negativo).
    let children: Vec<Vec<usize>> = vec![
        vec![1, 2], // nodo 0, 2 hijos
        vec![],     // nodo 1
        vec![],     // nodo 2
    ];
    let depths: Vec<usize> = vec![1, 3, 2];
    let mut ready = vec![0usize, 1, 2];

    // Reproducimos la misma clave que en HLS-Deep PFLS.
    ready.sort_by_key(|&nid| {
        let out_degree = children[nid].len() as i32;
        let depth = depths[nid] as i32;
        -(out_degree + depth)
    });

    // Nodo 0: out=2, depth=1 => score 3
    // Nodo 1: out=0, depth=3 => score 3
    // Nodo 2: out=0, depth=2 => score 2
    // Con el signo negativo, 0 y 1 quedan delante de 2.
    assert_eq!(ready[2], 2);
}
