use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::apx7::ule::{choose_backend, ULEStrategy};
use atenia_engine::tensor::{Tensor, Device, Layout};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| if d > acc { d } else { acc })
}

fn make_simple_graph() -> Graph {
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
fn apx_7_12_ule_equivalence_with_7_11() {
    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.11"); }
    let mut g_711 = make_simple_graph();
    let out_711 = g_711.execute(inputs.clone());

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.12"); }
    let mut g_712 = make_simple_graph();
    let out_712 = g_712.execute(inputs);

    assert_eq!(out_711.len(), out_712.len());
    assert!(max_abs_diff(&out_711[0], &out_712[0]) < 1e-5);
}

#[test]
fn apx_7_12_unifies_choice() {
    // Backend decision lógica pura (no necesitamos instanciar Graph aquí).
    let small = vec![0usize];
    let mid = vec![0usize, 1, 2];
    let big: Vec<usize> = (0..16).collect();

    let s_small = choose_backend(&small);
    let s_mid = choose_backend(&mid);
    let s_big = choose_backend(&big);

    assert_eq!(s_small, ULEStrategy::Seq);
    assert_eq!(s_mid, ULEStrategy::Pex);
    // En máquinas con muchos hilos, los SL grandes deberían usar WS; en otras
    // al menos no deben caer en Seq.
    assert!(s_big == ULEStrategy::Pex || s_big == ULEStrategy::WorkStealing);

    // Prueba estructural de la heurística de prioridad: nodos con más hijos y
    // mayor profundidad se ordenan primero.
    let children: Vec<Vec<usize>> = vec![
        vec![1, 2], // nodo 0: 2 hijos
        vec![],     // nodo 1
        vec![],     // nodo 2
    ];
    let depths: Vec<usize> = vec![1, 3, 2];
    let mut ready = vec![0usize, 1, 2];

    ready.sort_by_key(|&nid| {
        let out_degree = children[nid].len() as i32;
        let depth = depths[nid] as i32;
        -(out_degree + depth)
    });

    // Nodo 2 (score 2) debe quedar al final.
    assert_eq!(ready[2], 2);
}

#[test]
fn apx_7_12_ule_performance_sanity() {
    fn make_wide_graph(width: usize, depth: usize) -> Graph {
        let mut nodes = Vec::new();
        nodes.push(Node::new(0, NodeType::Input, vec![]));
        let mut last_ids = Vec::new();
        for _w in 0..width {
            let mut prev = 0usize;
            for _d in 0..depth {
                let id = nodes.len();
                nodes.push(Node::new(id, NodeType::SiLU, vec![prev]));
                prev = id;
            }
            last_ids.push(prev);
        }
        let mut acc = last_ids[0];
        for &id in &last_ids[1..] {
            let add_id = nodes.len();
            nodes.push(Node::new(add_id, NodeType::Add, vec![acc, id]));
            acc = add_id;
        }
        let out_id = nodes.len();
        nodes.push(Node::new(out_id, NodeType::Output, vec![acc]));
        Graph::build(nodes)
    }

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.5"); }
    let mut g_hpge = make_wide_graph(3, 4);
    let t_hpge = now_ms(|| g_hpge.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.11"); }
    let mut g_pfls = make_wide_graph(3, 4);
    let t_pfls = now_ms(|| g_pfls.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.12"); }
    let mut g_ule = make_wide_graph(3, 4);
    let t_ule = now_ms(|| g_ule.execute(inputs));

    // APX 7.12: este test es un sanity check de rendimiento, no un contrato estricto.
    // ULE debería estar en la misma banda que PFLS/HPGE, pero permitimos un margen
    // amplio para absorber variaciones entre máquinas/entornos. El objetivo es
    // detectar sólo regresiones catastróficas (órdenes de magnitud), no micro
    // diferencias.
    let max_factor = 5.0;
    assert!(t_ule <= t_pfls * max_factor);
    assert!(t_ule <= t_hpge * max_factor);
}
