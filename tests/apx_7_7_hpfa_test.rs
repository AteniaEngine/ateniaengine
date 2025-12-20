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
    // 0: Input
    // 1: SiLU(Input)
    // 2: Output(1)
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(2, NodeType::Output, vec![1]));
    Graph::build(nodes)
}

fn make_inputs() -> Vec<Tensor> {
    let x = Tensor::with_layout(vec![4, 4], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![x]
}

#[test]
fn apx_7_7_hpfa_equivalence_with_sequential() {
    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.4"); }
    let mut g_seq = make_simple_graph();
    let out_seq = g_seq.execute(inputs.clone());

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.7"); }
    let mut g_hpfa = make_simple_graph();
    let out_hpfa = g_hpfa.execute(inputs);

    assert_eq!(out_seq.len(), out_hpfa.len());
    assert!(max_abs_diff(&out_seq[0], &out_hpfa[0]) < 1e-5);
}

#[test]
fn apx_7_7_hpfa_priority_bonus_structural() {
    use atenia_engine::apx6_10::global_fusion_selector;
    use atenia_engine::apx7::hpfa::FusionAffinity;

    // Simulate a node with high fusion affinity and verify that the
    // base+bonus combination increases priority.
    let fa = FusionAffinity {
        qkv_chain: true,
        attn_fusable: true,
        proj_fusable: true,
        hot_factor: 50.0,
    };

    let base_score = 10.0 * 0.4 + 5.0 * 0.3 + 100.0 * 0.1;
    let bonus = fa.fusion_bonus();
    let boosted_score = base_score + 0.2 * bonus;

    assert!(boosted_score > base_score);

    // Also, populate the global selector so wiring does not fail when
    // HPGE queries it in mode 7.7 (even though we do not execute the graph here).
    if let Ok(mut sel) = global_fusion_selector().lock() {
        sel.qkv_candidates.insert(1);
        sel.attn_candidates.insert(2);
        sel.proj_candidates.insert(3);
        sel.hist_profile.insert(1, 10.0);
    }
}

#[test]
fn apx_7_7_hpfa_performance_sanity_like_7_6() {
    let inputs = make_inputs();

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.4"); }
    let mut g_seq = make_simple_graph();
    let t_seq = now_ms(|| g_seq.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.7"); }
    let mut g_hpfa = make_simple_graph();
    let t_hpfa = now_ms(|| g_hpfa.execute(inputs));

    // In debug we allow a wide margin similar to 7.6.
    assert!(t_hpfa <= t_seq * 1.5);
}
