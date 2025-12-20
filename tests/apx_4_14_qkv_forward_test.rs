use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::{Graph, FusedOutput};
use atenia_engine::tensor::{Tensor, Device, DType, Layout};

fn build_qkv_graph() -> Graph {
    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let wq = gb.input();
    let wk = gb.input();
    let wv = gb.input();

    let q = gb.linear(x, wq, None);
    let k = gb.linear(x, wk, None);
    let v = gb.linear(x, wv, None);
    gb.output(q);
    gb.output(k);
    gb.output(v);

    gb.build()
}

#[test]
fn test_qkv_fusion_forward_matches_naive() {
    // Build the "naive" graph without fusion (default mode 2.5).
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "2.5");
    }
    let mut g_naive = build_qkv_graph();

    // Deterministic inputs
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;

    let x = Tensor::with_layout(vec![m, k], 1.0, Device::CPU, Layout::Contiguous, DType::F32);
    let wq = Tensor::with_layout(vec![k, n], 0.5, Device::CPU, Layout::Contiguous, DType::F32);
    let wk = Tensor::with_layout(vec![k, n], 0.7, Device::CPU, Layout::Contiguous, DType::F32);
    let wv = Tensor::with_layout(vec![k, n], -0.3, Device::CPU, Layout::Contiguous, DType::F32);

    let out_naive = g_naive.execute(vec![x.clone(), wq.clone(), wk.clone(), wv.clone()]);
    assert_eq!(out_naive.len(), 3);

    // Same graph but with APX 4.14 enabled (QKV fusion).
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "4.14");
    }
    let mut g_fused = build_qkv_graph();
    let _ = g_fused.execute(vec![x, wq, wk, wv]);

    // Take the fused QKV output registered in fused_outputs.
    let fused_entry = g_fused
        .fused_outputs
        .values()
        .next()
        .expect("Expected a fused QKV output");

    let (q, k, v) = match fused_entry {
        FusedOutput::QKV { q, k, v } => (q, k, v),
        other => panic!("Expected FusedOutput::QKV, got {other:?}"),
    };

    let fused_list = [q, k, v];

    for (naive, fused) in out_naive.iter().zip(fused_list.iter()) {
        assert_eq!(naive.shape, fused.shape);
        assert_eq!(naive.data.len(), fused.data.len());
        for (a, b) in naive.data.iter().zip(fused.data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
