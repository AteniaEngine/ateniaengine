use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;

#[test]
fn test_qkv_fusion_detects() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "4.14");
    }

    // Build a simple graph with 3 linear layers that share X.
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

    let g: Graph = gb.build();

    // Verify that at least one FusedQKV was registered in fused_ops.
    let has_qkv = g
        .fused_ops
        .values()
        .any(|fop| matches!(fop, atenia_engine::apx4_13::fusion_engine::FusedOp::FusedQKV { .. }));

    assert!(has_qkv, "Expected at least one FusedQKV op detected in fused_ops");
}
