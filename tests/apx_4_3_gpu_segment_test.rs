#[test]
fn test_gpu_segment_builds() {
    unsafe {
        std::env::set_var("APX_TRACE", "1");
    }

    use atenia_engine::{
        apx4_3::gpu_plan::GpuPlan,
        amg::builder::GraphBuilder,
        nn::mini_flux::MiniFluxConfig,
    };

    let _cfg = MiniFluxConfig {
        vocab_size: 16,
        seq_len: 4,
        d_model: 32,
        d_hidden: 64,
        num_layers: 1,
        batch_size: 1,
    };

    let mut gb = GraphBuilder::new();
    let tokens = gb.input();
    gb.output(tokens);
    let graph = gb.build();
    // Extraer node_types reales del graph
    let node_types: Vec<_> = graph.nodes.iter().map(|n| n.node_type.clone()).collect();

    let plan = GpuPlan::build(&node_types);

    eprintln!("GPU plan: {:?}", plan);
}
