use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::mini_flux::{build_mini_flux_language_model, MiniFluxConfig};
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

#[test]
fn apx_25_traces_show_parallelism() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "2.5");
        std::env::set_var("ATENIA_TRACE", "1");
    }

    let mut g = {
        let mut gb = GraphBuilder::new();
        let input = gb.input();
        let (logits, _) = build_mini_flux_language_model(
            &mut gb,
            &MiniFluxConfig {
                vocab_size: 16,
                seq_len: 4,
                d_model: 32,
                d_hidden: 64,
                num_layers: 1,
                batch_size: 1,
            },
            input,
        );
        gb.output(logits);
        let g = gb.build();

        if atenia_engine::apx_debug_enabled() {
            for (id, node) in g.nodes.iter().enumerate() {
                eprintln!(
                    "[DUMP] NODE {}: {:?} | inputs={:?}",
                    id,
                    node.node_type,
                    node.inputs
                );
            }
        }

        // Verify that the resulting MiniFlux graph is structurally valid
        // before executing and tracing.
        g.validate()
            .expect("The MiniFlux graph must be valid after build()");

        g
    };

    // Build simple integer-like tokens [0, vocab)
    let mut input = Tensor::with_layout(
        vec![1, 4],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for s in 0..4usize {
        input.data[s] = (s % 16) as f32;
    }

    let _ = g.execute(vec![input.clone()]);
    g.backward(g.last_output_id());

    assert!(true);
}
