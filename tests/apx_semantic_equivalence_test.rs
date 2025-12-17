use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::nn::mini_flux::{build_mini_flux_language_model, MiniFluxConfig};
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

/// Adaptive Execution without Semantic Drift (experimental).
///
/// This test verifies that adaptive execution mechanisms (APX 6.15 stability engine and
/// APX 7.x adaptive scheduling) do not introduce semantic drift or numerical divergence.
///
/// Constraints:
/// - No runtime noise.
/// - Fixed model (Mini-Flux) + fixed deterministic weights.
/// - Fixed deterministic input tensor.
/// - Forward-only.
///
/// The test fails immediately if semantic drift is detected.
#[test]
fn apx_semantic_equivalence_mini_flux_forward() {
    let cfg = MiniFluxConfig {
        vocab_size: 64,
        seq_len: 12,
        d_model: 48,
        d_hidden: 96,
        num_layers: 2,
        batch_size: 4,
    };

    let input = build_deterministic_tokens(cfg.batch_size, cfg.seq_len, cfg.vocab_size);

    let out_baseline = run_forward_with_mode(&cfg, &input, "4.19");
    let out_adaptive = run_forward_with_mode(&cfg, &input, "7.12");

    assert_eq!(out_baseline.shape, out_adaptive.shape, "output shape mismatch");
    assert_eq!(out_baseline.data.len(), out_adaptive.data.len(), "output length mismatch");

    let (max_diff, mean_diff) = diff_stats(&out_baseline.data, &out_adaptive.data);

    // Machine-precision level equivalence.
    // We prefer exact equality; if a future change introduces tiny rounding differences,
    // this epsilon still catches real semantic drift.
    let eps = 1e-6f32;
    assert!(
        max_diff == 0.0 || max_diff <= eps,
        "semantic drift detected: max_abs_diff={} mean_abs_diff={} (eps={})",
        max_diff,
        mean_diff,
        eps
    );
}

fn run_forward_with_mode(cfg: &MiniFluxConfig, input: &Tensor, apx_mode: &str) -> Tensor {
    // Test-only: select mode through env. (Required by this codebase.)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", apx_mode);
    }

    let mut graph = build_mini_flux_graph(cfg);

    let mut out = graph.execute(vec![input.clone()]);
    out.pop().expect("Mini-Flux graph must return a single output")
}

fn build_mini_flux_graph(cfg: &MiniFluxConfig) -> Graph {
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let (logits_id, _param_ids) = build_mini_flux_language_model(&mut gb, cfg, tokens_id);
    gb.output(logits_id);
    gb.build()
}

fn build_deterministic_tokens(batch: usize, seq_len: usize, vocab_size: usize) -> Tensor {
    let mut x = Tensor::with_layout(
        vec![batch, seq_len],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );

    for b in 0..batch {
        for s in 0..seq_len {
            let idx = b * seq_len + s;
            x.data[idx] = ((b + 3 * s) % vocab_size) as f32;
        }
    }

    x
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "diff_stats length mismatch");

    let mut max_d = 0.0f32;
    let mut sum_d = 0.0f64;

    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d as f64;
    }

    let mean = (sum_d / (a.len() as f64)) as f32;
    (max_d, mean)
}
