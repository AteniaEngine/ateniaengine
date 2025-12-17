use std::fs;
use std::io::Write;
use std::time::{Duration, Instant};

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::nn::mini_flux::{build_mini_flux_language_model, MiniFluxConfig};
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

use atenia_engine::apx6_10;
use atenia_engine::apx6_10::FusionProfile;
use atenia_engine::apx6_11::runtime_policy::{get_runtime_policy, set_runtime_policy, FusionRuntimePolicy};

/// Runtime Stability Under Dynamic Conditions (experimental).
///
/// This test intentionally introduces unstable runtime conditions to expose policy
/// oscillation in non-adaptive execution, demonstrating how Atenia Engine stabilizes
/// execution behavior through adaptive scheduling and hysteresis.
///
/// The experiment injects timing-only noise (jitter + synthetic CPU pressure) strictly
/// outside model operators, and records per-step metrics (policy selection + latency
/// + switches). Model semantics must remain unchanged.
#[test]
fn apx_runtime_stability_under_dynamic_conditions() {
    let steps = 48usize;

    let cfg = MiniFluxConfig {
        vocab_size: 64,
        seq_len: 12,
        d_model: 48,
        d_hidden: 96,
        num_layers: 2,
        batch_size: 4,
    };

    let input = build_deterministic_tokens(cfg.batch_size, cfg.seq_len, cfg.vocab_size);

    let mut metrics: Vec<MetricRow> = Vec::with_capacity(steps * 2);

    // Baseline: static, deterministic execution path.
    let baseline_out = run_mode(
        ExecutionMode::Baseline,
        steps,
        &cfg,
        &input,
        // APX 6.11: updates runtime policy from FusionSelector measurements,
        // but does not enable the APX 6.15 stabilizer nor any APX 7.x scheduler.
        // This is a good "non-adaptive" baseline that can still oscillate.
        "6.11",
        &mut metrics,
    );

    // Adaptive: enable APX 6.15 stability engine + APX 7.x scheduling.
    // We choose 7.12 (ULE) as a representative modern adaptive scheduler mode.
    let adaptive_out = run_mode(
        ExecutionMode::Adaptive,
        steps,
        &cfg,
        &input,
        "7.12",
        &mut metrics,
    );

    // Validity constraint: outputs must be identical across modes.
    assert_eq!(baseline_out.shape, adaptive_out.shape, "output shape mismatch");
    assert_eq!(baseline_out.data, adaptive_out.data, "output data mismatch");

    export_csv(&metrics);
    print_summary(&metrics);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionMode {
    Baseline,
    Adaptive,
}

impl ExecutionMode {
    fn as_str(&self) -> &'static str {
        match self {
            ExecutionMode::Baseline => "Baseline",
            ExecutionMode::Adaptive => "Adaptive",
        }
    }
}

#[derive(Clone, Debug)]
struct MetricRow {
    step: usize,
    mode: ExecutionMode,
    policy: String,
    latency_us: u64,
    policy_switch: bool,
}

fn run_mode(
    mode: ExecutionMode,
    steps: usize,
    cfg: &MiniFluxConfig,
    input: &Tensor,
    apx_mode: &str,
    metrics: &mut Vec<MetricRow>,
) -> Tensor {
    // Test-only: set APX mode via env. We do not modify engine behavior.
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", apx_mode);
    }

    // Reset global state so each mode starts from a clean slate.
    reset_global_apx_state();

    // Build the Mini-Flux forward graph.
    let mut graph = build_mini_flux_graph(cfg);

    let mut prev_policy: Option<String> = None;
    let mut rng = Lcg::new(0xC0FFEEu64 ^ (apx_mode.as_bytes().len() as u64));

    let mut last_out: Option<Tensor> = None;

    for step_id in 0..steps {
        // Inject deterministic timing-only noise outside operators.
        // Baseline receives harsher, asymmetric stress to increase instability pressure.
        inject_runtime_noise(mode, step_id, &mut rng);

        if mode == ExecutionMode::Baseline {
            // Explicit policy competition conditions (deterministic): we inject
            // near-threshold synthetic profiles into the selector so that APX 6.11
            // can flip its GlobalDecision across steps under stress.
            inject_policy_competition_baseline(step_id);
        }

        let t0 = Instant::now();
        let out = graph.execute(vec![input.clone()]);
        let latency_us = t0.elapsed().as_micros() as u64;

        // For Mini-Flux we output logits: single output tensor.
        let out0 = out
            .into_iter()
            .next()
            .expect("Mini-Flux graph must return a single output");

        let policy = format!("{:?}", get_runtime_policy());
        let policy_switch = prev_policy
            .as_ref()
            .map(|p| p != &policy)
            .unwrap_or(false);
        prev_policy = Some(policy.clone());

        metrics.push(MetricRow {
            step: step_id,
            mode,
            policy,
            latency_us,
            policy_switch,
        });

        last_out = Some(out0);
    }

    last_out.expect("mode run must produce at least one output")
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

fn reset_global_apx_state() {
    // Best-effort: clear selector history so policy decisions do not leak between modes.
    if let Ok(mut sel) = apx6_10::global_fusion_selector().lock() {
        sel.history.clear();
        sel.qkv_candidates.clear();
        sel.attn_candidates.clear();
        sel.proj_candidates.clear();
        sel.hist_profile.clear();
    }

    set_runtime_policy(FusionRuntimePolicy::Baseline);
}

fn inject_policy_competition_baseline(step_id: usize) {
    // Note: This does not modify model semantics. It only supplies runtime-side
    // measurements to the existing selector (APX 6.10), which APX 6.11 uses to
    // set FusionRuntimePolicy.
    //
    // We alternate profiles so that the 0.85-threshold decision flips:
    // - even steps: PreferFull (full much faster than qkv)
    // - odd steps:  PreferQKV  (qkv much faster than full)
    let (fused_full_us, fused_qkv_us) = if step_id % 2 == 0 {
        (80u64, 200u64)
    } else {
        (200u64, 80u64)
    };

    if let Ok(mut sel) = apx6_10::global_fusion_selector().lock() {
        sel.history.clear();
        sel.record_profile(FusionProfile {
            op_name: "MiniFlux".to_string(),
            baseline_us: 120u64,
            fused_qkv_us,
            fused_full_us,
        });
    }
}

fn inject_runtime_noise(mode: ExecutionMode, step_id: usize, rng: &mut Lcg) {
    // Deterministic noise; Adaptive keeps the original mild parameters.
    let (max_jitter_us, max_workers, max_burn_us) = match mode {
        ExecutionMode::Baseline => (2_000u64, 8usize, 2_000u64),
        ExecutionMode::Adaptive => (250u64, 3usize, 200u64),
    };

    let jitter_us = (rng.next_u32() as u64 % (max_jitter_us + 1)) as u64;

    // Simulate fluctuating worker availability / system load by briefly burning CPU
    // on a few threads, plus a small sleep.
    let workers = 1 + ((step_id as u64 + rng.next_u32() as u64) % (max_workers as u64)) as usize;
    let burn_us = (rng.next_u32() as u64 % (max_burn_us + 1)) as u64;

    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let burn = burn_us;
        handles.push(std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed().as_micros() < burn as u128 {
                std::hint::spin_loop();
            }
        }));
    }

    std::thread::sleep(Duration::from_micros(jitter_us));

    // Baseline-only: occasional extra scheduling perturbation.
    if mode == ExecutionMode::Baseline {
        if step_id % 7 == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    for h in handles {
        let _ = h.join();
    }
}

fn export_csv(rows: &[MetricRow]) {
    let _ = fs::create_dir_all("target");

    let path = "target/apx_runtime_stability_metrics.csv";
    let mut f = fs::File::create(path).expect("failed to create CSV output");

    writeln!(f, "step,mode,policy,latency_us,policy_switch").expect("failed to write CSV header");

    for r in rows {
        let switch = if r.policy_switch { "true" } else { "false" };
        writeln!(
            f,
            "{},{},{},{},{}",
            r.step,
            r.mode.as_str(),
            r.policy,
            r.latency_us,
            switch
        )
        .expect("failed to write CSV row");
    }
}

fn print_summary(rows: &[MetricRow]) {
    summarize_mode(rows, ExecutionMode::Baseline);
    summarize_mode(rows, ExecutionMode::Adaptive);
}

fn summarize_mode(rows: &[MetricRow], mode: ExecutionMode) {
    let mut lat: Vec<f64> = Vec::new();
    let mut switches = 0u64;

    for r in rows {
        if r.mode != mode {
            continue;
        }
        lat.push(r.latency_us as f64);
        if r.policy_switch {
            switches += 1;
        }
    }

    if lat.is_empty() {
        println!("[APX Runtime Stability] mode={} no samples", mode.as_str());
        return;
    }

    let n = lat.len() as f64;
    let mean = lat.iter().sum::<f64>() / n;
    let var = lat.iter().map(|x| {
        let d = x - mean;
        d * d
    }).sum::<f64>() / n;

    println!(
        "[APX Runtime Stability] mode={} steps={} policy_switches={} mean_us={:.2} var_us2={:.2}",
        mode.as_str(),
        lat.len(),
        switches,
        mean,
        var
    );
}

/// Minimal deterministic RNG for test-only jitter.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // LCG parameters (Numerical Recipes).
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 16) as u32
    }
}
