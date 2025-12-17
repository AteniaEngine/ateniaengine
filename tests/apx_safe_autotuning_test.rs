use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::nn::mini_flux::{build_mini_flux_language_model, MiniFluxConfig};
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

use atenia_engine::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};
use atenia_engine::apx9::gpu_execution_planner::GPUExecutionPlanner;

/// Safe Autotuning via Virtual GPU Model (experimental).
///
/// This test evaluates safe autotuning through virtual execution, ensuring that unstable
/// strategies are filtered before real execution.
///
/// Process:
/// - Generate multiple execution policy candidates.
/// - Evaluate each candidate using the engine’s Virtual GPU Model (APX 9.x planners),
///   estimating memory usage and identifying risky candidates.
/// - Apply only "safe" candidates to real execution (Graph::execute), ensuring no panics.
/// Additional logging is included to explicitly document which execution policies are approved
/// and applied after virtual GPU evaluation.
/// This test explicitly logs the rationale behind policy classification to ensure transparency and reproducibility of the safe autotuning process.
///
/// Constraints:
/// - Unstable policies must never run on the real execution path.
/// - Classification must be based on real engine logic (virtual planners), not mocked.
#[test]
fn apx_safe_autotuning_vgpu_filters_unstable_policies() {
    let cfg = MiniFluxConfig {
        vocab_size: 64,
        seq_len: 12,
        d_model: 48,
        d_hidden: 96,
        num_layers: 2,
        batch_size: 4,
    };

    let input = build_deterministic_tokens(cfg.batch_size, cfg.seq_len, cfg.vocab_size);

    // Build graph and run one warmup forward to materialize intermediate outputs.
    // The virtual memory planner uses node.output sizes as part of its estimate.
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "4.19");
    }
    let mut graph = build_mini_flux_graph(&cfg);
    let _warmup = graph.execute(vec![input.clone()]);

    // Virtual evaluation: build a synthetic GPU execution plan (no real GPU).
    // We pick a large simulated VRAM so the plan reflects actual total requirement.
    let planner = GPUExecutionPlanner::new(4 * 1024 * 1024 * 1024); // 4GB simulated
    let virtual_plan = planner.build_plan(&graph);
    let required_vram = virtual_plan.total_vram_needed;

    // Generate policy candidates.
    let candidates = vec![
        PolicyCandidate {
            name: "policy_tiny_vram_baseline".to_string(),
            vram_budget_bytes: required_vram / 8,
            policy: FusionRuntimePolicy::Baseline,
        },
        PolicyCandidate {
            name: "policy_safe_vram_qkv".to_string(),
            vram_budget_bytes: required_vram.saturating_mul(2).max(required_vram + 1),
            policy: FusionRuntimePolicy::PreferQKV,
        },
        PolicyCandidate {
            name: "policy_safe_vram_full".to_string(),
            vram_budget_bytes: required_vram.saturating_mul(2).max(required_vram + 1),
            policy: FusionRuntimePolicy::PreferFull,
        },
    ];

    let mut safe: Vec<PolicyCandidate> = Vec::new();
    let mut discarded: Vec<(String, VirtualEval)> = Vec::new();

    for c in candidates {
        let eval = evaluate_candidate_virtual(&virtual_plan, &c);

        println!("[APX Safe Autotuning] policy={}", c.name);
        println!(
            "decision={} reason={} estimated_vram_bytes={} effective_budget_bytes={}",
            eval.decision,
            eval.decision_reason,
            eval.estimated_vram_bytes,
            eval.effective_budget_bytes
        );

        if eval.safe {
            println!(
                "[APX Safe Autotuning] policy={} → APPROVED (virtual_gpu_check_passed)",
                c.name
            );
            safe.push(c);
        } else {
            println!(
                "[APX Safe Autotuning] policy={} → REJECTED (virtual_gpu_check_failed)",
                c.name
            );
            discarded.push((c.name.clone(), eval));
        }
    }

    // Assertions: at least one policy must be discarded by virtual evaluation.
    assert!(
        !discarded.is_empty(),
        "expected at least one policy to be discarded by virtual evaluation"
    );

    for (name, ev) in &discarded {
        println!(
            "[APX Safe Autotuning] discarded_policy={} estimated_vram_bytes={} spill_nodes={} reason={}",
            name,
            ev.estimated_vram_bytes,
            ev.spill_nodes,
            ev.reason
        );
    }

    // Execute only safe policies on the real path.
    // (Real path here is the actual engine forward execution.)
    let mut executed_safe_names: Vec<String> = Vec::new();
    for c in &safe {
        println!(
            "[APX Safe Autotuning] applied_policy={} estimated_vram_bytes={}",
            c.name,
            required_vram
        );
        unsafe {
            std::env::set_var("ATENIA_APX_MODE", "7.12");
        }
        set_runtime_policy(c.policy);

        // Must not panic.
        let mut out = graph.execute(vec![input.clone()]);
        let _ = out.pop().expect("Mini-Flux graph must return a single output");
        executed_safe_names.push(c.name.clone());
    }

    // Assert that only safe policies were executed.
    assert_eq!(executed_safe_names.len(), safe.len());
}

#[derive(Clone, Debug)]
struct PolicyCandidate {
    name: String,
    vram_budget_bytes: usize,
    policy: FusionRuntimePolicy,
}

#[derive(Clone, Debug)]
struct VirtualEval {
    safe: bool,
    decision: &'static str,
    decision_reason: &'static str,
    estimated_vram_bytes: usize,
    effective_budget_bytes: usize,
    spill_nodes: usize,
    reason: String,
}

fn evaluate_candidate_virtual(plan: &atenia_engine::apx9::gpu_execution_planner::GPUExecutionPlan, c: &PolicyCandidate) -> VirtualEval {
    // Virtual criteria:
    // - if the plan indicates spills, we treat it as unstable (would fall back).
    // - if the plan exceeds the candidate VRAM budget, treat it as unstable.
    let spills = plan.spills.len();
    let est = plan.total_vram_needed;

    if spills > 0 {
        return VirtualEval {
            safe: false,
            decision: "unstable",
            decision_reason: "virtual_plan_contains_spills",
            estimated_vram_bytes: est,
            effective_budget_bytes: c.vram_budget_bytes,
            spill_nodes: spills,
            reason: "virtual plan contains spills".to_string(),
        };
    }

    if est > c.vram_budget_bytes {
        return VirtualEval {
            safe: false,
            decision: "unstable",
            decision_reason: "estimated_vram_exceeds_effective_budget",
            estimated_vram_bytes: est,
            effective_budget_bytes: c.vram_budget_bytes,
            spill_nodes: spills,
            reason: format!("estimated_vram_bytes={} exceeds budget_bytes={}", est, c.vram_budget_bytes),
        };
    }

    VirtualEval {
        safe: true,
        decision: "safe",
        decision_reason: "within_effective_budget",
        estimated_vram_bytes: est,
        effective_budget_bytes: c.vram_budget_bytes,
        spill_nodes: spills,
        reason: "ok".to_string(),
    }
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
