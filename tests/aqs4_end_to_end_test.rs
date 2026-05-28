//! **AQS-4** — end-to-end policy evaluation harness (TinyLlama + F64).
//!
//! This is the first orchestration that judges a [`QuantizationPolicy`]
//! the way it actually matters: perturb the real TinyLlama weights, run a
//! real CPU forward, and compare the resulting logits against the
//! certified F64 fixture (the ADR-004 gate, `max_abs_diff < 0.5`).
//!
//! It exists because AQS-2's *local* per-tensor drift signal is the wrong
//! instrument for GPTQ — GPTQ trades per-element drift for column-level
//! error cancellation, which only surfaces through a forward (AQS-3
//! handoff). AQS-4 answers the science question honestly:
//!
//!   > Does GPTQ improve end-to-end logit drift even though its local
//!   > per-element drift is worse?
//!
//! ## Architecture
//!
//! * The reusable, model-agnostic pieces (`PolicyEvalCandidate`,
//!   `EndToEndEvalResult`, `logit_drift_metrics`, `render_result_table`)
//!   live in `atenia_engine::quant::end_to_end`.
//! * Weight perturbation goes through the experimental
//!   `WeightStore::perturb_all_proj_with_policy(&dyn QuantizationPolicy, ...)`.
//! * The TinyLlama load / calibration / forward pattern is copied from
//!   the β / β-pivot harness (`int8_outlier_f64_validation_test.rs`) so
//!   the two agree on the model and fixture.
//!
//! ## Determinism / scope
//!
//! Deterministic, CPU-only (`kernel_dtype = F32`), TinyLlama-only, opt-in
//! via `TINYLLAMA_SAFETENSORS_PATH`. No search, no auto-selection, no
//! manifests, no CUDA. **Original weights are never mutated** — every
//! candidate runs on a fresh clone of the loaded store.
//!
//! ## How to run the heavy harness
//!
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "F:\Proyectos\artenia_engine\atenia-engine\models\tinyllama-1.1b\model.safetensors"
//! cargo test --release --test aqs4_end_to_end_test -- --ignored --nocapture
//! ```
//!
//! The non-`#[ignore]` tests run in CI and pin the orchestration
//! mechanics on a tiny synthetic store (no model, no fixture).

use std::collections::HashMap;

use atenia_engine::amg::weight_store::{SharedParam, WeightStore};
use atenia_engine::quant::end_to_end::{
    render_result_table, EndToEndEvalResult, PolicyEvalCandidate,
};
use atenia_engine::quant::policy::{
    AwqPolicy, Bf16Fallback, GptqPolicy, HybridPolicy, PlainInt8, QuantizationPolicy,
};
use std::sync::Arc;

const GROUP_SIZE: usize = 128;
const OUTLIER_K: usize = 64;

// ============================================================================
// Light CI tests — synthetic store, no model, no fixture.
// ============================================================================

/// Build a 2-param synthetic store: one `q_proj.weight` (eligible) and
/// one `norm.weight` (ineligible — does not end with `_proj.weight`).
fn synthetic_store() -> WeightStore {
    let proj = (0..(16 * 16)).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect::<Vec<f32>>();
    let norm = vec![1.0_f32; 16];
    WeightStore {
        params: vec![
            SharedParam::F32 {
                shape: vec![16, 16],
                arc: Arc::new(proj),
            },
            SharedParam::F32 {
                shape: vec![16],
                arc: Arc::new(norm),
            },
        ],
        names: vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.input_layernorm.weight".to_string(),
        ],
    }
}

fn param_f32(store: &WeightStore, idx: usize) -> Vec<f32> {
    match &store.params[idx] {
        SharedParam::F32 { arc, .. } => (**arc).clone(),
        _ => panic!("expected F32 param at idx {idx}"),
    }
}

#[test]
fn orchestrator_preserves_original_weights() {
    let original = synthetic_store();
    let original_proj = param_f32(&original, 0);

    // Clone, perturb the clone, original must be untouched.
    let mut clone = WeightStore {
        params: original.params.clone(),
        names: original.names.clone(),
    };
    let cal: HashMap<usize, Vec<f32>> = HashMap::new();
    let n = clone
        .perturb_all_proj_with_policy(&PlainInt8::new(8), &cal)
        .unwrap();
    assert_eq!(n, 1, "only the q_proj.weight is eligible");

    // Original clone source is unchanged (Arc clone is copy-on-write here
    // because perturb replaces the Arc, never mutates through it).
    assert_eq!(
        param_f32(&original, 0),
        original_proj,
        "original weights must not change"
    );
    // The non-proj param is left alone in the perturbed store.
    assert_eq!(param_f32(&clone, 1), vec![1.0_f32; 16]);
    // The proj param changed.
    assert_ne!(param_f32(&clone, 0), original_proj);
}

#[test]
fn bf16_policy_is_endtoend_noop_on_store() {
    let original = synthetic_store();
    let original_proj = param_f32(&original, 0);
    let mut clone = WeightStore {
        params: original.params.clone(),
        names: original.names.clone(),
    };
    let cal: HashMap<usize, Vec<f32>> = HashMap::new();
    clone
        .perturb_all_proj_with_policy(&Bf16Fallback, &cal)
        .unwrap();
    // BF16 fallback is identity — the buffer is byte-for-byte unchanged.
    assert_eq!(param_f32(&clone, 0), original_proj);
}

#[test]
fn real_gptq_perturbs_store_deterministically() {
    // K=16 eligible q_proj; build an [S=8, K=16] activation matrix.
    let base = synthetic_store();
    let matrix: Vec<f32> = (0..(8 * 16)).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
    let mshape = vec![8usize, 16usize];

    for run in 0..2 {
        let mut a = WeightStore {
            params: base.params.clone(),
            names: base.names.clone(),
        };
        a.perturb_param_with_policy_matrix(0, &GptqPolicy::new(8, 0.01), &matrix, &mshape)
            .unwrap();
        if run == 0 {
            DETERMINISM_BUF.with(|b| *b.borrow_mut() = param_f32(&a, 0));
        } else {
            DETERMINISM_BUF.with(|b| {
                assert_eq!(
                    *b.borrow(),
                    param_f32(&a, 0),
                    "real GPTQ must be deterministic"
                );
            });
        }
    }
}

thread_local! {
    static DETERMINISM_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[test]
fn awq_on_store_requires_activation_stats() {
    let mut store = synthetic_store();
    let empty: HashMap<usize, Vec<f32>> = HashMap::new();
    // AWQ needs activation stats; with an empty map the eligible param
    // gets None and the policy errors out.
    let err = store
        .perturb_all_proj_with_policy(&AwqPolicy::new(8, 0.3), &empty)
        .unwrap_err();
    assert!(
        format!("{err}").contains("activation"),
        "expected missing-activation error, got: {err}"
    );
}

#[test]
fn result_table_renders_all_candidates() {
    let results = vec![
        EndToEndEvalResult {
            candidate_name: "bf16".into(),
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
            rmse: 0.0,
            argmax_match: true,
            memory_bytes: 512,
        },
        EndToEndEvalResult {
            candidate_name: "gptq".into(),
            max_abs_diff: 0.42,
            mean_abs_diff: 0.01,
            rmse: 0.03,
            argmax_match: true,
            memory_bytes: 256,
        },
    ];
    let table = render_result_table(&results);
    assert!(table.contains("bf16"));
    assert!(table.contains("gptq"));
    assert!(table.contains("candidate"));
}

// ============================================================================
// Heavy harness — real TinyLlama forward vs F64 fixture.
// ============================================================================

#[cfg(test)]
mod heavy {
    use super::*;
    use std::env;
    use std::path::Path;

    use atenia_engine::amg::builder::GraphBuilder;
    use atenia_engine::amg::graph::Graph;
    use atenia_engine::gpu::tier_plan::{plan, TensorMeta, TierPlanInput};
    use atenia_engine::nn::llama::{
        build_llama, build_llama_with_store, llama_weight_mapper, LlamaConfig, LlamaRuntime,
    };
    use atenia_engine::quant::end_to_end::logit_drift_metrics;
    use atenia_engine::tensor::{DType, Tensor};
    use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

    const ADR_004_THRESHOLD: f64 = 0.5;

    // Verbatim TinyLlama 1.1B config (matches the β harness + M8.5).
    const TINYLLAMA_CONFIG: &str = r#"{
      "architectures": ["LlamaForCausalLM"],
      "attention_bias": false,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "initializer_range": 0.02,
      "intermediate_size": 5632,
      "max_position_embeddings": 2048,
      "model_type": "llama",
      "num_attention_heads": 32,
      "num_hidden_layers": 22,
      "num_key_value_heads": 4,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "rope_theta": 10000.0,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.35.0",
      "use_cache": true,
      "vocab_size": 32000
    }"#;

    fn load_f64_fixture(rel_dir: &str) -> Vec<f64> {
        let path = std::path::PathBuf::from("tests/fixtures")
            .join(rel_dir)
            .join("expected_logits_f64.json");
        let s = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("F64 fixture missing: {}", path.display()));
        let json: serde_json::Value = serde_json::from_str(&s).expect("malformed F64 fixture");
        json["values"]
            .as_array()
            .expect("`values` array")
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn load_tinyllama_cpu() -> Option<(WeightStore, Vec<String>, LlamaConfig, LlamaRuntime, usize)>
    {
        let path = env::var("TINYLLAMA_SAFETENSORS_PATH").ok()?;
        if !Path::new(&path).is_file() {
            eprintln!("TINYLLAMA_SAFETENSORS_PATH = `{path}` but no file exists; skipping");
            return None;
        }
        let config = LlamaConfig::from_json_str(TINYLLAMA_CONFIG).expect("parse config");
        let runtime = LlamaRuntime { batch: 1, seq: 4 };

        let mut gb_scratch = GraphBuilder::new();
        let token_input_id = gb_scratch.input();
        let handles = build_llama(&mut gb_scratch, &config, &runtime, token_input_id);
        let _ = gb_scratch.output(handles.logits_id);
        let mut scratch = gb_scratch.build();

        let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
        let mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
            .expect("mapper");

        let metas: Vec<TensorMeta> = reader
            .iter()
            .map(|e| TensorMeta {
                name: e.name.to_string(),
                shape: e.shape.to_vec(),
                dtype: e.dtype,
            })
            .collect();
        let model_total_bytes: u64 = metas
            .iter()
            .map(|m| (m.shape.iter().product::<usize>() as u64) * 4)
            .sum();
        let plan_input = TierPlanInput {
            tensors: metas,
            free_vram_bytes: 0,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let plan_out = plan(&plan_input);
        let (store, _report) = mapper
            .load_into_with_residency_plan(
                &mut scratch,
                &reader,
                &plan_out,
                &handles.param_ids,
                &handles.param_names,
            )
            .expect("load_into_with_residency_plan");
        drop(scratch);
        drop(reader);
        Some((store, handles.param_names, config, runtime, 32_000))
    }

    /// Full calibration activation accumulator: row-major `[S, K]`
    /// (samples appended across prompts). `k` is the input dimension.
    #[derive(Clone)]
    struct ActStats {
        matrix: Vec<f32>,
        k: usize,
    }
    impl ActStats {
        fn samples(&self) -> usize {
            if self.k == 0 {
                0
            } else {
                self.matrix.len() / self.k
            }
        }
        /// Per-K absmax (derived) for AWQ / Hybrid.
        fn absmax(&self) -> Vec<f32> {
            let mut out = vec![0.0_f32; self.k];
            let s = self.samples();
            for sample in 0..s {
                let base = sample * self.k;
                for ki in 0..self.k {
                    let v = self.matrix[base + ki].abs();
                    if v > out[ki] {
                        out[ki] = v;
                    }
                }
            }
            out
        }
    }

    /// Walk a post-forward graph and capture the full per-input-channel
    /// activation rows for every `*_proj.weight` MatMul. Rows from
    /// successive prompts are appended, building the `[S, K]` calibration
    /// matrix real GPTQ needs (AWQ / Hybrid derive absmax from it).
    fn capture_proj_activation_stats(
        graph: &Graph,
        param_ids: &[usize],
        param_names: &[String],
    ) -> HashMap<usize, ActStats> {
        use atenia_engine::amg::nodes::NodeType;
        let mut node_to_store: HashMap<usize, usize> = HashMap::new();
        for (store_idx, &node_id) in param_ids.iter().enumerate() {
            node_to_store.insert(node_id, store_idx);
        }
        let mut stats: HashMap<usize, ActStats> = HashMap::new();
        for node in &graph.nodes {
            if !matches!(node.node_type, NodeType::MatMul) || node.inputs.len() < 2 {
                continue;
            }
            let Some(&store_idx) = node_to_store.get(&node.inputs[1]) else {
                continue;
            };
            if !param_names[store_idx].ends_with("_proj.weight") {
                continue;
            }
            let Some(lhs_t) = graph.nodes[node.inputs[0]].output.as_ref() else {
                continue;
            };
            let Some(&k) = lhs_t.shape.last() else {
                continue;
            };
            if k == 0 || lhs_t.numel() == 0 {
                continue;
            }
            let data = lhs_t.copy_to_cpu_vec();
            stats
                .entry(store_idx)
                .and_modify(|e| e.matrix.extend_from_slice(&data))
                .or_insert(ActStats { matrix: data, k });
        }
        stats
    }

    fn build_calibration_tokens(seq: usize, model_dir: &Path) -> Vec<Vec<f32>> {
        use atenia_engine::tokenizer::AteniaTokenizer;
        let prompts = [
            "Hello, how are you?",
            "The capital of France is Paris.",
            "Rust is a systems programming language.",
            "Machine learning models require careful numerical validation.",
            "Once upon a time there was a small dragon.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence systems can drift numerically.",
            "Quantization changes the numerical properties of inference.",
        ];
        match AteniaTokenizer::from_model_dir(model_dir) {
            Ok(tok) => {
                let mut out = Vec::new();
                for p in &prompts {
                    if let Ok(ids) = tok.encode(p, true) {
                        if ids.is_empty() {
                            continue;
                        }
                        let mut w: Vec<f32> = ids.iter().take(seq).map(|&x| x as f32).collect();
                        while w.len() < seq {
                            let last = *w.last().unwrap_or(&1.0);
                            w.push(last);
                        }
                        out.push(w);
                    }
                }
                if out.is_empty() {
                    eprintln!("[AQS-4] tokenizer produced no tokens; using synthetic fallback");
                    synthetic_tokens()
                } else {
                    eprintln!("[AQS-4] using {} real-text calibration sequences", out.len());
                    out
                }
            }
            Err(e) => {
                eprintln!("[AQS-4] tokenizer unavailable ({e}); using synthetic fallback");
                synthetic_tokens()
            }
        }
    }

    fn synthetic_tokens() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 100.0, 200.0, 300.0],
            vec![50.0, 250.0, 450.0, 650.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![500.0, 600.0, 700.0, 800.0],
        ]
    }

    /// Run the calibration pass over `store` (cloned per prompt) and
    /// return merged per-idx `ActStats` (full `[S, K]` matrices).
    fn run_calibration(
        store: &WeightStore,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
    ) -> HashMap<usize, ActStats> {
        let tokens = match env::var("TINYLLAMA_SAFETENSORS_PATH").ok() {
            Some(p) => match Path::new(&p).parent() {
                Some(dir) => build_calibration_tokens(runtime.seq, dir),
                None => synthetic_tokens(),
            },
            None => synthetic_tokens(),
        };

        let mut merged: HashMap<usize, ActStats> = HashMap::new();
        for prompt in &tokens {
            let local = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            let mut gb = GraphBuilder::new();
            let token_input_id = gb.input();
            let handles =
                build_llama_with_store(&mut gb, config, runtime, token_input_id, &local, None)
                    .expect("build_llama_with_store");
            let _ = gb.output(handles.logits_id);
            let mut graph = gb.build();
            let toks = Tensor::new_cpu(vec![1, runtime.seq], prompt.clone());
            let _ = graph.execute(vec![toks]);
            let stats =
                capture_proj_activation_stats(&graph, &handles.param_ids, &handles.param_names);
            for (idx, s) in stats {
                merged
                    .entry(idx)
                    .and_modify(|e| e.matrix.extend_from_slice(&s.matrix))
                    .or_insert(s);
            }
        }
        merged
    }

    /// Run a forward on `store` and return logit-drift metrics vs the F64
    /// fixture (max, mean, rmse, argmax_match across all 4 positions).
    fn forward_drift(
        store: WeightStore,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        vocab: usize,
    ) -> (f32, f32, f32, bool) {
        let mut gb = GraphBuilder::new();
        let token_input_id = gb.input();
        let handles =
            build_llama_with_store(&mut gb, config, runtime, token_input_id, &store, None)
                .expect("build_llama_with_store");
        let _ = gb.output(handles.logits_id);
        let mut graph = gb.build();
        let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0, 100.0, 200.0, 300.0]);
        let outputs = graph.execute(vec![tokens]);
        let logits = outputs[0].as_cpu_slice();
        let f64_ref = load_f64_fixture("tinyllama_reference");
        logit_drift_metrics(logits, &f64_ref, vocab, runtime.seq)
            .expect("logit layout must match fixture")
    }

    /// Sum of `policy.memory_bytes(shape)` over every eligible
    /// `_proj.weight` in the store.
    fn policy_memory_total(store: &WeightStore, policy: &dyn QuantizationPolicy) -> u64 {
        let mut total = 0u64;
        for (i, name) in store.names.iter().enumerate() {
            if name.ends_with("_proj.weight") {
                if let SharedParam::F32 { shape, .. } | SharedParam::Bf16 { shape, .. } =
                    &store.params[i]
                {
                    total += policy.memory_bytes(shape);
                }
            }
        }
        total
    }

    /// **AQS-4 headline.** Evaluate BF16 / INT8 / AWQ / Hybrid / GPTQ
    /// end-to-end on TinyLlama vs the F64 fixture. Honest reporting — no
    /// auto-selection, no goalpost-moving. Asserts only that the
    /// certified baseline passes ADR-004 and that every drift is finite.
    #[test]
    #[ignore = "requires TINYLLAMA_SAFETENSORS_PATH; very slow (CPU F32 forward)"]
    fn aqs4_tinyllama_policy_comparison() {
        let Some((store, _names, config, runtime, vocab)) = load_tinyllama_cpu() else {
            panic!("TINYLLAMA_SAFETENSORS_PATH not set");
        };

        // (0) Certified baseline (no perturbation).
        eprintln!("[AQS-4] certified baseline forward ...");
        let (b_max, b_mean, b_rmse, b_argmax) = {
            let s = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            forward_drift(s, &config, &runtime, vocab)
        };
        eprintln!(
            "  baseline: max={b_max:.6} mean={b_mean:.6} rmse={b_rmse:.6} argmax={b_argmax}"
        );

        // (1) Calibration pass (shared by AWQ / Hybrid / GPTQ).
        eprintln!("[AQS-4] calibration pass ...");
        let cal_start = std::time::Instant::now();
        let act = run_calibration(&store, &config, &runtime);
        eprintln!(
            "  captured {} _proj.weight activation matrices in {:.1}s",
            act.len(),
            cal_start.elapsed().as_secs_f32()
        );

        // Derive the per-K absmax map (AWQ / Hybrid) and the [S, K] matrix
        // map (real GPTQ) from the captured activations.
        let absmax_map: HashMap<usize, Vec<f32>> =
            act.iter().map(|(i, s)| (*i, s.absmax())).collect();
        let matrix_map: HashMap<usize, (Vec<f32>, Vec<usize>)> = act
            .iter()
            .map(|(i, s)| (*i, (s.matrix.clone(), vec![s.samples(), s.k])))
            .collect();

        // (2) Candidates — explicit, ordered. AQS-4 does not select.
        let alpha = env::var("ATENIA_AWQ_ALPHA")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.25);
        let damp = env::var("ATENIA_GPTQ_DAMP")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.01);

        let bf16 = Bf16Fallback;
        let int8 = PlainInt8::new(GROUP_SIZE);
        let awq = AwqPolicy::new(GROUP_SIZE, alpha);
        let hybrid = HybridPolicy::new(GROUP_SIZE, alpha, OUTLIER_K);
        let gptq = GptqPolicy::new(GROUP_SIZE, damp);
        let candidates: Vec<PolicyEvalCandidate> = vec![
            PolicyEvalCandidate { name: "bf16", policy: &bf16 },
            PolicyEvalCandidate { name: "plain_int8", policy: &int8 },
            PolicyEvalCandidate { name: "awq", policy: &awq },
            PolicyEvalCandidate { name: "hybrid", policy: &hybrid },
            PolicyEvalCandidate { name: "gptq", policy: &gptq },
        ];

        let mut results: Vec<EndToEndEvalResult> = Vec::new();
        // Baseline row first (reference frame).
        results.push(EndToEndEvalResult {
            candidate_name: "certified(f32)".into(),
            max_abs_diff: b_max,
            mean_abs_diff: b_mean,
            rmse: b_rmse,
            argmax_match: b_argmax,
            memory_bytes: 0,
        });

        for cand in &candidates {
            eprintln!("[AQS-4] candidate `{}` ...", cand.name);
            // Fresh clone — original store is never mutated.
            let mut s = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            let mem = policy_memory_total(&s, cand.policy);
            let cand_start = std::time::Instant::now();
            // Real GPTQ needs the full [S, K] matrix; the others take the
            // per-K absmax map (BF16 / INT8 ignore it).
            let converted = if cand.name == "gptq" {
                s.perturb_all_proj_with_policy_matrix(cand.policy, &matrix_map)
            } else {
                s.perturb_all_proj_with_policy(cand.policy, &absmax_map)
            }
            .unwrap_or_else(|e| panic!("perturb `{}` failed: {e}", cand.name));
            eprintln!(
                "  perturbed {converted} params in {:.1}s; mem≈{mem} bytes; forward ...",
                cand_start.elapsed().as_secs_f32()
            );
            let (max, mean, rmse, argmax) = forward_drift(s, &config, &runtime, vocab);
            results.push(EndToEndEvalResult {
                candidate_name: cand.name.into(),
                max_abs_diff: max,
                mean_abs_diff: mean,
                rmse,
                argmax_match: argmax,
                memory_bytes: mem,
            });
        }

        eprintln!("\n==================== AQS-4 TinyLlama end-to-end ====================");
        eprintln!("  group_size={GROUP_SIZE} alpha={alpha} outlier_k={OUTLIER_K} gptq_damp={damp}");
        eprint!("{}", render_result_table(&results));
        eprintln!("====================================================================");
        eprintln!(
            "  ADR-004 threshold = {ADR_004_THRESHOLD}. Local AQS-2 reference (random 64x64):"
        );
        eprintln!("    AWQ local max_abs_diff ≈ 0.016  |  GPTQ local max_abs_diff ≈ 0.048");
        eprintln!("  => compare GPTQ's END-TO-END row above against AWQ's to answer:");
        eprintln!("     'does GPTQ win end-to-end despite worse local drift?'");

        // Honest assertions only.
        assert!(
            (b_max as f64) < ADR_004_THRESHOLD,
            "certified baseline must PASS ADR-004 (got {b_max})"
        );
        for r in &results {
            assert!(r.max_abs_diff.is_finite(), "{} drift must be finite", r.candidate_name);
        }
    }
}
