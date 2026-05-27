//! **β.4** — INT8 outlier-decomposition validation harness against
//! the production checkpoints used by the M8.5 F64 fixture.
//!
//! ## What this measures (and what it does not)
//!
//! The β.4 question is: *does CpuInt8Outlier reduce real-world
//! weight reconstruction drift versus M9 plain INT8?* The honest
//! answer to that depends on two things:
//!
//!   1. **Per-tensor reconstruction error** — how close
//!      `decompose → reconstruct(W)` is to the original W vs how
//!      close `absmax_per_group(W)` is. This file measures that
//!      across every `_proj.weight` of the four production
//!      models (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B,
//!      Llama 3.2 1B). The numbers are read directly from the
//!      safetensors files; no forward pass, no store
//!      mutation, no runtime wiring.
//!
//!   2. **End-to-end logit drift versus the F64 fixture** — the
//!      ADR-004 gate (`max_abs_diff < 0.5`). That measurement
//!      requires injecting `CpuInt8Outlier` into the
//!      `WeightStore` and running the full forward, which today
//!      means extending `SharedParam` with a new variant. That
//!      wiring belongs to β.5, not β.4 (scope rule 9: "NO
//!      integrar todavía en manifest").
//!
//! The per-tensor signal **is the necessary condition** for the
//! end-to-end signal: if the outlier reconstruction does not beat
//! plain INT8 at the tensor level, the cascaded drift cannot be
//! better either. β.4 therefore acts as a **kill switch** — if
//! the harness shows no improvement, we abort Track β before
//! paying the β.5 store-rewiring cost.
//!
//! ## How to run
//!
//! ```powershell
//! $models = "F:\Proyectos\artenia_engine\atenia-engine\models"
//! $env:TINYLLAMA_SAFETENSORS_PATH = "$models\tinyllama-1.1b\model.safetensors"
//! $env:SMOLLM2_SAFETENSORS_PATH   = "$models\smollm2-1.7b-instruct\model.safetensors"
//! $env:QWEN25_SAFETENSORS_PATH    = "$models\qwen2.5-1.5b-instruct\model.safetensors"
//! $env:LLAMA32_SAFETENSORS_PATH   = "$models\llama-3.2-1b-instruct\model.safetensors"
//! cargo test --release --test int8_outlier_f64_validation_test \
//!     -- --ignored --nocapture
//! ```
//!
//! The lightweight tests in this file (no `#[ignore]`) run in CI
//! and pin the mechanics — they do not touch the real
//! checkpoints.

use std::env;
use std::path::Path;

use atenia_engine::tensor::quantizer::{
    absmax_per_group_symmetric, decompose_outliers_topk_by_absmax,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

/// Configuration parameters reused across every model run.
const GROUP_SIZE: usize = 128; // M9.4 production value
const OUTLIER_K: usize = 64;   // β.4 starting point (audit recommendation)

/// Per-tensor measurement.
#[derive(Debug, Clone)]
struct TensorResult {
    name: String,
    shape: Vec<usize>,
    plain_int8_max_err: f32,
    outlier_max_err: f32,
}

impl TensorResult {
    /// Ratio improvement (plain / outlier). >1 means the outlier
    /// decomposition is strictly better than plain INT8.
    fn improvement_ratio(&self) -> f32 {
        if self.outlier_max_err == 0.0 {
            f32::INFINITY
        } else {
            self.plain_int8_max_err / self.outlier_max_err
        }
    }
}

/// Per-model aggregate.
#[derive(Debug, Clone)]
struct ModelResult {
    label: &'static str,
    tensor_count: usize,
    plain_int8_max_err: f32,
    outlier_max_err: f32,
    median_improvement: f32,
    worst_improvement: f32,
}

/// Read an F32 weight slice from a safetensors entry. Accepts
/// both F32 and BF16 sources — `TensorEntry::to_vec_f32` performs
/// the upcast internally. Returns the raw `Vec<f32>` and the shape.
fn read_f32_weight(reader: &SafetensorsReader, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let entry = reader.get(name)?;
    let shape = entry.shape.to_vec();
    let weights = entry.to_vec_f32().ok()?;
    Some((weights, shape))
}

/// Compute the per-tensor max-abs reconstruction error for plain
/// INT8 (per-group absmax) vs outlier decomposition. Clamps
/// `group_size` and `outlier_k` to the tensor dimensions so the
/// helper composes cleanly on both production-scale weights and
/// small synthetic inputs used by the lightweight CI tests.
fn measure_tensor(name: &str, weights: &[f32], shape: &[usize]) -> Option<TensorResult> {
    if shape.len() != 2 {
        return None;
    }
    let (k, n) = (shape[0], shape[1]);
    if k == 0 || n == 0 {
        return None;
    }
    let group_size = GROUP_SIZE.min(k.max(1));
    let outlier_k = OUTLIER_K.min(n);

    // Plain INT8: round-trip and measure max error vs original.
    let (q, scales) = absmax_per_group_symmetric(weights, shape, group_size);
    let num_cols = n;
    let mut plain_recon = vec![0.0_f32; weights.len()];
    for idx in 0..weights.len() {
        let row = idx / num_cols;
        let col = idx % num_cols;
        let g = row / group_size;
        plain_recon[idx] = (q[idx] as f32) * scales[g * num_cols + col];
    }
    let plain_int8_max_err = weights
        .iter()
        .zip(&plain_recon)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // Outlier decomposition + reconstruction.
    let decomp =
        decompose_outliers_topk_by_absmax(weights, shape, group_size, outlier_k).ok()?;
    let outlier_recon =
        atenia_engine::tensor::quantizer::reconstruct_outlier_decomposition(&decomp);
    let outlier_max_err = weights
        .iter()
        .zip(&outlier_recon)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    Some(TensorResult {
        name: name.to_string(),
        shape: shape.to_vec(),
        plain_int8_max_err,
        outlier_max_err,
    })
}

/// Resolve the safetensors path from an env var or skip the test
/// with a clear message. Returns `None` to signal "operator did
/// not opt in"; the per-model test bodies handle that gracefully.
fn resolve_path(env_var: &str) -> Option<String> {
    match env::var(env_var) {
        Ok(p) if Path::new(&p).is_file() => Some(p),
        Ok(p) => {
            eprintln!(
                "{env_var} = `{p}` but no file exists at that path. \
                 See docs/MODELS_LAYOUT.md for the canonical layout."
            );
            None
        }
        Err(_) => {
            eprintln!("{env_var} unset — skipping model.");
            None
        }
    }
}

/// Run the per-tensor harness over every `_proj.weight` of one
/// model. Aggregates into a [`ModelResult`] and prints a
/// per-tensor table to stderr.
fn run_one_model(label: &'static str, path: &str) -> ModelResult {
    println!("\n=== β.4: per-tensor reconstruction harness for {label} ===");
    let reader = SafetensorsReader::open(Path::new(path)).expect("open safetensors");
    let proj_names: Vec<String> = reader
        .iter()
        .filter(|e| e.name.ends_with("_proj.weight"))
        .map(|e| e.name.to_string())
        .collect();
    eprintln!("{label}: {} _proj.weight tensors", proj_names.len());

    let mut results: Vec<TensorResult> = Vec::new();
    for name in &proj_names {
        let Some((weights, shape)) = read_f32_weight(&reader, name) else {
            eprintln!("  skipped (unsupported dtype): {name}");
            continue;
        };
        if let Some(r) = measure_tensor(name, &weights, &shape) {
            eprintln!(
                "  {:<55} {:?}  plain={:.4e}  outlier={:.4e}  ratio={:.2}x",
                r.name,
                r.shape,
                r.plain_int8_max_err,
                r.outlier_max_err,
                r.improvement_ratio()
            );
            results.push(r);
        }
    }

    let plain_int8_max_err = results
        .iter()
        .map(|r| r.plain_int8_max_err)
        .fold(0.0_f32, f32::max);
    let outlier_max_err = results
        .iter()
        .map(|r| r.outlier_max_err)
        .fold(0.0_f32, f32::max);

    let mut ratios: Vec<f32> = results.iter().map(|r| r.improvement_ratio()).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_improvement = if ratios.is_empty() {
        0.0
    } else {
        ratios[ratios.len() / 2]
    };
    let worst_improvement = ratios.first().copied().unwrap_or(0.0);

    ModelResult {
        label,
        tensor_count: results.len(),
        plain_int8_max_err,
        outlier_max_err,
        median_improvement,
        worst_improvement,
    }
}

/// Pretty-print the per-model summary table.
fn print_summary(results: &[ModelResult]) {
    println!("\n========================================================");
    println!("β.4 SUMMARY (per-tensor max-abs reconstruction error)");
    println!("group_size = {GROUP_SIZE},  outlier_k = {OUTLIER_K}");
    println!("--------------------------------------------------------");
    println!(
        "{:<22} {:>8} {:>14} {:>14} {:>10} {:>10}",
        "Model", "tensors", "plain_max", "outlier_max", "median_x", "worst_x"
    );
    for r in results {
        println!(
            "{:<22} {:>8} {:>14.4e} {:>14.4e} {:>9.2}x {:>9.2}x",
            r.label,
            r.tensor_count,
            r.plain_int8_max_err,
            r.outlier_max_err,
            r.median_improvement,
            r.worst_improvement
        );
    }
    println!("========================================================");
    println!(
        "Note: these are reconstruction maxima at the weight level. \
         End-to-end logit drift vs the F64 fixture (the ADR-004 gate) \
         requires the β.5 store integration to measure."
    );
}

// ----- per-model ignored tests (require local checkpoints) -----

#[test]
#[ignore = "requires <MODEL>_SAFETENSORS_PATH env vars; see file docs"]
fn beta4_all_models_per_tensor_reconstruction() {
    let candidates: &[(&'static str, &str)] = &[
        ("TinyLlama 1.1B", "TINYLLAMA_SAFETENSORS_PATH"),
        ("SmolLM2 1.7B", "SMOLLM2_SAFETENSORS_PATH"),
        ("Qwen 2.5 1.5B", "QWEN25_SAFETENSORS_PATH"),
        ("Llama 3.2 1B", "LLAMA32_SAFETENSORS_PATH"),
    ];

    let mut results: Vec<ModelResult> = Vec::new();
    for (label, env_var) in candidates {
        let Some(path) = resolve_path(env_var) else {
            continue;
        };
        results.push(run_one_model(label, &path));
    }

    if results.is_empty() {
        panic!(
            "no models were available — set the *_SAFETENSORS_PATH env vars and re-run with \
             --ignored --nocapture"
        );
    }
    print_summary(&results);

    // The β.4 contract on the tensor level: outlier must be
    // **strictly better** than plain INT8 on every measured
    // tensor. Anything worse than 1× improvement on any single
    // proj weight is a hard fail and means the policy needs
    // tuning before β.5 invests in store rewiring.
    for r in &results {
        assert!(
            r.worst_improvement >= 1.0,
            "{}: worst-case improvement {:.2}x is below 1.0 — outlier \
             policy regresses on at least one tensor",
            r.label,
            r.worst_improvement
        );
    }
}

// ============================================================
// β.5 — end-to-end forward validation against F64 fixture.
//
// This block extends the β.4 harness with the actual ADR-004
// measurement: load a model into a CPU-resident WeightStore,
// replace each `_proj.weight` with `SharedParam::CpuInt8Outlier`
// via `WeightStore::quantize_param_to_outlier`, run the forward,
// and compare logits versus the F64 fixture used by M8.5.
//
// The forward runs on CPU (no CUDA, no BF16 GPU shortcut). The
// flag `ATENIA_BETA_OUTLIER=1` plus the explicit
// `kernel_dtype = F32` make the experimental path 100 %
// opt-in — every other test path is unchanged.
// ============================================================

#[cfg(test)]
mod beta5_forward {
    use std::env;
    use std::path::Path;

    use atenia_engine::amg::builder::GraphBuilder;
    use atenia_engine::amg::weight_store::{SharedParam, WeightStore};
    use atenia_engine::gpu::tier_plan::{TensorMeta, TierPlanInput, plan};
    use atenia_engine::nn::llama::{
        LlamaConfig, LlamaRuntime, build_llama, build_llama_with_store, llama_weight_mapper,
    };
    use atenia_engine::tensor::{DType, Tensor};
    use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

    const GROUP_SIZE: usize = 128;
    const OUTLIER_K: usize = 64;
    const TOKENS: [f32; 4] = [1.0, 100.0, 200.0, 300.0];
    const ADR_004_THRESHOLD: f64 = 0.5;

    /// Minimal Llama config for TinyLlama 1.1B — copied verbatim
    /// from `m8_5_full_family_validation_test.rs` so the two
    /// harnesses agree on what model is being validated.
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

    /// Load TinyLlama into a CPU-only `WeightStore` (kernel_dtype
    /// = F32). Returns the store, the param names, the build
    /// runtime, and the config so the caller can build the
    /// execution graph and run the forward.
    fn load_tinyllama_cpu() -> Option<(
        WeightStore,
        Vec<String>,
        LlamaConfig,
        LlamaRuntime,
        usize,
    )> {
        let path = env::var("TINYLLAMA_SAFETENSORS_PATH").ok()?;
        if !Path::new(&path).is_file() {
            eprintln!(
                "TINYLLAMA_SAFETENSORS_PATH = `{}` but no file exists; skipping",
                path
            );
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

        // CPU-only plan: kernel_dtype = F32 → loader produces
        // SharedParam::F32 (or shared) variants in the store,
        // not BF16-in-VRAM. β.5 needs a CPU-resident store so
        // `quantize_param_to_outlier` can mutate it without
        // racing the GPU path.
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

    /// Replace every `_proj.weight` in the store with its
    /// outlier-decomposed counterpart. Skips parameters that
    /// already live on Cuda / Disk (β.5 contract: CPU-only).
    fn convert_proj_weights_to_outlier(
        store: &mut WeightStore,
        group_size: usize,
        k: usize,
    ) -> usize {
        let mut count = 0;
        let mut targets: Vec<usize> = Vec::new();
        for (i, name) in store.names.iter().enumerate() {
            if name.ends_with("_proj.weight") {
                // Only convert the variants we can dequant to F32
                // without GPU/disk access.
                if matches!(
                    store.params[i],
                    SharedParam::F32 { .. } | SharedParam::Bf16 { .. }
                ) {
                    targets.push(i);
                }
            }
        }
        for idx in targets {
            store
                .quantize_param_to_outlier(idx, group_size, k)
                .expect("quantize_param_to_outlier must succeed on a valid _proj.weight");
            count += 1;
        }
        count
    }

    fn forward_and_drift(
        store: WeightStore,
        param_names: Vec<String>,
        config: LlamaConfig,
        runtime: LlamaRuntime,
        vocab_size: usize,
        fixture_dir: &str,
    ) -> (f64, [bool; 4]) {
        let _ = param_names; // names already in store
        let mut gb_exec = GraphBuilder::new();
        let token_input_id = gb_exec.input();
        let handles_exec = build_llama_with_store(
            &mut gb_exec,
            &config,
            &runtime,
            token_input_id,
            &store,
            None,
        )
        .expect("build_llama_with_store");
        let _ = gb_exec.output(handles_exec.logits_id);
        let mut graph_for_exec = gb_exec.build();

        let tokens = Tensor::new_cpu(vec![1, 4], TOKENS.to_vec());
        let outputs = graph_for_exec.execute(vec![tokens]);
        let logits = outputs[0].as_cpu_slice();

        let f64_ref = load_f64_fixture(fixture_dir);
        assert_eq!(logits.len(), f64_ref.len());

        let drift: f64 = logits
            .iter()
            .zip(f64_ref.iter())
            .map(|(a, t)| ((*a as f64) - t).abs())
            .fold(0.0_f64, f64::max);

        let mut matches = [false; 4];
        for pos in 0..4 {
            let s = pos * vocab_size;
            let e = s + vocab_size;
            let a_pos = &logits[s..e];
            let f_pos = &f64_ref[s..e];
            let (a_id, _) = a_pos
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| {
                    x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let (f_id, _) = f_pos
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| {
                    x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            matches[pos] = a_id == f_id;
        }

        (drift, matches)
    }

    // β-pivot.2 — real activation calibration helpers.
    //
    // The math axis. Llama loader transposes HF weights from
    // `[N_out, K_in]` to `[K_in, N_out]` before they hit the store
    // (the `linear` builder consumes `[B, K] @ [K, N] = [B, N]`).
    // For AWQ to align with that convention, the per-row scale
    // `s[k]` must index the **leading axis (= K_in = input
    // channels)** of the stored weight, which matches the **last
    // axis** of the LHS activation tensor.

    use std::collections::HashMap;

    /// Per-K-channel activation statistics.
    #[derive(Debug, Clone)]
    struct ActivationStats {
        absmax: Vec<f32>,
        sample_count: usize,
    }

    impl ActivationStats {
        fn new(k: usize) -> Self {
            Self {
                absmax: vec![0.0; k],
                sample_count: 0,
            }
        }

        /// Update with a flat activation slice `[N_samples, K]`.
        fn update_from_flat(&mut self, data: &[f32], k: usize) {
            let n_samples = data.len() / k;
            for s in 0..n_samples {
                for ki in 0..k {
                    let v = data[s * k + ki].abs();
                    if v > self.absmax[ki] {
                        self.absmax[ki] = v;
                    }
                }
            }
            self.sample_count += n_samples;
        }

        /// Merge stats from `other` (max-of-max semantics).
        fn merge(&mut self, other: &ActivationStats) {
            assert_eq!(
                self.absmax.len(),
                other.absmax.len(),
                "ActivationStats merge: K mismatch"
            );
            for (s, o) in self.absmax.iter_mut().zip(&other.absmax) {
                if *o > *s {
                    *s = *o;
                }
            }
            self.sample_count += other.sample_count;
        }
    }

    /// Walk a graph after a forward and capture per-input-channel
    /// activation absmax for every `*_proj.weight` MatMul.
    ///
    /// Returns a map `store_idx -> ActivationStats`. Skips MatMuls
    /// where either the RHS does not match a store param, the LHS
    /// node lost its `.output` to the executor's intermediate
    /// cleanup, or the LHS shape is degenerate.
    fn capture_proj_activation_stats(
        graph: &atenia_engine::amg::graph::Graph,
        param_ids: &[usize],
        param_names: &[String],
    ) -> HashMap<usize, ActivationStats> {
        use atenia_engine::amg::nodes::NodeType;

        // node_id -> store_idx (for parameter nodes that came from
        // `build_llama_with_store`).
        let mut node_to_store: HashMap<usize, usize> = HashMap::new();
        for (store_idx, &node_id) in param_ids.iter().enumerate() {
            node_to_store.insert(node_id, store_idx);
        }

        let mut stats: HashMap<usize, ActivationStats> = HashMap::new();
        for node in &graph.nodes {
            if !matches!(node.node_type, NodeType::MatMul) {
                continue;
            }
            if node.inputs.len() < 2 {
                continue;
            }
            let rhs_id = node.inputs[1];
            let Some(&store_idx) = node_to_store.get(&rhs_id) else {
                continue;
            };
            if !param_names[store_idx].ends_with("_proj.weight") {
                continue;
            }
            let lhs_id = node.inputs[0];
            let Some(lhs_t) = graph.nodes[lhs_id].output.as_ref() else {
                // Intermediate output was consumed by the executor.
                // Without an engine-side hook we cannot recover it.
                eprintln!(
                    "[β-pivot.2] LHS output missing for matmul -> {} (store_idx {})",
                    param_names[store_idx], store_idx
                );
                continue;
            };
            let shape = &lhs_t.shape;
            let Some(&k) = shape.last() else {
                continue;
            };
            if k == 0 || lhs_t.numel() == 0 {
                continue;
            }
            let data = lhs_t.copy_to_cpu_vec();
            let new_stats = {
                let mut s = ActivationStats::new(k);
                s.update_from_flat(&data, k);
                s
            };
            stats
                .entry(store_idx)
                .and_modify(|e| e.merge(&new_stats))
                .or_insert(new_stats);
        }
        stats
    }

    /// Compute calibrated per-K scales for every captured tensor.
    fn calibrated_scales(
        captured: &HashMap<usize, ActivationStats>,
        alpha: f32,
    ) -> HashMap<usize, Vec<f32>> {
        let mut out = HashMap::new();
        for (idx, stats) in captured {
            let s = atenia_engine::tensor::quantizer::awq_per_row_scales_from_activations(
                &stats.absmax,
                alpha,
            )
            .expect("scale derivation must succeed on captured stats");
            out.insert(*idx, s);
        }
        out
    }

    /// Replace every `_proj.weight` with its AWQ-perturbed F32
    /// version using `WeightStore::perturb_param_with_awq`. The
    /// runtime sees a plain F32 weight — no new storage variant
    /// is needed because the AWQ math collapses to a "perturbed
    /// F32" buffer at the boundary.
    fn convert_proj_weights_to_awq(
        store: &mut WeightStore,
        group_size: usize,
        alpha: f32,
    ) -> usize {
        let mut count = 0;
        let mut targets: Vec<usize> = Vec::new();
        for (i, name) in store.names.iter().enumerate() {
            if name.ends_with("_proj.weight")
                && matches!(
                    store.params[i],
                    SharedParam::F32 { .. } | SharedParam::Bf16 { .. }
                )
            {
                targets.push(i);
            }
        }
        for idx in targets {
            store
                .perturb_param_with_awq(idx, group_size, alpha)
                .expect("perturb_param_with_awq must succeed on a valid _proj.weight");
            count += 1;
        }
        count
    }

    /// **β-pivot.3 real-text calibration prompts.**
    ///
    /// Tokenises 8 short natural-language sentences via the
    /// existing `AteniaTokenizer` (HF byte-exact), takes the first
    /// `runtime.seq` tokens of each, casts to F32 to fit the
    /// `Tensor::new_cpu(vec![1, S], f32_data)` token-input
    /// convention. Returns one `[1, runtime.seq]` Tensor per
    /// prompt.
    ///
    /// The model directory carrying the safetensors also ships
    /// `tokenizer.json` next to it, so resolving the tokenizer
    /// path piggybacks on the same `TINYLLAMA_SAFETENSORS_PATH`
    /// env var the β.5 harness already requires.
    fn build_real_text_calibration_tokens(
        seq: usize,
        model_dir: &std::path::Path,
    ) -> Result<Vec<Vec<f32>>, String> {
        use atenia_engine::tokenizer::AteniaTokenizer;
        let tok = AteniaTokenizer::from_model_dir(model_dir)
            .map_err(|e| format!("AteniaTokenizer::from_model_dir: {e}"))?;
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
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let ids = tok
                .encode(p, true) // BOS prepended; matches Llama-family convention
                .map_err(|e| format!("encode `{p}`: {e}"))?;
            if ids.is_empty() {
                continue;
            }
            // Take the first `seq` ids; pad with the last id if the
            // prompt tokenises shorter than `seq` (BOS + short
            // prompt → unlikely to fall under 4 with these
            // sentences, but defensive).
            let mut window: Vec<f32> = ids.iter().take(seq).map(|&x| x as f32).collect();
            while window.len() < seq {
                let last = *window.last().unwrap_or(&1.0_f32);
                window.push(last);
            }
            out.push(window);
        }
        if out.is_empty() {
            return Err("no calibration prompts tokenised".into());
        }
        Ok(out)
    }

    /// **β-pivot.2 calibration forward**. Builds the graph wired
    /// to `store` (cloned), runs one forward per calibration
    /// prompt, and merges per-param activation stats. Returns
    /// the merged map plus the number of prompts processed.
    fn run_calibration(
        store: &WeightStore,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
    ) -> HashMap<usize, ActivationStats> {
        // β.5 / β-pivot.2 used synthetic token ids. β-pivot.3
        // (the active variant) prefers real tokenised text via
        // the existing tokenizer; falls back to the synthetic
        // batch only if the tokenizer is unavailable (e.g. env
        // var TINYLLAMA_SAFETENSORS_PATH points at a model
        // directory that no longer has tokenizer.json next to it).
        let real_tokens: Vec<Vec<f32>> = match env::var("TINYLLAMA_SAFETENSORS_PATH").ok() {
            Some(path) => match Path::new(&path).parent() {
                Some(model_dir) => match build_real_text_calibration_tokens(runtime.seq, model_dir) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "[β-pivot.3] real-text calibration unavailable ({e}); falling back \
                             to β-pivot.2 synthetic prompts"
                        );
                        Vec::new()
                    }
                },
                None => Vec::new(),
            },
            None => Vec::new(),
        };

        let synthetic_fallback: Vec<Vec<f32>> = vec![
            vec![1.0, 100.0, 200.0, 300.0],
            vec![50.0, 250.0, 450.0, 650.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![500.0, 600.0, 700.0, 800.0],
        ];
        let prompts: Vec<Vec<f32>> = if real_tokens.is_empty() {
            synthetic_fallback
        } else {
            eprintln!(
                "[β-pivot.3] using {} real-text calibration sequences (seq = {})",
                real_tokens.len(),
                runtime.seq
            );
            real_tokens
        };

        let mut merged: HashMap<usize, ActivationStats> = HashMap::new();
        for prompt in &prompts {
            // Each prompt gets its own graph (the executor
            // populates node.output; cloning the store lets us
            // start clean per prompt).
            let local_store = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            let mut gb = GraphBuilder::new();
            let token_input_id = gb.input();
            let handles = build_llama_with_store(
                &mut gb,
                config,
                runtime,
                token_input_id,
                &local_store,
                None,
            )
            .expect("build_llama_with_store");
            let _ = gb.output(handles.logits_id);
            let mut graph = gb.build();

            let tokens = Tensor::new_cpu(vec![1, runtime.seq], prompt.clone());
            let _ = graph.execute(vec![tokens]);

            let stats = capture_proj_activation_stats(
                &graph,
                &handles.param_ids,
                &handles.param_names,
            );
            for (idx, s) in stats {
                merged.entry(idx).and_modify(|e| e.merge(&s)).or_insert(s);
            }
        }
        merged
    }

    /// β-pivot.2 headline test: capture real activation absmax
    /// during a 4-prompt calibration forward, derive per-K scales,
    /// AWQ-perturb every `_proj.weight`, and compare drift versus
    /// the F64 fixture. Honest reporting — does not panic on
    /// ADR-004 FAIL.
    #[test]
    #[ignore = "requires TINYLLAMA_SAFETENSORS_PATH; very slow"]
    fn beta_pivot2_tinyllama_calibrated_awq_reports_adr_004() {
        let Some((store, names, config, runtime, vocab)) = load_tinyllama_cpu() else {
            panic!("TINYLLAMA_SAFETENSORS_PATH not set");
        };

        eprintln!("β-pivot.2 TinyLlama: certified forward (F32 CPU baseline) ...");
        let (baseline_drift, baseline_matches) = {
            let baseline_store = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            forward_and_drift(
                baseline_store,
                names.clone(),
                config.clone(),
                runtime.clone(),
                vocab,
                "tinyllama_reference",
            )
        };
        eprintln!("  certified drift = {:.6}, argmax = {:?}", baseline_drift, baseline_matches);

        eprintln!("β-pivot.2 TinyLlama: running calibration pass ...");
        let cal_start = std::time::Instant::now();
        let captured = run_calibration(&store, &config, &runtime);
        eprintln!(
            "  captured stats for {} _proj.weight params in {:.1}s",
            captured.len(),
            cal_start.elapsed().as_secs_f32()
        );

        let alpha = env::var("ATENIA_AWQ_ALPHA")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5);

        let scales_map = calibrated_scales(&captured, alpha);
        eprintln!(
            "β-pivot.2 TinyLlama: applying calibrated AWQ (g={}, α={}, prompts=4) ...",
            GROUP_SIZE, alpha
        );
        let mut store_awq = store; // consume baseline
        let mut converted = 0;
        for (idx, scales) in &scales_map {
            store_awq
                .perturb_param_with_awq_calibrated(*idx, GROUP_SIZE, scales)
                .expect("perturbation must succeed");
            converted += 1;
        }
        eprintln!("  perturbed {converted} params via calibrated AWQ");
        assert!(converted >= 100, "expected ≥100 perturbations");

        eprintln!("β-pivot.2 TinyLlama: AWQ-calibrated forward ...");
        let (awq_drift, awq_matches) = forward_and_drift(
            store_awq,
            names,
            config,
            runtime,
            vocab,
            "tinyllama_reference",
        );
        let argmax_count = awq_matches.iter().filter(|m| **m).count();

        eprintln!("\n========================================================");
        eprintln!("β-pivot.2 TinyLlama 1.1B — calibrated AWQ vs F64 fixture");
        eprintln!("  group_size           = {GROUP_SIZE}");
        eprintln!("  alpha                = {alpha}");
        eprintln!("  calibration prompts  = 4 (synthetic)");
        eprintln!("  captured tensors     = {}", scales_map.len());
        eprintln!(
            "  certified drift      = {:.6}  argmax 4/4 = {}",
            baseline_drift,
            baseline_matches.iter().all(|m| *m)
        );
        eprintln!(
            "  calibrated AWQ drift = {:.6}  argmax 4/4 = {}  ADR-004 = {}",
            awq_drift,
            argmax_count == 4,
            if awq_drift < ADR_004_THRESHOLD { "PASS" } else { "FAIL" }
        );
        eprintln!("  Reference β.5 outlier k=64    = 1.200 (FAIL)");
        eprintln!("  Reference β-piv.1 AWQ α=0.5    = 2.478 (FAIL, weight-norm proxy)");
        eprintln!("========================================================");

        assert!(awq_drift.is_finite(), "AWQ drift must be finite");
        assert!(
            baseline_drift < ADR_004_THRESHOLD,
            "certified baseline must PASS ADR-004"
        );
    }

    /// β-pivot.1 headline test: load TinyLlama 1.1B on CPU, AWQ-
    /// perturb every `_proj.weight` with weight-norm scales
    /// (α = 0.5 default), run the forward, compare the logits
    /// versus the F64 fixture. Honest reporting — the test does
    /// not panic on ADR-004 FAIL; the verdict is printed and
    /// recorded in the milestone doc.
    #[test]
    #[ignore = "requires TINYLLAMA_SAFETENSORS_PATH; very slow (CPU F32 forward)"]
    fn beta_pivot1_tinyllama_awq_forward_reports_adr_004() {
        let Some((mut store, names, config, runtime, vocab)) = load_tinyllama_cpu() else {
            panic!(
                "TINYLLAMA_SAFETENSORS_PATH not set; rerun with the env var pointing at \
                 model.safetensors (see docs/MODELS_LAYOUT.md)"
            );
        };

        eprintln!("β-pivot.1 TinyLlama: certified forward (F32 CPU baseline) ...");
        let baseline_drift;
        let baseline_matches;
        {
            let baseline_store = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            let (d, m) = forward_and_drift(
                baseline_store,
                names.clone(),
                config.clone(),
                runtime.clone(),
                vocab,
                "tinyllama_reference",
            );
            baseline_drift = d;
            baseline_matches = m;
        }
        eprintln!(
            "  certified drift = {:.6}, argmax matches = {:?}",
            baseline_drift, baseline_matches
        );

        let alpha = env::var("ATENIA_AWQ_ALPHA")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5);

        eprintln!(
            "β-pivot.1 TinyLlama: applying AWQ perturbation (g={}, α={}, no-cal weight-norm) ...",
            GROUP_SIZE, alpha
        );
        let converted = convert_proj_weights_to_awq(&mut store, GROUP_SIZE, alpha);
        eprintln!("  perturbed {converted} params via AWQ");
        assert!(converted >= 100, "expected ≥100 _proj.weight conversions");

        eprintln!("β-pivot.1 TinyLlama: AWQ forward (this will take a few minutes) ...");
        let (awq_drift, awq_matches) = forward_and_drift(
            store,
            names,
            config,
            runtime,
            vocab,
            "tinyllama_reference",
        );
        let argmax_count = awq_matches.iter().filter(|m| **m).count();

        let baseline_pass = baseline_drift < ADR_004_THRESHOLD;
        let awq_pass = awq_drift < ADR_004_THRESHOLD;

        eprintln!("\n========================================================");
        eprintln!("β-pivot.1 TinyLlama 1.1B — AWQ vs F64 fixture");
        eprintln!("  group_size      = {GROUP_SIZE}");
        eprintln!("  alpha           = {alpha}");
        eprintln!(
            "  certified drift = {:.6}  argmax 4/4 = {}  ADR-004 = {}",
            baseline_drift,
            baseline_matches.iter().all(|m| *m),
            if baseline_pass { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "  AWQ       drift = {:.6}  argmax 4/4 = {}  ADR-004 = {}",
            awq_drift,
            argmax_count == 4,
            if awq_pass { "PASS" } else { "FAIL" }
        );
        eprintln!("  ADR-004 gate    = max_abs_diff < {ADR_004_THRESHOLD}");
        eprintln!("  Reference β.5   = outlier k=64 drift was 1.200, k=256 was 1.143 (both FAIL)");
        eprintln!("========================================================");

        assert!(awq_drift.is_finite(), "AWQ drift must be finite");
        assert!(
            baseline_drift < ADR_004_THRESHOLD,
            "certified baseline must PASS ADR-004 (harness sanity)"
        );
    }

    /// β.5 headline test: load TinyLlama 1.1B on CPU, convert
    /// every `_proj.weight` to `SharedParam::CpuInt8Outlier`,
    /// run the forward, compare the logits versus the F64
    /// fixture. The certified baseline runs in the same process
    /// for a same-conditions comparison.
    ///
    /// **Honest reporting contract.** The test does **not** panic
    /// on ADR-004 failure — β.5 is an experimental measurement,
    /// and a numerical regression here is data we *need* (the
    /// "STOP and report" branch in the audit). The test asserts
    /// only on the harness invariants (model loaded, conversion
    /// applied, forward returned logits of the right shape).
    /// ADR-004 PASS/FAIL is printed and recorded in
    /// `docs/HANDOFF_INT8_OUTLIER_BETA.md`.
    #[test]
    #[ignore = "requires TINYLLAMA_SAFETENSORS_PATH; very slow (CPU F32 forward)"]
    fn beta5_tinyllama_outlier_forward_reports_adr_004() {
        let Some((mut store, names, config, runtime, vocab)) = load_tinyllama_cpu() else {
            panic!(
                "TINYLLAMA_SAFETENSORS_PATH not set; rerun with the env var pointing at \
                 model.safetensors (see docs/MODELS_LAYOUT.md)"
            );
        };

        eprintln!(
            "β.5 TinyLlama: certified forward (F32 CPU baseline) ..."
        );
        let baseline_drift;
        let baseline_matches;
        {
            let baseline_store = WeightStore {
                params: store.params.clone(),
                names: store.names.clone(),
            };
            let (d, m) = forward_and_drift(
                baseline_store,
                names.clone(),
                config.clone(),
                runtime.clone(),
                vocab,
                "tinyllama_reference",
            );
            baseline_drift = d;
            baseline_matches = m;
        }
        eprintln!(
            "  certified drift = {:.6}, argmax matches = {:?}",
            baseline_drift, baseline_matches
        );

        let k_under_test = env::var("ATENIA_BETA_OUTLIER_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(OUTLIER_K);

        eprintln!(
            "β.5 TinyLlama: converting _proj.weight params to CpuInt8Outlier (g={}, k={}) ...",
            GROUP_SIZE, k_under_test
        );
        let converted = convert_proj_weights_to_outlier(&mut store, GROUP_SIZE, k_under_test);
        eprintln!("  converted {converted} params to CpuInt8Outlier");
        assert!(converted >= 100, "expected ≥100 _proj.weight conversions");

        eprintln!("β.5 TinyLlama: outlier forward (this will take a few minutes) ...");
        let (outlier_drift, outlier_matches) = forward_and_drift(
            store,
            names,
            config,
            runtime,
            vocab,
            "tinyllama_reference",
        );

        let argmax_count = outlier_matches.iter().filter(|m| **m).count();
        let baseline_pass = baseline_drift < ADR_004_THRESHOLD;
        let outlier_pass = outlier_drift < ADR_004_THRESHOLD;

        eprintln!("\n========================================================");
        eprintln!("β.5 TinyLlama 1.1B — full forward F64 comparison");
        eprintln!("  group_size      = {GROUP_SIZE}");
        eprintln!("  outlier_k       = {k_under_test}");
        eprintln!(
            "  certified drift = {:.6}  argmax 4/4 = {}  ADR-004 = {}",
            baseline_drift,
            baseline_matches.iter().all(|m| *m),
            if baseline_pass { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "  outlier   drift = {:.6}  argmax 4/4 = {}  ADR-004 = {}",
            outlier_drift,
            argmax_count == 4,
            if outlier_pass { "PASS" } else { "FAIL" }
        );
        eprintln!("  ADR-004 gate    = max_abs_diff < {ADR_004_THRESHOLD}");
        eprintln!("========================================================");

        // Harness invariants only — the numerical verdict above
        // is the actual β.5 deliverable, recorded in the
        // milestone doc.
        assert!(
            outlier_drift.is_finite(),
            "β.5 TinyLlama: outlier drift must be finite"
        );
        assert!(
            baseline_drift < ADR_004_THRESHOLD,
            "β.5 TinyLlama: certified baseline must pass ADR-004 (harness sanity)"
        );
    }
}

// ----- lightweight CI tests (no real model needed) -----

#[test]
fn harness_measure_tensor_matches_known_baseline() {
    // Build a deterministic 32x16 weight with two outlier
    // columns 1000x larger than the bulk. Confirm that the
    // harness reports a strict improvement on the same input
    // β.1 already proved an improvement on.
    let (k, n) = (32_usize, 16_usize);
    let outlier_cols = [3_usize, 11];
    let mut w = vec![0.0_f32; k * n];
    for row in 0..k {
        for col in 0..n {
            let base = ((row * n + col) as f32 * 0.013) - 0.5;
            w[row * n + col] = if outlier_cols.contains(&col) {
                base * 1000.0
            } else {
                base
            };
        }
    }
    let result =
        measure_tensor("synthetic.proj.weight", &w, &[k, n]).expect("measure_tensor must succeed");
    eprintln!("synthetic improvement: {:.2}x", result.improvement_ratio());
    assert!(
        result.improvement_ratio() >= 5.0,
        "harness must report >=5x improvement on a strongly outlier-heavy synthetic; \
         got {:.2}x (plain={}, outlier={})",
        result.improvement_ratio(),
        result.plain_int8_max_err,
        result.outlier_max_err,
    );
}

#[test]
fn harness_skips_non_2d_tensors() {
    // A 3D shape is rejected upstream — the harness must return
    // None rather than panicking, so the per-model loop can keep
    // iterating.
    let w = vec![0.5_f32; 2 * 4 * 8];
    assert!(measure_tensor("not_proj.bias", &w, &[2, 4, 8]).is_none());
}

#[test]
fn harness_handles_uniform_weight_gracefully() {
    // A perfectly uniform weight has no outliers; plain INT8
    // already round-trips to within `scale × 0.5`. The harness
    // must not divide-by-zero or panic.
    let w = vec![0.123_f32; 64];
    let r = measure_tensor("uniform.proj.weight", &w, &[16, 4]).expect("must succeed");
    assert!(r.improvement_ratio().is_finite() || r.improvement_ratio().is_infinite());
    // No outliers exist → outlier_max_err should equal or beat
    // plain (k=64 with N=4 means k is clamped to N internally and
    // every column becomes outlier-preserved → zero error).
    assert!(r.outlier_max_err <= r.plain_int8_max_err + 1e-6);
}
