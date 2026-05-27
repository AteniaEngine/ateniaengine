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
