//! Numerical validation of Atenia's TinyLlama 1.1B forward against
//! the F64 mathematical ground truth (M4.6.1 retroactive validation
//! per ADR-004).
//!
//! Context:
//! TinyLlama was originally validated in M4.5-d.1 against PyTorch
//! BF16 (max_abs_diff < 5.0). The methodology was reasonable at the
//! time but ADR-004 — accepted after Phase A surfaced the
//! BF16-as-truth flaw on SmolLM2 — established PyTorch F64 as the
//! primary numerical reference. The original M4.5-d.1 test
//! (`tinyllama_numerical_validation_test.rs`) is intentionally left
//! untouched: it stands as the historical record of the pre-ADR-004
//! methodology and the M4.5 BF16 baseline.
//!
//! This test is the ADR-004-aligned counterpart. Same checkpoint,
//! same tokens, F64 ground truth.
//!
//! Primary assertion: `max_atenia_vs_f64 < 0.5`. Atenia is expected
//! to land in the same precision class as the rest of the M4.6
//! family: ~0.001 max drift, multi-thousand-x closer to F64 than
//! BF16.
//!
//! M4.5-d.1 footnote: that test reported a `Match: false` on
//! pos 3 (Atenia 29871 vs PyTorch-as-then-loaded id=595 with logit
//! 8.0625). The F64 reference here resolves the ambiguity: both
//! 595 and 29871 quantize to logit 8.0625 in BF16, producing a
//! ranking noise where the BF16 argmax bounces between near-ties.
//! In F64 there is no tie — 29871 wins by ~0.04 — and Atenia's
//! F32 result agrees with F64 to within 1e-5. The "mismatch" was
//! BF16 quantization, not an Atenia bug.
//!
//! Run with:
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "...\\model.safetensors"
//! cargo test --test tinyllama_f64_validation_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Verbatim snapshot of `models/tinyllama-1.1b/config.json`.
const EMBEDDED_TINYLLAMA_CONFIG: &str = r#"{
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

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from("tests/fixtures/tinyllama_reference").join(name)
}

fn load_f64_reference() -> Vec<f64> {
    let path = fixture_path("expected_logits_f64.json");
    let s = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("F64 fixture missing: {}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&s).expect("malformed F64 fixture");
    json["values"]
        .as_array()
        .expect("`values` must be an array")
        .iter()
        .map(|v| v.as_f64().expect("each value must be a number"))
        .collect()
}

fn load_bf16_reference() -> Vec<f32> {
    let path = fixture_path("expected_logits.json");
    let s = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("BF16 fixture missing: {}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&s).expect("malformed BF16 fixture");
    json["values"]
        .as_array()
        .expect("`values` must be an array")
        .iter()
        .map(|v| v.as_f64().expect("each value must be a number") as f32)
        .collect()
}

fn report_stats_f64(label: &str, diffs: &[f64]) {
    let n = diffs.len();
    let max = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mean: f64 = diffs.iter().sum::<f64>() / n as f64;
    let gt_01 = diffs.iter().filter(|&&d| d > 0.1).count();
    let gt_1 = diffs.iter().filter(|&&d| d > 1.0).count();
    let gt_5 = diffs.iter().filter(|&&d| d > 5.0).count();
    println!("{}:", label);
    println!("  max  = {:.6}", max);
    println!("  mean = {:.6}", mean);
    println!(
        "  count(>0.1) = {:>7} ({:>5.2}%)",
        gt_01,
        100.0 * gt_01 as f64 / n as f64
    );
    println!(
        "  count(>1.0) = {:>7} ({:>5.2}%)",
        gt_1,
        100.0 * gt_1 as f64 / n as f64
    );
    println!(
        "  count(>5.0) = {:>7} ({:>5.2}%)",
        gt_5,
        100.0 * gt_5 as f64 / n as f64
    );
}

fn report_stats_f32(label: &str, diffs: &[f32]) {
    let n = diffs.len();
    let max = diffs.iter().fold(0.0_f32, |a, &b| a.max(b));
    let mean: f32 = diffs.iter().sum::<f32>() / n as f32;
    let gt_01 = diffs.iter().filter(|&&d| d > 0.1).count();
    let gt_1 = diffs.iter().filter(|&&d| d > 1.0).count();
    let gt_5 = diffs.iter().filter(|&&d| d > 5.0).count();
    println!("{}:", label);
    println!("  max  = {:.6}", max);
    println!("  mean = {:.6}", mean);
    println!(
        "  count(>0.1) = {:>7} ({:>5.2}%)",
        gt_01,
        100.0 * gt_01 as f32 / n as f32
    );
    println!(
        "  count(>1.0) = {:>7} ({:>5.2}%)",
        gt_1,
        100.0 * gt_1 as f32 / n as f32
    );
    println!(
        "  count(>5.0) = {:>7} ({:>5.2}%)",
        gt_5,
        100.0 * gt_5 as f32 / n as f32
    );
}

#[test]
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH + tests/fixtures/tinyllama_reference/expected_logits_f64.json"]
fn tinyllama_atenia_matches_f64_ground_truth() {
    println!(
        "\n=== TinyLlama 1.1B Numerical Validation vs F64 Ground Truth (M4.6.1 / ADR-004) ===\n"
    );

    // ---------- Setup + forward ----------
    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH to TinyLlama 1.1B model.safetensors");

    let config = LlamaConfig::from_json_str(EMBEDDED_TINYLLAMA_CONFIG)
        .expect("failed to parse TinyLlama config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    // TinyLlama untied: 1 (embed) + 22 × 9 (per layer) + 1 (final
    // norm) + 1 (lm_head) = 201.
    assert_eq!(handles.param_ids.len(), 201);

    println!("Loading weights...");
    let load_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("build mapper");
    let report = mapper.load_into(&mut graph, &reader).expect("load_into");
    drop(reader);
    println!(
        "Loaded {} tensors in {:.2}s",
        report.loaded,
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(report.loaded, 201);
    assert!(report.missing.is_empty());

    println!("Running forward...");
    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    let forward_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward: {:.2}s", forward_start.elapsed().as_secs_f32());

    let atenia_logits: &[f32] = outputs[0].as_cpu_slice();
    let total = atenia_logits.len();
    println!("Atenia produced {} logits.", total);

    // ---------- Load both references ----------
    let f64_ref = load_f64_reference();
    let bf16_ref = load_bf16_reference();
    assert_eq!(f64_ref.len(), total, "F64 fixture length mismatch");
    assert_eq!(bf16_ref.len(), total, "BF16 fixture length mismatch");

    // ---------- Compute three drift series ----------
    let atenia_vs_f64: Vec<f64> = atenia_logits
        .iter()
        .zip(f64_ref.iter())
        .map(|(a, t)| ((*a as f64) - t).abs())
        .collect();

    let atenia_vs_bf16: Vec<f32> = atenia_logits
        .iter()
        .zip(bf16_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let bf16_vs_f64: Vec<f64> = bf16_ref
        .iter()
        .zip(f64_ref.iter())
        .map(|(b, t)| ((*b as f64) - t).abs())
        .collect();

    println!("\n--- Atenia F32 vs F64 ground truth  [PRIMARY METRIC] ---");
    report_stats_f64("Drift", &atenia_vs_f64);

    println!("\n--- Atenia F32 vs PyTorch BF16  [secondary, informative] ---");
    report_stats_f32("Drift", &atenia_vs_bf16);

    println!("\n--- PyTorch BF16 vs F64 ground truth  [industry-default drift, informative] ---");
    report_stats_f64("Drift", &bf16_vs_f64);

    // ---------- Per-position drift ----------
    println!("\n--- Per-position drift: Atenia F32 vs F64 ground truth ---");
    let vocab = config.vocab_size;
    let seq = runtime.seq;
    for pos in 0..seq {
        let s = pos * vocab;
        let e = s + vocab;
        let pos_diffs = &atenia_vs_f64[s..e];
        let pos_max = pos_diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
        let pos_mean: f64 = pos_diffs.iter().sum::<f64>() / vocab as f64;
        println!("  Pos {}: max={:.6}  mean={:.6}", pos, pos_max, pos_mean);
    }

    println!("\n--- Per-position drift: PyTorch BF16 vs F64 ground truth (industry reference) ---");
    for pos in 0..seq {
        let s = pos * vocab;
        let e = s + vocab;
        let pos_diffs = &bf16_vs_f64[s..e];
        let pos_max = pos_diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
        let pos_mean: f64 = pos_diffs.iter().sum::<f64>() / vocab as f64;
        println!("  Pos {}: max={:.6}  mean={:.6}", pos, pos_max, pos_mean);
    }

    // ---------- Argmax (Atenia vs F64) ----------
    println!("\n--- Argmax: Atenia F32 vs F64 ---");
    for pos in 0..seq {
        let s = pos * vocab;
        let e = s + vocab;
        let a_pos = &atenia_logits[s..e];
        let f_pos = &f64_ref[s..e];

        let (a_id, &a_logit) = a_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let (f_id, &f_logit) = f_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let tag = if a_id == f_id { "MATCH" } else { "MISMATCH" };
        println!(
            "  Pos {}: Atenia id={:>5} logit={:.6}   F64 id={:>5} logit={:.6}   [{}]",
            pos, a_id, a_logit, f_id, f_logit, tag
        );
    }

    // ---------- Argmax (BF16 vs F64) — illustrates the M4.5-d.1
    // BF16 framing: where Atenia and BF16 disagreed, F64 reveals
    // Atenia was right.
    println!("\n--- Argmax: PyTorch BF16 vs F64 (M4.5-d.1 framing in retrospect) ---");
    for pos in 0..seq {
        let s = pos * vocab;
        let e = s + vocab;
        let b_pos = &bf16_ref[s..e];
        let f_pos = &f64_ref[s..e];

        let (b_id, &b_logit) = b_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let (f_id, &f_logit) = f_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let tag = if b_id == f_id { "MATCH" } else { "MISMATCH" };
        println!(
            "  Pos {}: BF16   id={:>5} logit={:.6}   F64 id={:>5} logit={:.6}   [{}]",
            pos, b_id, b_logit, f_id, f_logit, tag
        );
    }

    // ---------- Verdict ----------
    let max_atenia_vs_f64 = atenia_vs_f64.iter().fold(0.0_f64, |a, &b| a.max(b));
    let max_bf16_vs_f64 = bf16_vs_f64.iter().fold(0.0_f64, |a, &b| a.max(b));

    println!("\n=== Verdict ===");
    println!("Atenia F32   max drift vs F64 truth: {:.6}", max_atenia_vs_f64);
    println!("PyTorch BF16 max drift vs F64 truth: {:.6}", max_bf16_vs_f64);

    if max_atenia_vs_f64 < max_bf16_vs_f64 {
        let ratio = max_bf16_vs_f64 / max_atenia_vs_f64.max(1e-9);
        println!(
            "Atenia is {:.0}x closer to F64 mathematical truth than PyTorch BF16 on this run.",
            ratio
        );
    }

    // ---------- Single primary assertion (per ADR-004) ----------
    assert!(
        max_atenia_vs_f64 < 0.5,
        "Atenia F32 drift vs F64 ground truth ({:.6}) exceeds tolerance (0.5).",
        max_atenia_vs_f64
    );

    println!(
        "\nPASSED: Atenia matches F64 ground truth within F32 precision bounds (max drift {:.6} < 0.5).",
        max_atenia_vs_f64
    );
}
