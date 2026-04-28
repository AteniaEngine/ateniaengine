//! Numerical validation of Atenia's Llama 3.2 1B forward against the
//! F64 mathematical ground truth (M4.6 Phase C.5; methodology per
//! ADR-002 / ADR-004).
//!
//! Following the methodology established with SmolLM2 and Qwen 2.5,
//! the primary assertion compares Atenia F32 against PyTorch F64
//! (mathematical truth). PyTorch BF16 stats are reported as
//! informative diagnostics tracking how industry-default BF16
//! inference drifts from truth on this specific model — but BF16
//! agreement is not a pass/fail gate.
//!
//! IMPORTANT: this is the seq_len = 4 check. The high-frequency
//! band of the Llama 3 RoPE schedule dominates at small `s` so the
//! scaled and unscaled inv_freq vectors give nearly the same
//! rotation. A green C.5 confirms there is no obvious wiring bug,
//! but does NOT falsify the scaling itself. The decisive check on
//! the mid + low frequency bands lives in
//! `tests/llama_3_2_long_context_validation_test.rs` (Phase C.6).
//!
//! Run with:
//! ```powershell
//! $env:LLAMA32_SAFETENSORS_PATH = "...\\model.safetensors"
//! cargo test --test llama_3_2_numerical_validation_test --release \
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

/// Verbatim snapshot of the Llama 3.2 1B Instruct config.json.
const EMBEDDED_LLAMA_3_2_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pad_token_id": 128004,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 128256
}"#;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from("tests/fixtures/llama32_reference").join(name)
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
#[ignore = "requires LLAMA32_SAFETENSORS_PATH + tests/fixtures/llama32_reference/expected_logits_f64.json"]
fn llama_3_2_atenia_matches_f64_ground_truth_seq4() {
    println!(
        "\n=== Llama 3.2 1B Numerical Validation vs F64 Ground Truth (M4.6 C.5 / ADR-004) ==="
    );
    println!("Note: seq_len=4 — high-freq band only, scaling NOT decisive here.\n");

    // ---------- Setup + forward ----------
    let path = env::var("LLAMA32_SAFETENSORS_PATH")
        .expect("Set LLAMA32_SAFETENSORS_PATH to Llama 3.2 1B model.safetensors");

    let config = LlamaConfig::from_json_str(EMBEDDED_LLAMA_3_2_CONFIG)
        .expect("failed to parse Llama 3.2 config");
    assert!(
        config.effective_rope_scaling().is_some(),
        "Llama 3.2 must carry rope_scaling"
    );
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), 146);

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
    assert_eq!(report.loaded, 146);
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
            "  Pos {}: Atenia id={:>6} logit={:.6}   F64 id={:>6} logit={:.6}   [{}]",
            pos, a_id, a_logit, f_id, f_logit, tag
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
    println!(
        "REMINDER: this 4-token check passes whether or not the Llama 3 \
         scaling is correctly applied. C.6 is the falsifier."
    );
}
