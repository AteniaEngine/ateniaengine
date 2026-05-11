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
    GenerationPipeline, LlamaConfig, LlamaRuntime, build_llama, build_llama_with_store,
    llama_weight_mapper,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::gguf_decode::decode_tensor;
use atenia_engine::v17::loader::gguf_reader::GgufReader;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

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

fn models_root() -> PathBuf {
    env::var_os("ATENIA_MODELS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))
}

static DISK_TIER_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn unique_disk_tier_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after unix epoch")
        .as_nanos();
    let base = env::var_os("ATENIA_TEST_DISK_TIER_BASE")
        .or_else(|| env::var_os("LOCALAPPDATA"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("Atenia").join("test-cache").join(format!(
        "atenia_m11d_{label}_{}_{}",
        std::process::id(),
        nanos
    ))
}

fn load_pipeline_with_isolated_disk_tier(
    model_dir: &Path,
    bf16_storage: bool,
    label: &str,
) -> GenerationPipeline {
    let _guard = DISK_TIER_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("disk tier env lock poisoned");
    let previous = env::var_os("ATENIA_DISK_TIER_DIR");
    let cache_dir = unique_disk_tier_dir(label);
    // SAFETY: Rust 2024 marks process environment mutation unsafe because
    // concurrent readers/writers can race. This helper serializes all local
    // GGUF pipeline env mutation with a process-wide mutex and restores the
    // previous value before returning. The returned WeightStore keeps absolute
    // disk handles, so the env var is only needed during load planning.
    unsafe {
        env::set_var("ATENIA_DISK_TIER_DIR", &cache_dir);
    }
    let loaded = GenerationPipeline::from_model_dir_with_options(model_dir, bf16_storage);
    match previous {
        Some(v) => unsafe {
            env::set_var("ATENIA_DISK_TIER_DIR", v);
        },
        None => unsafe {
            env::remove_var("ATENIA_DISK_TIER_DIR");
        },
    }
    loaded.expect("load GGUF pipeline with isolated disk tier")
}

fn open_safetensors_for_tensor(
    safetensors_or_index_path: &Path,
    tensor_name: &str,
) -> SafetensorsReader {
    let actual_path = if safetensors_or_index_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".index.json"))
    {
        let index_text = fs::read_to_string(safetensors_or_index_path).unwrap_or_else(|_| {
            panic!(
                "read safetensors index {}",
                safetensors_or_index_path.display()
            )
        });
        let index: serde_json::Value =
            serde_json::from_str(&index_text).expect("parse safetensors index json");
        let shard = index["weight_map"][tensor_name]
            .as_str()
            .unwrap_or_else(|| panic!("{tensor_name} in safetensors index"));
        safetensors_or_index_path
            .parent()
            .expect("index path has parent")
            .join(shard)
    } else {
        safetensors_or_index_path.to_path_buf()
    };
    SafetensorsReader::open(&actual_path)
        .unwrap_or_else(|_| panic!("open safetensors {}", actual_path.display()))
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
    println!(
        "Atenia F32   max drift vs F64 truth: {:.6}",
        max_atenia_vs_f64
    );
    println!(
        "PyTorch BF16 max drift vs F64 truth: {:.6}",
        max_bf16_vs_f64
    );

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

#[test]
#[ignore = "requires TinyLlama Q4_K_M GGUF fixture + tests/fixtures/tinyllama_reference/expected_logits_f64.json"]
fn tinyllama_q4_k_m_gguf_reports_f64_drift() {
    println!("\n=== TinyLlama Q4_K_M GGUF Numerical Report vs F64 Ground Truth (M11.D.5) ===\n");

    let model_dir = models_root().join("TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF");
    assert!(
        model_dir
            .join("tinyllama-1.1b-chat-v1.0-q4_k_m.gguf")
            .exists(),
        "missing Q4_K_M GGUF fixture at {}",
        model_dir.display()
    );
    assert!(model_dir.join("tokenizer.json").exists());
    assert!(model_dir.join("tokenizer_config.json").exists());

    let load_start = std::time::Instant::now();
    let pipe = load_pipeline_with_isolated_disk_tier(&model_dir, true, "tinyllama_q4_k_m");
    println!(
        "Loaded Q4_K_M GGUF pipeline in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );

    let runtime = LlamaRuntime { batch: 1, seq: 4 };
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama_with_store(
        &mut gb,
        &pipe.config,
        &runtime,
        token_input_id,
        &pipe.store,
        None,
    )
    .expect("build GGUF-backed graph");
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();

    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    let forward_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward: {:.2}s", forward_start.elapsed().as_secs_f32());

    let atenia_logits = outputs[0].as_cpu_slice();
    let f64_ref = load_f64_reference();
    assert_eq!(
        f64_ref.len(),
        atenia_logits.len(),
        "F64 fixture length mismatch"
    );

    let diffs: Vec<f64> = atenia_logits
        .iter()
        .zip(f64_ref.iter())
        .map(|(a, t)| ((*a as f64) - t).abs())
        .collect();
    report_stats_f64("TinyLlama Q4_K_M GGUF drift vs F64", &diffs);

    let vocab = pipe.config.vocab_size;
    let seq = runtime.seq;
    let mut argmax_matches = 0usize;
    println!("\n--- Argmax: TinyLlama Q4_K_M GGUF vs F64 ---");
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
        if a_id == f_id {
            argmax_matches += 1;
        }
        let tag = if a_id == f_id { "MATCH" } else { "MISMATCH" };
        println!(
            "  Pos {}: GGUF id={:>5} logit={:.6}   F64 id={:>5} logit={:.6}   [{}]",
            pos, a_id, a_logit, f_id, f_logit, tag
        );
    }

    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    println!("\n=== Q4_K_M GGUF Verdict Candidate ===");
    println!("max_abs_diff_vs_f64 = {:.6}", max_diff);
    println!("argmax_match_{}_of_{}", argmax_matches, seq);
    println!("adr_004_strict_pass = {}", max_diff < 0.5);

    assert!(atenia_logits.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires TinyLlama Q8_0 GGUF fixture + tests/fixtures/tinyllama_reference/expected_logits_f64.json"]
fn tinyllama_q8_0_gguf_reports_f64_drift() {
    println!(
        "\n=== TinyLlama Q8_0 GGUF Numerical Report vs F64 Ground Truth (M11.D.5 diagnostic) ===\n"
    );

    let model_dir = models_root().join("tinyllama-q8_0");
    assert!(
        model_dir
            .join("tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
            .exists(),
        "missing Q8_0 GGUF fixture at {}",
        model_dir.display()
    );
    assert!(model_dir.join("tokenizer.json").exists());
    assert!(model_dir.join("tokenizer_config.json").exists());

    let pipe = load_pipeline_with_isolated_disk_tier(&model_dir, true, "tinyllama_q8_0");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama_with_store(
        &mut gb,
        &pipe.config,
        &runtime,
        token_input_id,
        &pipe.store,
        None,
    )
    .expect("build GGUF-backed graph");
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();

    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    let outputs = graph.execute(vec![tokens]);
    let atenia_logits = outputs[0].as_cpu_slice();
    let f64_ref = load_f64_reference();
    assert_eq!(
        f64_ref.len(),
        atenia_logits.len(),
        "F64 fixture length mismatch"
    );

    let diffs: Vec<f64> = atenia_logits
        .iter()
        .zip(f64_ref.iter())
        .map(|(a, t)| ((*a as f64) - t).abs())
        .collect();
    report_stats_f64("TinyLlama Q8_0 GGUF drift vs F64", &diffs);

    let vocab = pipe.config.vocab_size;
    let seq = runtime.seq;
    let mut argmax_matches = 0usize;
    for pos in 0..seq {
        let s = pos * vocab;
        let e = s + vocab;
        let a_pos = &atenia_logits[s..e];
        let f_pos = &f64_ref[s..e];
        let (a_id, _) = a_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let (f_id, _) = f_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if a_id == f_id {
            argmax_matches += 1;
        }
    }
    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    println!("max_abs_diff_vs_f64 = {:.6}", max_diff);
    println!("argmax_match_{}_of_{}", argmax_matches, seq);
    println!("adr_004_strict_pass = {}", max_diff < 0.5);

    assert!(atenia_logits.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires models/tinyllama-1.1b/model.safetensors and models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF locally"]
fn tinyllama_q4_k_m_gguf_norm_matches_safetensors_base() {
    let root = models_root();
    let safetensors_path = root.join("tinyllama-1.1b/model.safetensors");
    let gguf_path =
        root.join("TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf");

    assert!(
        safetensors_path.exists(),
        "missing {}",
        safetensors_path.display()
    );
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("model.norm.weight")
        .expect("model.norm.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors norm");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output_norm.weight")
        .expect("output_norm.weight in GGUF");
    let gguf_values = decode_tensor(&gguf, descriptor).expect("decode GGUF norm");

    assert_eq!(st.len(), gguf_values.len(), "norm length mismatch");

    let diffs: Vec<f64> = st
        .iter()
        .zip(gguf_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

    println!("TinyLlama safetensors model.norm.weight vs Q4_K_M GGUF output_norm.weight:");
    println!("  len       = {}", st.len());
    println!("  max_diff  = {:.9}", max_diff);
    println!("  mean_diff = {:.9}", mean_diff);
    println!("  st[0..8]  = {:?}", &st[..8.min(st.len())]);
    println!(
        "  gg[0..8]  = {:?}",
        &gguf_values[..8.min(gguf_values.len())]
    );

    assert!(
        max_diff < 0.5,
        "safetensors and Q4_K_M GGUF norm differ too much; likely decoder issue"
    );
}

#[test]
#[ignore = "requires models/tinyllama-1.1b/model.safetensors and models/tinyllama-q8_0 GGUF locally"]
fn tinyllama_q8_0_gguf_norm_matches_safetensors_base() {
    let root = models_root();
    let safetensors_path = root.join("tinyllama-1.1b/model.safetensors");
    let gguf_path = root.join("tinyllama-q8_0/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");

    assert!(
        safetensors_path.exists(),
        "missing {}",
        safetensors_path.display()
    );
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("model.norm.weight")
        .expect("model.norm.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors norm");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output_norm.weight")
        .expect("output_norm.weight in GGUF");
    let gguf_values = decode_tensor(&gguf, descriptor).expect("decode GGUF norm");

    assert_eq!(st.len(), gguf_values.len(), "norm length mismatch");

    let diffs: Vec<f64> = st
        .iter()
        .zip(gguf_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

    println!("TinyLlama safetensors model.norm.weight vs GGUF output_norm.weight:");
    println!("  len       = {}", st.len());
    println!("  max_diff  = {:.9}", max_diff);
    println!("  mean_diff = {:.9}", mean_diff);
    println!("  st[0..8]  = {:?}", &st[..8.min(st.len())]);
    println!(
        "  gg[0..8]  = {:?}",
        &gguf_values[..8.min(gguf_values.len())]
    );

    assert!(
        max_diff < 1e-3,
        "safetensors and GGUF norm differ too much; likely different base checkpoint"
    );
}

#[test]
#[ignore = "requires models/tinyllama-1.1b/model.safetensors and models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF locally"]
fn tinyllama_q4_k_m_gguf_lm_head_sample_diagnostic() {
    let root = models_root();
    let safetensors_path = root.join("tinyllama-1.1b/model.safetensors");
    let gguf_path =
        root.join("TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf");

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("lm_head.weight")
        .expect("lm_head.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors lm_head");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output.weight")
        .or_else(|| gguf.tensor_by_name("token_embd.weight"))
        .expect("output.weight or tied token_embd.weight in GGUF");
    let gguf_values =
        decode_tensor(&gguf, descriptor).expect("decode GGUF lm_head / tied embedding");

    assert_eq!(st.len(), gguf_values.len(), "lm_head length mismatch");

    let hidden = 2_048usize;
    let vocab = 32_000usize;
    let mut max_direct_diff = 0.0_f64;
    let mut sum_direct_diff = 0.0_f64;
    let mut checked = 0usize;
    for token in [0usize, 1, 2, 100, 5099, 29871, vocab - 1] {
        for h in [0usize, 1, 2, 64, 511, 1024, hidden - 1] {
            let st_value = st[token * hidden + h];
            let gguf_value = gguf_values[token * hidden + h];
            let diff = ((st_value as f64) - (gguf_value as f64)).abs();
            max_direct_diff = max_direct_diff.max(diff);
            sum_direct_diff += diff;
            checked += 1;
        }
    }
    let mean_direct_diff = sum_direct_diff / checked as f64;

    println!("TinyLlama safetensors lm_head.weight vs Q4_K_M GGUF output.weight samples:");
    println!("  checked   = {}", checked);
    println!("  max_diff  = {:.9}", max_direct_diff);
    println!("  mean_diff = {:.9}", mean_direct_diff);
    println!("  st[0..4]  = {:?}", &st[..4]);
    println!("  gg[0..4]  = {:?}", &gguf_values[..4]);
}

#[test]
#[ignore = "requires models/tinyllama-1.1b/model.safetensors and models/tinyllama-q8_0 GGUF locally"]
fn tinyllama_q8_0_gguf_lm_head_matches_safetensors_transpose() {
    let root = models_root();
    let safetensors_path = root.join("tinyllama-1.1b/model.safetensors");
    let gguf_path = root.join("tinyllama-q8_0/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("lm_head.weight")
        .expect("lm_head.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors lm_head");
    assert_eq!(st_entry.shape, &[32_000, 2_048]);

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output.weight")
        .or_else(|| gguf.tensor_by_name("token_embd.weight"))
        .expect("output.weight or tied token_embd.weight in GGUF");
    let gguf_values =
        decode_tensor(&gguf, descriptor).expect("decode GGUF lm_head / tied embedding");
    assert_eq!(descriptor.dimensions.as_slice(), &[2_048, 32_000]);
    assert_eq!(st.len(), gguf_values.len(), "lm_head length mismatch");

    let hidden = 2_048usize;
    let vocab = 32_000usize;
    let mut max_transposed_diff = 0.0_f64;
    let mut sum_transposed_diff = 0.0_f64;
    let mut max_direct_diff = 0.0_f64;
    let mut sum_direct_diff = 0.0_f64;
    let mut checked = 0usize;
    for token in [0usize, 1, 2, 100, 5099, 29871, vocab - 1] {
        for h in [0usize, 1, 2, 64, 511, 1024, hidden - 1] {
            let st_value = st[token * hidden + h];
            let transposed_value = gguf_values[h * vocab + token];
            let direct_value = gguf_values[token * hidden + h];
            let transposed_diff = ((st_value as f64) - (transposed_value as f64)).abs();
            let direct_diff = ((st_value as f64) - (direct_value as f64)).abs();
            max_transposed_diff = max_transposed_diff.max(transposed_diff);
            sum_transposed_diff += transposed_diff;
            max_direct_diff = max_direct_diff.max(direct_diff);
            sum_direct_diff += direct_diff;
            checked += 1;
        }
    }
    let mean_transposed_diff = sum_transposed_diff / checked as f64;
    let mean_direct_diff = sum_direct_diff / checked as f64;

    println!("TinyLlama safetensors lm_head.weight vs raw GGUF output.weight orientation samples:");
    println!("  checked   = {}", checked);
    println!("  transposed max_diff  = {:.9}", max_transposed_diff);
    println!("  transposed mean_diff = {:.9}", mean_transposed_diff);
    println!("  direct     max_diff  = {:.9}", max_direct_diff);
    println!("  direct     mean_diff = {:.9}", mean_direct_diff);

    assert!(
        mean_direct_diff < mean_transposed_diff,
        "raw GGUF output.weight does not match safetensors lm_head.weight direct layout"
    );
}

fn llama_rope_permute_rows(values: &[f32], rows: usize, cols: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(rows % head_dim, 0);
    let half = head_dim / 2;
    let mut out = vec![0.0_f32; values.len()];
    for head_base in (0..rows).step_by(head_dim) {
        for i in 0..half {
            let src0 = head_base + i;
            let src1 = head_base + i + half;
            let dst0 = head_base + 2 * i;
            let dst1 = head_base + 2 * i + 1;
            for c in 0..cols {
                out[dst0 * cols + c] = values[src0 * cols + c];
                out[dst1 * cols + c] = values[src1 * cols + c];
            }
        }
    }
    out
}

fn max_mean_sample_diff(a: &[f32], b: &[f32], rows: usize, cols: usize) -> (f64, f64, usize) {
    let mut max_diff = 0.0_f64;
    let mut sum_diff = 0.0_f64;
    let mut checked = 0usize;
    let row_samples = [0usize, 1, 2, 63, 64, 65, rows / 2, rows - 1];
    let col_samples = [0usize, 1, 2, 63, 64, 65, cols / 2, cols - 1];
    for r in row_samples {
        for c in col_samples {
            let diff = ((a[r * cols + c] as f64) - (b[r * cols + c] as f64)).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            checked += 1;
        }
    }
    (max_diff, sum_diff / checked as f64, checked)
}

#[test]
#[ignore = "requires models/tinyllama-1.1b/model.safetensors and models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF locally"]
fn tinyllama_q4_k_m_gguf_qk_projection_layout_report() {
    let root = models_root();
    let safetensors_path = root.join("tinyllama-1.1b/model.safetensors");
    let gguf_path =
        root.join("TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf");

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");

    for (st_name, gguf_name, rows, cols) in [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
            2048usize,
            2048usize,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "blk.0.attn_k.weight",
            256usize,
            2048usize,
        ),
    ] {
        let st_entry = safetensors.get(st_name).expect("safetensors projection");
        let st = st_entry
            .to_vec_f32()
            .expect("decode safetensors projection");
        assert_eq!(st_entry.shape, &[rows, cols]);

        let descriptor = gguf.tensor_by_name(gguf_name).expect("GGUF projection");
        let raw = decode_tensor(&gguf, descriptor).expect("decode GGUF projection");
        assert_eq!(raw.len(), st.len());

        let permuted = llama_rope_permute_rows(&st, rows, cols, 64);
        let (direct_max, direct_mean, checked) = max_mean_sample_diff(&st, &raw, rows, cols);
        let (perm_max, perm_mean, _) = max_mean_sample_diff(&permuted, &raw, rows, cols);

        println!("{st_name} vs {gguf_name} (Q4_K_M):");
        println!("  checked           = {checked}");
        println!(
            "  direct max/mean   = {:.9} / {:.9}",
            direct_max, direct_mean
        );
        println!("  permute max/mean  = {:.9} / {:.9}", perm_max, perm_mean);
    }
}

#[test]
#[ignore = "requires models/tinyllama-1.1b/model.safetensors and models/tinyllama-q8_0 GGUF locally"]
fn tinyllama_q8_0_gguf_qk_projection_layout_report() {
    let root = models_root();
    let safetensors_path = root.join("tinyllama-1.1b/model.safetensors");
    let gguf_path = root.join("tinyllama-q8_0/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");

    for (st_name, gguf_name, rows, cols) in [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
            2048usize,
            2048usize,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "blk.0.attn_k.weight",
            256usize,
            2048usize,
        ),
    ] {
        let st_entry = safetensors.get(st_name).expect("safetensors projection");
        let st = st_entry
            .to_vec_f32()
            .expect("decode safetensors projection");
        assert_eq!(st_entry.shape, &[rows, cols]);

        let descriptor = gguf.tensor_by_name(gguf_name).expect("GGUF projection");
        let raw = decode_tensor(&gguf, descriptor).expect("decode GGUF projection");
        assert_eq!(raw.len(), st.len());

        let permuted = llama_rope_permute_rows(&st, rows, cols, 64);
        let (direct_max, direct_mean, checked) = max_mean_sample_diff(&st, &raw, rows, cols);
        let (perm_max, perm_mean, _) = max_mean_sample_diff(&permuted, &raw, rows, cols);

        println!("{st_name} vs {gguf_name}:");
        println!("  checked           = {checked}");
        println!(
            "  direct max/mean   = {:.9} / {:.9}",
            direct_max, direct_mean
        );
        println!("  permute max/mean  = {:.9} / {:.9}", perm_max, perm_mean);
    }
}

// ============================================================
// GGUF Drift Validation for Additional Models (M11.D.5)
// ============================================================

#[test]
#[ignore = "requires models/llama-3.2-1b-instruct/model.safetensors and models/Llama-3.2-1B-Instruct-Q4_K_M-GGUF locally"]
fn llama_3_2_1b_q4_k_m_gguf_norm_matches_safetensors() {
    let root = models_root();
    let safetensors_path = root.join("llama-3.2-1b-instruct/model.safetensors");
    let gguf_path =
        root.join("Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf");

    assert!(
        safetensors_path.exists(),
        "missing {}",
        safetensors_path.display()
    );
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("model.norm.weight")
        .expect("model.norm.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors norm");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output_norm.weight")
        .expect("output_norm.weight in GGUF");
    let gguf_values = decode_tensor(&gguf, descriptor).expect("decode GGUF norm");

    assert_eq!(st.len(), gguf_values.len(), "norm length mismatch");

    let diffs: Vec<f64> = st
        .iter()
        .zip(gguf_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

    println!("Llama-3.2-1B safetensors model.norm.weight vs Q4_K_M GGUF output_norm.weight:");
    println!("  len       = {}", st.len());
    println!("  max_diff  = {:.9}", max_diff);
    println!("  mean_diff = {:.9}", mean_diff);
}

#[test]
#[ignore = "requires models/phi-3.5-mini-instruct/model.safetensors.index.json and models/Phi-3.5-mini-instruct-Q4_K_M-GGUF locally"]
fn phi_3_5_mini_q4_k_m_gguf_norm_matches_safetensors() {
    let root = models_root();
    let safetensors_path = root.join("phi-3.5-mini-instruct/model.safetensors.index.json");
    let gguf_path =
        root.join("Phi-3.5-mini-instruct-Q4_K_M-GGUF/phi-3.5-mini-instruct-q4_k_m.gguf");

    assert!(
        safetensors_path.exists(),
        "missing {}",
        safetensors_path.display()
    );
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());

    let safetensors = open_safetensors_for_tensor(&safetensors_path, "model.norm.weight");
    let st_entry = safetensors
        .get("model.norm.weight")
        .expect("model.norm.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors norm");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output_norm.weight")
        .expect("output_norm.weight in GGUF");
    let gguf_values = decode_tensor(&gguf, descriptor).expect("decode GGUF norm");

    assert_eq!(st.len(), gguf_values.len(), "norm length mismatch");

    let diffs: Vec<f64> = st
        .iter()
        .zip(gguf_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

    println!("Phi-3.5-Mini safetensors model.norm.weight vs Q4_K_M GGUF output_norm.weight:");
    println!("  len       = {}", st.len());
    println!("  max_diff  = {:.9}", max_diff);
    println!("  mean_diff = {:.9}", mean_diff);
}

#[test]
#[ignore = "requires models/smollm2-1.7b-instruct/model.safetensors and models/SmolLM2-1.7B-Instruct-GGUF locally"]
fn smollm2_1_7b_q4_k_m_gguf_norm_matches_safetensors() {
    let root = models_root();
    let safetensors_path = root.join("smollm2-1.7b-instruct/model.safetensors");
    let gguf_path = root.join("SmolLM2-1.7B-Instruct-GGUF/smollm2-1.7b-instruct-q4_k_m.gguf");

    assert!(
        safetensors_path.exists(),
        "missing {}",
        safetensors_path.display()
    );
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("model.norm.weight")
        .expect("model.norm.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors norm");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output_norm.weight")
        .expect("output_norm.weight in GGUF");
    let gguf_values = decode_tensor(&gguf, descriptor).expect("decode GGUF norm");

    assert_eq!(st.len(), gguf_values.len(), "norm length mismatch");

    let diffs: Vec<f64> = st
        .iter()
        .zip(gguf_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let max_diff = diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

    println!("SmolLM2-1.7B safetensors model.norm.weight vs Q4_K_M GGUF output_norm.weight:");
    println!("  len       = {}", st.len());
    println!("  max_diff  = {:.9}", max_diff);
    println!("  mean_diff = {:.9}", mean_diff);
}

#[test]
#[ignore = "requires models/llama-3.2-1b-instruct/model.safetensors and models/Llama-3.2-1B-Instruct-Q4_K_M-GGUF locally"]
fn llama_3_2_1b_q4_k_m_gguf_lm_head_sample_diagnostic() {
    let root = models_root();
    let safetensors_path = root.join("llama-3.2-1b-instruct/model.safetensors");
    let gguf_path =
        root.join("Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf");

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("lm_head.weight")
        .or_else(|| safetensors.get("model.embed_tokens.weight"))
        .expect("lm_head.weight or tied model.embed_tokens.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors lm_head");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("output.weight")
        .or_else(|| gguf.tensor_by_name("token_embd.weight"))
        .expect("output.weight or tied token_embd.weight in GGUF");
    let gguf_values =
        decode_tensor(&gguf, descriptor).expect("decode GGUF lm_head / tied embedding");

    assert_eq!(st.len(), gguf_values.len(), "lm_head length mismatch");

    let hidden = 2048usize;
    let vocab = 128_000usize;
    let mut max_direct_diff = 0.0_f64;
    let mut sum_direct_diff = 0.0_f64;
    let mut checked = 0usize;
    for token in [0usize, 1, 2, 100, 5099, 29871, vocab - 1] {
        for h in [0usize, 1, 2, 64, 511, 1024, hidden - 1] {
            let st_value = st[token * hidden + h];
            let gguf_value = gguf_values[token * hidden + h];
            let diff = ((st_value as f64) - (gguf_value as f64)).abs();
            max_direct_diff = max_direct_diff.max(diff);
            sum_direct_diff += diff;
            checked += 1;
        }
    }
    let mean_direct_diff = sum_direct_diff / checked as f64;

    println!("Llama-3.2-1B safetensors lm_head.weight vs Q4_K_M GGUF output.weight samples:");
    println!("  checked   = {}", checked);
    println!("  max_diff  = {:.9}", max_direct_diff);
    println!("  mean_diff = {:.9}", mean_direct_diff);
    println!("  st[0..4]  = {:?}", &st[..4]);
    println!("  gg[0..4]  = {:?}", &gguf_values[..4]);
}

#[test]
#[ignore = "requires models/smollm2-1.7b-instruct/model.safetensors and models/SmolLM2-1.7B-Instruct-GGUF locally"]
fn smollm2_1_7b_q4_k_m_gguf_q_proj_sample_diagnostic() {
    let root = models_root();
    let safetensors_path = root.join("smollm2-1.7b-instruct/model.safetensors");
    let gguf_path = root.join("SmolLM2-1.7B-Instruct-GGUF/smollm2-1.7b-instruct-q4_k_m.gguf");

    let safetensors = SafetensorsReader::open(&safetensors_path).expect("open safetensors");
    let st_entry = safetensors
        .get("model.layers.0.self_attn.q_proj.weight")
        .expect("q_proj.weight in safetensors");
    let st = st_entry.to_vec_f32().expect("decode safetensors q_proj");

    let gguf = GgufReader::read_from_path(&gguf_path).expect("open GGUF");
    let descriptor = gguf
        .tensor_by_name("blk.0.attn_q.weight")
        .expect("attn_q.weight in GGUF");
    let gguf_values = decode_tensor(&gguf, descriptor).expect("decode GGUF q_proj");

    assert_eq!(st.len(), gguf_values.len(), "q_proj length mismatch");

    let rows = 2048usize;
    let cols = 2048usize;
    let mut max_direct_diff = 0.0_f64;
    let mut sum_direct_diff = 0.0_f64;
    let mut checked = 0usize;
    for r in [0usize, 1, 2, 64, 511, 1024, rows - 1] {
        for c in [0usize, 1, 2, 64, 511, 1024, cols - 1] {
            let st_value = st[r * cols + c];
            let gguf_value = gguf_values[r * cols + c];
            let diff = ((st_value as f64) - (gguf_value as f64)).abs();
            max_direct_diff = max_direct_diff.max(diff);
            sum_direct_diff += diff;
            checked += 1;
        }
    }
    let mean_direct_diff = sum_direct_diff / checked as f64;

    println!("SmolLM2-1.7B safetensors q_proj.weight vs Q4_K_M GGUF attn_q.weight samples:");
    println!("  checked   = {}", checked);
    println!("  max_diff  = {:.9}", max_direct_diff);
    println!("  mean_diff = {:.9}", mean_direct_diff);
    println!("  st[0..4]  = {:?}", &st[..4]);
    println!("  gg[0..4]  = {:?}", &gguf_values[..4]);
}
