//! Numerical validation of Atenia's TinyLlama forward against a
//! PyTorch reference (M4.5-d.1).
//!
//! Runs the full Atenia pipeline (build → load → forward) with the
//! same token IDs that `tests/fixtures/tinyllama_reference/generate.py`
//! used in PyTorch, then reports element-wise drift between the two
//! logit tensors. The first run is exploratory: it does not assert a
//! tight tolerance, only that the drift is not catastrophically large
//! (would indicate a bug rather than F32-vs-BF16 precision).
//!
//! Run with:
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "...\\model.safetensors"
//! cargo test --test tinyllama_numerical_validation_test --release \
//!     -- --ignored --nocapture
//! ```
//!
//! `--release` is strongly recommended: M4.5-c showed a debug-build
//! forward of ~70 s; release should drop that into the low seconds.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{LlamaConfig, LlamaRuntime, build_llama, llama_weight_mapper};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EMBEDDED_CONFIG: &str = r#"{
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

#[test]
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH env var + PyTorch fixture in tests/fixtures/tinyllama_reference/"]
fn tinyllama_logits_match_pytorch_reference() {
    println!("\n=== TinyLlama Numerical Validation vs PyTorch (M4.5-d.1) ===");

    // ---- 1. Load fixtures ----
    let fixture_dir = PathBuf::from("tests/fixtures/tinyllama_reference");
    let inputs_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(fixture_dir.join("inputs.json"))
            .expect("missing tests/fixtures/tinyllama_reference/inputs.json"),
    )
    .expect("malformed inputs.json");
    let expected_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(fixture_dir.join("expected_logits.json"))
            .expect("missing tests/fixtures/tinyllama_reference/expected_logits.json"),
    )
    .expect("malformed expected_logits.json");

    let token_ids_f32: Vec<f32> = inputs_json["token_ids"]
        .as_array()
        .expect("token_ids must be array")
        .iter()
        .map(|v| v.as_i64().expect("token_id must be int") as f32)
        .collect();
    let seq_len = token_ids_f32.len();

    let expected_logits: Vec<f32> = expected_json["values"]
        .as_array()
        .expect("expected values must be array")
        .iter()
        .map(|v| v.as_f64().expect("logit must be float") as f32)
        .collect();

    let pytorch_predicted_id = expected_json["predicted_token_id"].as_u64().unwrap() as usize;
    let pytorch_max_abs = expected_json["max_abs"].as_f64().unwrap() as f32;
    let pytorch_mean_abs = expected_json["mean_abs"].as_f64().unwrap() as f32;

    println!("PyTorch reference loaded:");
    println!("  Tokens: {:?}", token_ids_f32);
    println!("  Logits values: {}", expected_logits.len());
    println!("  Predicted token id: {}", pytorch_predicted_id);
    println!("  Max abs: {:.4}", pytorch_max_abs);
    println!("  Mean abs: {:.4}", pytorch_mean_abs);

    // ---- 2. Build + load Atenia graph (same recipe as M4.5-c) ----
    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH to TinyLlama model.safetensors");

    let config = LlamaConfig::from_json_str(EMBEDDED_CONFIG)
        .expect("failed to parse embedded TinyLlama config");
    let runtime = LlamaRuntime {
        batch: 1,
        seq: seq_len,
    };

    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), 201);

    println!("\nLoading weights...");
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

    // ---- 3. Forward ----
    println!("Running forward...");
    let tokens = Tensor::new_cpu(vec![1, seq_len], token_ids_f32.clone());
    let forward_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward: {:.2}s", forward_start.elapsed().as_secs_f32());

    let atenia_logits = outputs[0].as_cpu_slice();
    assert_eq!(
        atenia_logits.len(),
        expected_logits.len(),
        "logit count mismatch: atenia {} vs pytorch {}",
        atenia_logits.len(),
        expected_logits.len()
    );

    // ---- 4. Element-wise drift ----
    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    let mut count_over_1e2 = 0_usize;
    let mut count_over_1e1 = 0_usize;
    let mut count_over_1 = 0_usize;
    let mut argmax_idx_of_worst = 0_usize;

    for (i, (got, expected)) in atenia_logits.iter().zip(expected_logits.iter()).enumerate() {
        let abs_diff = (got - expected).abs();
        let rel_diff = if expected.abs() > 1e-6 {
            abs_diff / expected.abs()
        } else {
            abs_diff
        };
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            argmax_idx_of_worst = i;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
        sum_abs_diff += abs_diff as f64;
        if abs_diff > 1e-2 {
            count_over_1e2 += 1;
        }
        if abs_diff > 1e-1 {
            count_over_1e1 += 1;
        }
        if abs_diff > 1.0 {
            count_over_1 += 1;
        }
    }
    let n = atenia_logits.len();
    let mean_abs_diff = (sum_abs_diff / n as f64) as f32;

    let atenia_max_abs = atenia_logits
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let atenia_mean_abs: f32 = atenia_logits.iter().map(|v| v.abs()).sum::<f32>() / n as f32;

    println!("\nGlobal stats:");
    println!(
        "  PyTorch:  max_abs={:.4} mean_abs={:.4}",
        pytorch_max_abs, pytorch_mean_abs
    );
    println!(
        "  Atenia:   max_abs={:.4} mean_abs={:.4}",
        atenia_max_abs, atenia_mean_abs
    );

    println!("\nElement-wise drift (Atenia vs PyTorch):");
    println!("  Max abs diff:  {:.6}", max_abs_diff);
    println!("  Max rel diff:  {:.6}", max_rel_diff);
    println!("  Mean abs diff: {:.6}", mean_abs_diff);
    println!(
        "  Worst at flat idx {} (pos {} / vocab {}): atenia={} pytorch={}",
        argmax_idx_of_worst,
        argmax_idx_of_worst / config.vocab_size,
        argmax_idx_of_worst % config.vocab_size,
        atenia_logits[argmax_idx_of_worst],
        expected_logits[argmax_idx_of_worst],
    );
    println!(
        "  diff > 1e-2: {:>7} ({:>5.2}%)",
        count_over_1e2,
        100.0 * count_over_1e2 as f32 / n as f32
    );
    println!(
        "  diff > 1e-1: {:>7} ({:>5.2}%)",
        count_over_1e1,
        100.0 * count_over_1e1 as f32 / n as f32
    );
    println!(
        "  diff > 1.0:  {:>7} ({:>5.2}%)",
        count_over_1,
        100.0 * count_over_1 as f32 / n as f32
    );

    // ---- 5. Argmax comparison ----
    let last_pos_start = (seq_len - 1) * config.vocab_size;
    let last_pos_end = last_pos_start + config.vocab_size;
    let atenia_last = &atenia_logits[last_pos_start..last_pos_end];
    let (atenia_pred_id, atenia_pred_logit) =
        atenia_last
            .iter()
            .enumerate()
            .fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
    let pytorch_logit_for_atenia_pick = expected_logits[last_pos_start + atenia_pred_id];
    let atenia_logit_for_pytorch_pick = atenia_last[pytorch_predicted_id];
    println!("\nArgmax of last-position logits:");
    println!(
        "  PyTorch:  id={:5} logit={:.4}",
        pytorch_predicted_id,
        expected_logits[last_pos_start + pytorch_predicted_id]
    );
    println!(
        "  Atenia:   id={:5} logit={:.4}",
        atenia_pred_id, atenia_pred_logit
    );
    println!("  Match:    {}", atenia_pred_id == pytorch_predicted_id);
    println!(
        "  PyTorch logit at Atenia's pick:  {:.4}",
        pytorch_logit_for_atenia_pick
    );
    println!(
        "  Atenia  logit at PyTorch's pick: {:.4}",
        atenia_logit_for_pytorch_pick
    );

    // ---- 6. Catastrophic-drift safeguard ----
    // First run: do not assert tight tolerance. Only fail if drift is
    // so large it must be a bug, not BF16-vs-F32 precision.
    assert!(
        max_abs_diff < 5.0,
        "drift catastrophically large (max_abs_diff = {}); likely a bug, not precision",
        max_abs_diff
    );

    println!("\n=== M4.5-d.1: comparison run completed ===\n");
}
