//! End-to-end Qwen 2.5 1.5B Instruct inference smoke test (M4.6 Phase B.4).
//!
//! Loads the real `model.safetensors` of Qwen/Qwen2.5-1.5B-Instruct,
//! runs a forward pass through the full 28-layer Llama-family graph
//! with QKV biases enabled, and validates that the logits come out
//! with the expected shape, are finite, and have plausible
//! magnitudes.
//!
//! This is the first end-to-end exercise of:
//! - Phase B.1 — `model_type`-aware `effective_attention_bias`
//! - Phase B.2 — builder branch that registers q/k/v bias params
//!   and inserts BroadcastAdd post-projection
//! - Phase B.3 — weight-mapper handlers for the three new bias
//!   tensor families (84 entries for 28 layers)
//!
//! Numerical comparison vs PyTorch F64 ground truth lives in
//! `tests/qwen25_numerical_validation_test.rs` (Phase B.5).
//!
//! Marked `#[ignore]`. Run with:
//!
//! ```powershell
//! $env:QWEN25_SAFETENSORS_PATH = "F:\Proyectos\artenia_engine\atenia-engine\models\qwen2.5-1.5b-instruct\model.safetensors"
//! cargo test --test qwen25_end_to_end_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::path::Path;

/// Verbatim snapshot of `models/qwen2.5-1.5b-instruct/config.json`.
/// Note the absence of `attention_bias` — Qwen2 omits the field and
/// the parser resolves the effective value to `true` via
/// `LlamaConfig::effective_attention_bias` (see Phase B.1).
const EMBEDDED_QWEN25_CONFIG: &str = r#"{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"#;

#[test]
#[ignore = "requires QWEN25_SAFETENSORS_PATH env var pointing to Qwen 2.5 1.5B model.safetensors"]
fn qwen25_loads_and_executes_forward_with_real_weights() {
    println!("\n=== Qwen 2.5 1.5B Instruct End-to-End Forward Test (M4.6 B.4) ===");

    let path = env::var("QWEN25_SAFETENSORS_PATH")
        .expect("Set QWEN25_SAFETENSORS_PATH to model.safetensors");
    println!("Loading from: {}", path);

    let config = LlamaConfig::from_json_str(EMBEDDED_QWEN25_CONFIG)
        .expect("failed to parse embedded Qwen 2.5 config");
    println!(
        "Config: vocab={} hidden={} layers={} heads={} kv_heads={} \
         intermediate={} rope_theta={} eps={} tied={} attn_bias={}",
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.intermediate_size,
        config.rope_theta,
        config.rms_norm_eps,
        config.tie_word_embeddings,
        config.effective_attention_bias(),
    );
    assert!(
        config.effective_attention_bias(),
        "Qwen 2.5 must resolve attention_bias=true via model_type=qwen2"
    );

    let runtime = LlamaRuntime { batch: 1, seq: 4 };
    println!("Runtime: batch={} seq={}", runtime.batch, runtime.seq);

    let build_start = std::time::Instant::now();
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    println!(
        "Graph built: {} parameters in {:.2}s",
        handles.param_ids.len(),
        build_start.elapsed().as_secs_f32()
    );

    // Qwen 2.5 1.5B with tied embeddings + QKV bias:
    //   1 (embed) + 28 × (9 weights + 3 biases) + 1 (final norm) = 338.
    // No `lm_head.weight` (tied).
    assert_eq!(
        handles.param_ids.len(),
        338,
        "expected 338 params (28 layers × 12 + 2 globals, no lm_head), got {}",
        handles.param_ids.len()
    );
    assert!(
        !handles.param_names.iter().any(|n| n == "lm_head.weight"),
        "lm_head.weight must NOT be in param_names under tied embeddings"
    );

    // Sanity check: 84 bias params should be registered (3 per layer × 28).
    let bias_count = handles
        .param_names
        .iter()
        .filter(|n| n.ends_with(".bias"))
        .count();
    assert_eq!(
        bias_count, 84,
        "expected 84 QKV bias params (3 × 28 layers), got {}",
        bias_count
    );

    println!("Reading safetensors...");
    let reader_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let in_file_count = reader.iter().count();
    println!(
        "Reader opened in {:.2}s ({} tensors in file)",
        reader_start.elapsed().as_secs_f32(),
        in_file_count
    );
    assert_eq!(
        in_file_count, 338,
        "expected 338 tensors in Qwen 2.5 1.5B safetensors, got {}",
        in_file_count
    );

    let mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("failed to build weight mapper");

    println!("Loading weights into graph...");
    let load_start = std::time::Instant::now();
    let report = mapper
        .load_into(&mut graph, &reader)
        .expect("failed to load weights");
    println!(
        "Loaded {} tensors, {} skipped, {} missing in {:.2}s",
        report.loaded,
        report.skipped.len(),
        report.missing.len(),
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(
        report.loaded, 338,
        "expected 338 tensors loaded, got {}",
        report.loaded
    );
    assert!(
        report.missing.is_empty(),
        "missing weights: {:?}",
        report.missing
    );

    drop(reader);
    println!("Reader dropped (raw bytes freed)");

    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    println!("Input tokens: [1, 100, 200, 300]");

    println!("Executing forward pass...");
    let forward_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward completed in {:.2}s", forward_start.elapsed().as_secs_f32());

    assert_eq!(outputs.len(), 1, "expected 1 output, got {}", outputs.len());
    let logits = &outputs[0];
    let expected_shape = vec![1, 4, 151_936];
    assert_eq!(
        logits.shape, expected_shape,
        "logits shape mismatch: expected {:?}, got {:?}",
        expected_shape, logits.shape
    );
    println!("Logits shape: {:?}", logits.shape);

    let slice = logits.as_cpu_slice();
    let finite_count = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite_count,
        slice.len(),
        "non-finite logits: {}/{} finite",
        finite_count,
        slice.len()
    );
    println!("All {} logit values finite", slice.len());

    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
    assert!(max_abs < 1000.0, "logits suspiciously large: {}", max_abs);
    println!("Logit stats: max |v|={:.4} mean |v|={:.4}", max_abs, mean_abs);

    let last = &slice[3 * config.vocab_size..4 * config.vocab_size];
    let (pred_id, pred_logit) = last.iter().enumerate().fold(
        (0_usize, f32::NEG_INFINITY),
        |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
    );
    println!(
        "Predicted next token at position 3: id={} logit={:.4}",
        pred_id, pred_logit
    );

    println!("\n=== Qwen 2.5 1.5B M4.6 B.4 PASSED ===\n");
}
