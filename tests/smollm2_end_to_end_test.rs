//! End-to-end SmolLM2 inference smoke test (M4.6, Phase A.1 validation).
//!
//! Loads the real `model.safetensors` of HuggingFaceTB/SmolLM2-1.7B-Instruct,
//! runs a forward pass through the full 24-layer graph using the Phase A.1
//! tied-embedding path, and validates that the logits come out with the
//! expected shape, are finite, and have plausible magnitudes.
//!
//! Numerical comparison vs PyTorch lives in
//! `tests/smollm2_numerical_validation_test.rs`.
//!
//! Marked `#[ignore]`. Run with:
//!
//! ```powershell
//! $env:SMOLLM2_SAFETENSORS_PATH = "F:\Proyectos\artenia_engine\atenia-engine\models\smollm2-1.7b-instruct\model.safetensors"
//! cargo test --test smollm2_end_to_end_test --release \
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

/// Embedded SmolLM2 1.7B Instruct config.json.
const EMBEDDED_SMOLLM2_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "pad_token_id": 2,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 130000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "use_cache": true,
  "vocab_size": 49152
}"#;

#[test]
#[ignore = "requires SMOLLM2_SAFETENSORS_PATH env var pointing to SmolLM2 model.safetensors"]
fn smollm2_loads_and_executes_forward_with_real_weights() {
    println!("\n=== SmolLM2 1.7B End-to-End Forward Test (M4.6 A.1 validation) ===");

    let path = env::var("SMOLLM2_SAFETENSORS_PATH")
        .expect("Set SMOLLM2_SAFETENSORS_PATH to model.safetensors");
    println!("Loading from: {}", path);

    let config = LlamaConfig::from_json_str(EMBEDDED_SMOLLM2_CONFIG)
        .expect("failed to parse embedded SmolLM2 config");
    println!(
        "Config: vocab={} hidden={} layers={} heads={} kv_heads={} \
         intermediate={} rope_theta={} tied={}",
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.intermediate_size,
        config.rope_theta,
        config.tie_word_embeddings,
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

    // SmolLM2 with tied embeddings: 1 (embed) + 24 × 9 + 1 (final norm) = 218.
    // No `lm_head.weight` because of A.1.
    assert_eq!(
        handles.param_ids.len(),
        218,
        "expected 218 params (no lm_head, tied), got {}",
        handles.param_ids.len()
    );
    assert!(
        !handles.param_names.iter().any(|n| n == "lm_head.weight"),
        "lm_head.weight must NOT be in param_names under tied embeddings"
    );

    println!("Reading safetensors...");
    let reader_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    println!(
        "Reader opened in {:.2}s ({} tensors in file)",
        reader_start.elapsed().as_secs_f32(),
        reader.iter().count()
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
        report.loaded, 218,
        "expected 218 tensors loaded, got {}",
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
    let expected_shape = vec![1, 4, 49_152];
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

    println!("\n=== SmolLM2 1.7B M4.6 PASSED ===\n");
}
