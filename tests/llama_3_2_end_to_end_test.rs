//! End-to-end Llama 3.2 1B Instruct inference smoke test (M4.6 Phase C.4).
//!
//! Loads the real `model.safetensors` of meta-llama/Llama-3.2-1B-Instruct,
//! runs a forward pass through the full 16-layer Llama-family graph
//! with the Llama 3 piecewise RoPE scaling enabled, and validates that
//! the logits come out with the expected shape, are finite, and have
//! plausible magnitudes.
//!
//! This is the first end-to-end exercise of:
//! - Phase C.1 — `head_dim` + `rope_scaling` parsing.
//! - Phase C.2 — `compute_inv_freqs_llama3` piecewise transform.
//! - Phase C.3 — `NodeType::RoPE { scaling: Some(_) }` wired into
//!   forward + backward, dispatched by `build_llama`.
//!
//! Numerical comparison vs PyTorch F64 ground truth lives in
//! `tests/llama_3_2_numerical_validation_test.rs` (Phase C.5), and
//! a long-context check that actually exercises the mid + low
//! frequency bands lives in
//! `tests/llama_3_2_long_context_validation_test.rs` (Phase C.6).
//!
//! Important caveat: with seq_len = 4 every angle `t · inv_freq[i]`
//! is small enough that the high-frequency band dominates and
//! `inv_freq_scaled ≈ inv_freq_unscaled`. A passing forward here
//! does NOT prove the scaling is correctly wired — only that nothing
//! exploded. The C.6 long-context test is the falsifier.
//!
//! Marked `#[ignore]`. Run with:
//!
//! ```powershell
//! $env:LLAMA32_SAFETENSORS_PATH = "F:\Proyectos\artenia_engine\atenia-engine\models\llama-3.2-1b-instruct\model.safetensors"
//! cargo test --test llama_3_2_end_to_end_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime, RopeScaling,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::path::Path;

/// Verbatim snapshot of `models/llama-3.2-1b-instruct/config.json`.
/// Carries an explicit `head_dim: 64` AND a populated
/// `rope_scaling` block — both new for Llama 3.x.
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

#[test]
#[ignore = "requires LLAMA32_SAFETENSORS_PATH env var pointing to Llama 3.2 1B model.safetensors"]
fn llama_3_2_loads_and_executes_forward_with_real_weights() {
    println!("\n=== Llama 3.2 1B Instruct End-to-End Forward Test (M4.6 C.4) ===");

    let path = env::var("LLAMA32_SAFETENSORS_PATH")
        .expect("Set LLAMA32_SAFETENSORS_PATH to model.safetensors");
    println!("Loading from: {}", path);

    let config = LlamaConfig::from_json_str(EMBEDDED_LLAMA_3_2_CONFIG)
        .expect("failed to parse embedded Llama 3.2 config");
    println!(
        "Config: vocab={} hidden={} layers={} heads={} kv_heads={} \
         head_dim={} intermediate={} rope_theta={} eps={} tied={}",
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.effective_head_dim(),
        config.intermediate_size,
        config.rope_theta,
        config.rms_norm_eps,
        config.tie_word_embeddings,
    );

    // Phase C.1 / C.3 invariants.
    assert_eq!(config.effective_head_dim(), 64);
    let scaling = config
        .effective_rope_scaling()
        .expect("Llama 3.2 must carry rope_scaling");
    match scaling {
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        } => {
            assert!((*factor - 32.0).abs() < 1e-6);
            assert!((*low_freq_factor - 1.0).abs() < 1e-6);
            assert!((*high_freq_factor - 4.0).abs() < 1e-6);
            assert_eq!(*original_max_position_embeddings, 8192);
        }
    }
    // Llama 3.2 1B has no QKV biases (Phase B path inactive here).
    assert_eq!(config.effective_attention_bias(), false);

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

    // Llama 3.2 1B with tied embeddings, no QKV biases:
    //   1 (embed) + 16 × 9 + 1 (final norm) = 146.
    // The safetensors file ships exactly 146 tensors — no
    // `lm_head.weight` (tied) and no `*.bias` (attention_bias=false),
    // so graph params and on-disk tensors match one-to-one.
    assert_eq!(
        handles.param_ids.len(),
        146,
        "expected 146 params (1 embed + 16 × 9 + 1 final norm, no lm_head, no QKV biases), got {}",
        handles.param_ids.len()
    );
    assert!(
        !handles.param_names.iter().any(|n| n == "lm_head.weight"),
        "lm_head.weight must NOT be in param_names under tied embeddings"
    );
    // Phase B regression: Llama 3.2 has zero QKV biases.
    let bias_count = handles
        .param_names
        .iter()
        .filter(|n| n.ends_with(".bias"))
        .count();
    assert_eq!(
        bias_count, 0,
        "expected 0 bias params for Llama 3.2 (no attention_bias), got {}",
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
        report.loaded, 146,
        "expected 146 graph params populated, got {}",
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
    let expected_shape = vec![1, 4, 128_256];
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

    println!("\n=== Llama 3.2 1B M4.6 C.4 PASSED ===\n");
    println!(
        "NOTE: a 4-token forward exercises the high-frequency band only. \
         Long-context correctness is asserted in C.6."
    );
}
