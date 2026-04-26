//! End-to-end TinyLlama inference smoke test (M4.5-c).
//!
//! Loads the real `model.safetensors` checkpoint, runs a forward
//! pass through the full 22-layer graph, and validates that the
//! logits come out with the expected shape, are finite, and have
//! plausible magnitudes. Numerical validation against PyTorch is
//! M4.5-d.
//!
//! Marked `#[ignore]` because the model file is ~2 GB and is not
//! shipped with the repo. Run manually with:
//!
//! ```powershell
//! # PowerShell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "F:\Proyectos\artenia_engine\atenia-engine\models\tinyllama-1.1b\model.safetensors"
//! cargo test --test tinyllama_end_to_end_test -- --ignored --nocapture
//! ```
//!
//! ```bash
//! # Bash
//! TINYLLAMA_SAFETENSORS_PATH=/path/to/model.safetensors \
//!     cargo test --test tinyllama_end_to_end_test -- --ignored --nocapture
//! ```
//!
//! The `--nocapture` flag is required to see the timing and stats
//! diagnostics. Without it the test still runs but cargo hides
//! the `println!` output.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::tinyllama::{
    build_tinyllama, tinyllama_weight_mapper, TinyLlamaConfig, TinyLlamaRuntime,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::path::Path;

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
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH env var pointing to TinyLlama model.safetensors"]
fn tinyllama_loads_and_executes_forward_with_real_weights() {
    println!("\n=== TinyLlama End-to-End Forward Test (M4.5-c) ===");

    // ---- 1. Setup ----
    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH to model.safetensors");
    println!("Loading from: {}", path);

    // ---- 2. Parse config ----
    let config = TinyLlamaConfig::from_json_str(EMBEDDED_CONFIG)
        .expect("failed to parse embedded TinyLlama config");
    println!(
        "Config: vocab={} hidden={} layers={} heads={} kv_heads={} intermediate={}",
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.intermediate_size,
    );

    // ---- 3. Build graph ----
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    println!("Runtime: batch={} seq={}", runtime.batch, runtime.seq);

    let build_start = std::time::Instant::now();
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_tinyllama(&mut gb, &config, &runtime, token_input_id);
    let _output_id = gb.output(handles.logits_id);
    let mut graph = gb.build();
    let build_elapsed = build_start.elapsed();

    println!(
        "Graph built: {} parameters in {:.2}s",
        handles.param_ids.len(),
        build_elapsed.as_secs_f32()
    );
    assert_eq!(handles.param_ids.len(), 201);

    // ---- 4. Load weights ----
    println!("Reading safetensors...");
    let reader_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path))
        .expect("failed to open safetensors");
    let reader_elapsed = reader_start.elapsed();
    println!(
        "Reader opened in {:.2}s ({} tensors in file)",
        reader_elapsed.as_secs_f32(),
        reader.iter().count()
    );

    let mapper = tinyllama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("failed to build weight mapper");

    println!("Loading weights into graph...");
    let load_start = std::time::Instant::now();
    let report = mapper
        .load_into(&mut graph, &reader)
        .expect("failed to load weights");
    let load_elapsed = load_start.elapsed();

    println!(
        "Loaded {} tensors, {} skipped, {} missing in {:.2}s",
        report.loaded,
        report.skipped.len(),
        report.missing.len(),
        load_elapsed.as_secs_f32()
    );
    assert_eq!(
        report.loaded, 201,
        "expected 201 tensors loaded, got {}",
        report.loaded
    );
    assert!(
        report.missing.is_empty(),
        "missing weights: {:?}",
        report.missing
    );
    if !report.skipped.is_empty() {
        println!(
            "  (skipped {} extra tensors in file: first few {:?})",
            report.skipped.len(),
            report.skipped.iter().take(3).collect::<Vec<_>>()
        );
    }

    // Free the ~2 GB raw safetensors buffer before activations allocate.
    drop(reader);
    println!("Reader dropped (raw bytes freed)");

    // ---- 5. Forward pass ----
    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    println!("Input tokens: [1, 100, 200, 300] (BOS + 3 arbitrary IDs)");

    println!("Executing forward pass...");
    let forward_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    let forward_elapsed = forward_start.elapsed();
    println!("Forward completed in {:.2}s", forward_elapsed.as_secs_f32());

    // ---- 6. Output shape check ----
    assert_eq!(outputs.len(), 1, "expected 1 output, got {}", outputs.len());
    let logits = &outputs[0];
    let expected_shape = vec![1, 4, 32_000];
    assert_eq!(
        logits.shape, expected_shape,
        "logits shape mismatch: expected {:?}, got {:?}",
        expected_shape, logits.shape
    );
    println!("Logits shape: {:?}", logits.shape);

    // ---- 7. Sanity on logit values ----
    let logits_slice = logits.as_cpu_slice();

    let finite_count = logits_slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite_count,
        logits_slice.len(),
        "non-finite logits detected: {}/{} finite",
        finite_count,
        logits_slice.len()
    );
    println!("All {} logit values finite", logits_slice.len());

    let max_abs = logits_slice
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let min_abs_nonzero = logits_slice
        .iter()
        .map(|v| v.abs())
        .filter(|&v| v > 0.0)
        .fold(f32::INFINITY, f32::min);
    let mean_abs: f32 =
        logits_slice.iter().map(|v| v.abs()).sum::<f32>() / logits_slice.len() as f32;

    assert!(
        max_abs < 1000.0,
        "logits suspiciously large: max_abs = {}",
        max_abs
    );
    println!("Logit value stats:");
    println!("  Max |v|:         {:.4}", max_abs);
    println!("  Min nonzero |v|: {:.6}", min_abs_nonzero);
    println!("  Mean |v|:        {:.4}", mean_abs);

    // ---- 8. Argmax of the last-position logits (predicted next token) ----
    let vocab = config.vocab_size;
    let last_token_logits = &logits_slice[3 * vocab..4 * vocab];
    let (predicted_id, max_logit) =
        last_token_logits
            .iter()
            .enumerate()
            .fold(
                (0_usize, f32::NEG_INFINITY),
                |(best_i, best_v), (i, &v)| {
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_i, best_v)
                    }
                },
            );
    println!(
        "Predicted next token at position 3: id={} logit={:.4}",
        predicted_id, max_logit
    );
    // Note: WHICH token is predicted is not validated here — that's
    // M4.5-d's job (numerical match against a PyTorch reference run
    // with the same tokens). M4.5-c only verifies that the model
    // runs end-to-end without crashing.

    println!("\n=== M4.5-c PASSED ===\n");
}
