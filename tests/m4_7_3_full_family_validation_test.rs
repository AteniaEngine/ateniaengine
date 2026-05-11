//! M4.7.3.f — full 4-model F64 re-validation under M4.7.3
//! residency-aware GPU MatMul dispatch.
//!
//! Re-runs the M4.7.2.e validation on each of the four M4.6
//! production checkpoints (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5
//! 1.5B, Llama 3.2 1B) with the M4.7.3 dispatch active, and
//! asserts:
//!
//!   1. Atenia F32 max drift vs F64 truth stays under the
//!      ADR-004 threshold (`< 0.5`). Per the M4.7.3 investigation
//!      F32 GPU MatMul kernel drift is ~5e-5 absolute, so the
//!      end-to-end drift should remain dominated by BF16 storage
//!      decode (~1e-4 to ~3e-2 across the family) — the same
//!      envelope as M4.7.2.e.
//!   2. Argmax MATCH on every position (4/4 per model — kill-
//!      switch on any miss).
//!   3. Reports the `try_gpu_matmul` counters for observability.
//!      Not asserted because the default APX mode (4.19) runs
//!      `record_tape = true` and the MatMul executor arm gates
//!      `try_gpu_matmul` behind `!in_gpu_segment` — and every
//!      MatMul is its own 1-node GPU segment in `GpuPlan`, so the
//!      counter stays at 0 even when the GPU is actively executing
//!      these kernels via the legacy APX 4.3 dispatcher
//!      (`dispatch_matmul_gpu` → `gpu_matmul`). The residency-aware
//!      `cuda_matmul_inplace` / `cuda_batch_matmul` device-pointer
//!      branches that M4.7.3 introduced are exercised end-to-end at
//!      the kernel level by `cuda_matmul_residency_test` and
//!      `cuda_batch_matmul_residency_test`. Wiring them onto the
//!      Llama hot path requires the `in_gpu_segment` / segment-
//!      dispatcher rework scheduled for M4.7.5+. The contract this
//!      family test enforces today is therefore: M4.7.3's executor-
//!      arm changes do not regress end-to-end correctness vs the
//!      M4.7.2 / M4.6.1 baselines.
//!
//! Marked `#[ignore]`. Run with the same env-var setup as the
//! M4.7.2.e tests, plus a working CUDA driver:
//!
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "...\\model.safetensors"
//! $env:SMOLLM2_SAFETENSORS_PATH = "...\\model.safetensors"
//! $env:QWEN25_SAFETENSORS_PATH = "...\\model.safetensors"
//! $env:LLAMA32_SAFETENSORS_PATH = "...\\model.safetensors"
//! cargo test --test m4_7_3_full_family_validation_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::cuda::cuda_available;
use atenia_engine::gpu::dispatch::hooks::{gpu_matmul_resident_count, gpu_matmul_roundtrip_count};
use atenia_engine::nn::llama::{LlamaConfig, LlamaRuntime, build_llama, llama_weight_mapper};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TOKENS: [f32; 4] = [1.0, 100.0, 200.0, 300.0];
const ADR_004_THRESHOLD: f64 = 0.5;

fn load_f64_fixture(rel_dir: &str) -> Vec<f64> {
    let path = PathBuf::from("tests/fixtures")
        .join(rel_dir)
        .join("expected_logits_f64.json");
    let s = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("F64 fixture missing: {}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&s).expect("malformed F64 fixture");
    json["values"]
        .as_array()
        .expect("`values` array")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

/// One end-to-end run for a single model. Returns
/// `(max_drift, argmax_matches, gpu_matmul_delta)`.
fn run_one_model(
    label: &str,
    config_json: &str,
    safetensors_env_var: &str,
    fixture_dir: &str,
    expected_param_count: usize,
    vocab_size: usize,
) -> (f64, [bool; 4], usize) {
    println!(
        "\n=== {} F64 re-validation under M4.7.3 dispatch ===",
        label
    );

    let path =
        env::var(safetensors_env_var).unwrap_or_else(|_| panic!("Set {}", safetensors_env_var));

    let config = LlamaConfig::from_json_str(config_json).expect("parse config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // Build graph.
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), expected_param_count);

    // Load with BF16 flag on (same as M4.7.2.e — exercises the
    // decode-on-access M4.7.2.c + .d ensure_cpu defenses).
    println!(
        "Loading {} weights with store_params_as_bf16 = true ...",
        label
    );
    let load_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mut mapper =
        llama_weight_mapper(&config, &handles.param_names, &handles.param_ids).expect("mapper");
    mapper.set_store_params_as_bf16(true);
    let report = mapper.load_into(&mut graph, &reader).expect("load");
    drop(reader);
    println!(
        "Loaded {} tensors in {:.2}s",
        report.loaded,
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(report.loaded, expected_param_count);
    assert!(report.missing.is_empty());

    // Snapshot GPU MatMul counters.
    let resident_before = gpu_matmul_resident_count();
    let roundtrip_before = gpu_matmul_roundtrip_count();

    // Forward.
    let tokens = Tensor::new_cpu(vec![1, 4], TOKENS.to_vec());
    println!("Running forward (M4.7.3 dispatch active)...");
    let fwd_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward: {:.2}s", fwd_start.elapsed().as_secs_f32());

    let resident_after = gpu_matmul_resident_count();
    let roundtrip_after = gpu_matmul_roundtrip_count();
    let gpu_matmul_delta =
        (resident_after - resident_before) + (roundtrip_after - roundtrip_before);
    println!(
        "GPU MatMul invocations during forward: resident={}, roundtrip={}, total_delta={}",
        resident_after - resident_before,
        roundtrip_after - roundtrip_before,
        gpu_matmul_delta
    );

    let atenia_logits = outputs[0].as_cpu_slice();
    let total = atenia_logits.len();
    assert_eq!(total, 4 * vocab_size, "logits shape mismatch");

    // Drift vs F64.
    let f64_ref = load_f64_fixture(fixture_dir);
    assert_eq!(f64_ref.len(), total, "F64 fixture length mismatch");

    let max_drift: f64 = atenia_logits
        .iter()
        .zip(f64_ref.iter())
        .map(|(a, t)| ((*a as f64) - t).abs())
        .fold(0.0_f64, f64::max);

    // Per-position argmax check.
    let mut matches = [false; 4];
    for pos in 0..4 {
        let s = pos * vocab_size;
        let e = s + vocab_size;
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
        matches[pos] = a_id == f_id;
        let tag = if matches[pos] { "MATCH" } else { "MISMATCH" };
        println!(
            "  Pos {}: M4.7.3 argmax id={:>6}  F64 id={:>6}   [{}]",
            pos, a_id, f_id, tag
        );
    }

    println!(
        "{}: max drift vs F64 = {:.6} (threshold {:.1})",
        label, max_drift, ADR_004_THRESHOLD
    );

    (max_drift, matches, gpu_matmul_delta)
}

fn skip_if_no_cuda() -> bool {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return true;
    }
    false
}

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

const SMOLLM2_CONFIG: &str = r#"{
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

const QWEN25_CONFIG: &str = r#"{
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

const LLAMA32_CONFIG: &str = r#"{
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
#[ignore = "requires the four checkpoint env vars + F64 fixtures + CUDA driver"]
fn tinyllama_m4_7_3_matches_f64() {
    if skip_if_no_cuda() {
        return;
    }
    let (drift, matches, gpu_delta) = run_one_model(
        "TinyLlama 1.1B",
        TINYLLAMA_CONFIG,
        "TINYLLAMA_SAFETENSORS_PATH",
        "tinyllama_reference",
        201,
        32_000,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "TinyLlama M4.7.3 drift {:.6} exceeds {} (ADR-004)",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "TinyLlama M4.7.3 argmax mismatch");
    // Counter is observability-only: see module docstring. At default
    // APX mode the Llama hot path routes through the APX 4.3 GPU
    // segment dispatcher, not `try_gpu_matmul`, so this stays 0 today.
    let _ = gpu_delta;
}

#[test]
#[ignore = "requires the four checkpoint env vars + F64 fixtures + CUDA driver"]
fn smollm2_m4_7_3_matches_f64() {
    if skip_if_no_cuda() {
        return;
    }
    let (drift, matches, gpu_delta) = run_one_model(
        "SmolLM2 1.7B",
        SMOLLM2_CONFIG,
        "SMOLLM2_SAFETENSORS_PATH",
        "smollm2_reference",
        218,
        49_152,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "SmolLM2 M4.7.3 drift {:.6} exceeds {} (ADR-004)",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "SmolLM2 M4.7.3 argmax mismatch");
    let _ = gpu_delta; // counter observability-only — see TinyLlama test for rationale
}

#[test]
#[ignore = "requires the four checkpoint env vars + F64 fixtures + CUDA driver"]
fn qwen25_m4_7_3_matches_f64() {
    if skip_if_no_cuda() {
        return;
    }
    let (drift, matches, gpu_delta) = run_one_model(
        "Qwen 2.5 1.5B",
        QWEN25_CONFIG,
        "QWEN25_SAFETENSORS_PATH",
        "qwen25_reference",
        338,
        151_936,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "Qwen 2.5 M4.7.3 drift {:.6} exceeds {} (ADR-004)",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "Qwen 2.5 M4.7.3 argmax mismatch");
    let _ = gpu_delta; // counter observability-only — see TinyLlama test for rationale
}

#[test]
#[ignore = "requires the four checkpoint env vars + F64 fixtures + CUDA driver"]
fn llama_3_2_m4_7_3_matches_f64() {
    if skip_if_no_cuda() {
        return;
    }
    let (drift, matches, gpu_delta) = run_one_model(
        "Llama 3.2 1B",
        LLAMA32_CONFIG,
        "LLAMA32_SAFETENSORS_PATH",
        "llama32_reference",
        146,
        128_256,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "Llama 3.2 M4.7.3 drift {:.6} exceeds {} (ADR-004)",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "Llama 3.2 M4.7.3 argmax mismatch");
    let _ = gpu_delta; // counter observability-only — see TinyLlama test for rationale
}
