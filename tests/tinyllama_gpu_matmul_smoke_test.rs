//! M4.7.3.e — TinyLlama 1.1B end-to-end smoke test under the
//! M4.7.3 residency-aware MatMul dispatch.
//!
//! Goal: prove that with all M4.7.3 changes landed
//! (`cuda_matmul_inplace`, residency-aware `try_gpu_matmul`, the
//! per-storage gating in MatMul/BatchMatMul executor arms, the
//! `ensure_cpu` audit) the TinyLlama hot path still:
//!
//!   1. Produces logits of the expected shape and finite values,
//!      with magnitudes plausible for a 32k vocabulary.
//!   2. Argmax of every position matches the F64 ground truth from
//!      the M4.6.1 fixture — same constraint as the M4.7.2.d BF16
//!      smoke test, so any drift introduced by .a–.d is caught
//!      immediately.
//!   3. Reports the `try_gpu_matmul` counters for observability.
//!      These are NOT asserted to be > 0 because the default APX
//!      mode (4.19) runs the executor with `record_tape = true`,
//!      which routes MatMul nodes through the legacy APX 4.3 GPU
//!      segment dispatcher (`dispatch_matmul_gpu` → `gpu_matmul`)
//!      instead of `try_gpu_matmul`. The residency-aware
//!      `try_gpu_matmul` path is exercised end-to-end by the
//!      `cuda_matmul_residency_test` / `cuda_batch_matmul_residency_test`
//!      kernel-level tests; wiring it into the Llama hot path
//!      requires the `in_gpu_segment` gate relaxation that is
//!      scheduled for M4.7.5+. This smoke test's contract is
//!      therefore: M4.7.3 must not regress end-to-end correctness,
//!      and the counters are surfaced so a future routing change
//!      can spot when they start firing.
//!
//! Marked `#[ignore]` because it requires the TinyLlama safetensors
//! file, the F64 fixture, AND a working CUDA driver. Run with:
//!
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "...\\model.safetensors"
//! cargo test --test tinyllama_gpu_matmul_smoke_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::cuda::cuda_available;
use atenia_engine::gpu::dispatch::hooks::{gpu_matmul_resident_count, gpu_matmul_roundtrip_count};
use atenia_engine::nn::llama::{LlamaConfig, LlamaRuntime, build_llama, llama_weight_mapper};
use atenia_engine::tensor::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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

fn fixture_f64_reference() -> Vec<f64> {
    let path = PathBuf::from("tests/fixtures/tinyllama_reference/expected_logits_f64.json");
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

#[test]
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH + F64 fixture + CUDA driver"]
fn tinyllama_gpu_matmul_forward_logits_finite_and_argmax_matches_f64() {
    println!("\n=== TinyLlama 1.1B GPU MatMul Smoke Test (M4.7.3.e) ===\n");

    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH to TinyLlama model.safetensors");

    let config = LlamaConfig::from_json_str(EMBEDDED_TINYLLAMA_CONFIG).expect("parse config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // ---- 1. Build graph ----
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), 201);

    // ---- 2. Load weights (BF16 storage to exercise the
    //         M4.7.2.c decode-on-access through .d ensure_cpu) ----
    println!("Loading weights with store_params_as_bf16 = true ...");
    let load_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mut mapper =
        llama_weight_mapper(&config, &handles.param_names, &handles.param_ids).expect("mapper");
    let mapper_mut: &mut WeightMapper = &mut mapper;
    mapper_mut.set_store_params_as_bf16(true);
    let report = mapper.load_into(&mut graph, &reader).expect("load");
    drop(reader);
    println!(
        "Loaded {} tensors in {:.2}s",
        report.loaded,
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(report.loaded, 201);
    assert!(report.missing.is_empty());

    // ---- 3. Snapshot GPU MatMul counters before forward ----
    let resident_before = gpu_matmul_resident_count();
    let roundtrip_before = gpu_matmul_roundtrip_count();
    println!(
        "GPU MatMul counters (before forward): resident={}, roundtrip={}",
        resident_before, roundtrip_before
    );

    // ---- 4. Forward pass ----
    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    println!("Running forward (M4.7.3 dispatch active)...");
    let fwd_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    let fwd_secs = fwd_start.elapsed().as_secs_f32();
    println!("Forward: {:.2}s", fwd_secs);

    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, 4, 32_000]);
    let slice = logits.as_cpu_slice();

    // ---- 5. Logit sanity ----
    let finite = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, slice.len(), "all logits must be finite");
    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
    println!(
        "Logit stats: max |v|={:.4}  mean |v|={:.4}",
        max_abs, mean_abs
    );
    assert!(max_abs < 1000.0, "logits suspiciously large: {}", max_abs);

    // ---- 6. GPU MatMul counter observability ----
    // NOT asserted > 0: see module docstring. At default APX mode
    // 4.19 with `record_tape = true`, MatMul routes through the
    // legacy APX 4.3 GPU segment dispatcher, not `try_gpu_matmul`,
    // so these counters stay at 0 even when the GPU is actively
    // executing the kernels. Surfacing them here gives a free
    // canary for the day the routing changes.
    let resident_after = gpu_matmul_resident_count();
    let roundtrip_after = gpu_matmul_roundtrip_count();
    let resident_delta = resident_after - resident_before;
    let roundtrip_delta = roundtrip_after - roundtrip_before;
    println!(
        "GPU MatMul counters (after forward):  resident={}, roundtrip={}  (Δ resident={}, Δ roundtrip={})",
        resident_after, roundtrip_after, resident_delta, roundtrip_delta
    );
    let total_gpu_matmul = resident_delta + roundtrip_delta;

    // ---- 7. Argmax sanity vs F64 ground truth (M4.6.1 fixture) ----
    let f64_ref = fixture_f64_reference();
    assert_eq!(f64_ref.len(), slice.len(), "F64 fixture length mismatch");

    let vocab = config.vocab_size;
    let mut argmax_match_count = 0usize;
    for pos in 0..runtime.seq {
        let s = pos * vocab;
        let e = s + vocab;
        let a_pos = &slice[s..e];
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
        let tag = if a_id == f_id {
            argmax_match_count += 1;
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!(
            "  Pos {}: M4.7.3 argmax id={:>5}  F64 id={:>5}   [{}]",
            pos, a_id, f_id, tag
        );
    }

    assert_eq!(
        argmax_match_count, runtime.seq,
        "argmax mismatch under M4.7.3 dispatch on at least one position; \
         this is a real regression from the M4.7.2 / M4.6.1 baseline"
    );

    println!(
        "\nPASSED: TinyLlama M4.7.3 forward green, GPU MatMul invocations={}, argmax 4/4 vs F64.",
        total_gpu_matmul
    );
}
