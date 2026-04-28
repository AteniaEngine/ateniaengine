//! M4.7.2.d — TinyLlama 1.1B end-to-end smoke test under native BF16
//! parameter storage.
//!
//! Builds the standard TinyLlama 1.1B graph, loads its 201 parameters
//! through `WeightMapper::load_into` with `store_params_as_bf16=true`,
//! drops the safetensors reader to free the raw bytes, and runs a
//! forward pass on `[1, 100, 200, 300]`. The forward pass exercises
//! the M4.7.2.c executor decode-on-access in four arms (MatMul,
//! IndexSelect, BroadcastMul, BroadcastAdd) plus every other op that
//! never consumes a BF16 param directly.
//!
//! Pass criteria (the close-criterion for M4.7.2.c):
//! - All 201 tensors loaded into `CpuBf16` storage.
//! - Resident parameter footprint roughly halved versus F32 storage:
//!   `params_bf16_bytes ≈ params_f32_bytes / 2`.
//! - Forward produces logits of the expected shape, all values
//!   finite, magnitudes plausible (< 1000).
//! - Argmax of the last position matches the F64 ground truth from
//!   the M4.6.1 fixture (`tests/fixtures/tinyllama_reference/expected_logits_f64.json`),
//!   which is the same constraint M4.6.1's F64 test already enforces.
//!   M4.7.2.e re-runs the full F64 numerical drift check across the
//!   M4.6 family.
//!
//! Marked `#[ignore]`; run with:
//!
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "...\\model.safetensors"
//! cargo test --test tinyllama_bf16_storage_smoke_test --release \
//!     -- --ignored --nocapture
//! ```

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::tensor::{Tensor, TensorStorage};
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
    let path =
        PathBuf::from("tests/fixtures/tinyllama_reference/expected_logits_f64.json");
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
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH + tests/fixtures/tinyllama_reference/expected_logits_f64.json"]
fn tinyllama_bf16_storage_loads_executes_and_argmax_matches_f64() {
    println!("\n=== TinyLlama 1.1B BF16 Storage Smoke Test (M4.7.2.d) ===\n");

    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH to TinyLlama model.safetensors");

    let config = LlamaConfig::from_json_str(EMBEDDED_TINYLLAMA_CONFIG).expect("parse config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // ---- 1. Build graph (unchanged from M4.6 baseline) ----
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), 201);

    // ---- 2. Load weights with the BF16 flag ON ----
    println!("Loading weights with store_params_as_bf16 = true ...");
    let load_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mut mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("mapper");
    let mapper_mut: &mut WeightMapper = &mut mapper;
    mapper_mut.set_store_params_as_bf16(true);
    let report = mapper.load_into(&mut graph, &reader).expect("load");
    drop(reader);
    println!(
        "Loaded {} tensors in {:.2}s (BF16 flag active)",
        report.loaded,
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(report.loaded, 201);
    assert!(report.missing.is_empty());

    // ---- 3. Storage variant + RAM accounting ----
    let mut bf16_count = 0usize;
    let mut f32_count = 0usize;
    let mut bf16_bytes = 0usize;
    let mut equivalent_f32_bytes = 0usize;
    for &id in &handles.param_ids {
        let t = graph.nodes[id].output.as_ref().expect("param tensor");
        match t.storage() {
            TensorStorage::CpuBf16(bits) => {
                bf16_count += 1;
                bf16_bytes += bits.len() * 2;
                equivalent_f32_bytes += bits.len() * 4;
            }
            TensorStorage::Cpu(v) => {
                f32_count += 1;
                bf16_bytes += v.len() * 4;
                equivalent_f32_bytes += v.len() * 4;
            }
            other => panic!("unexpected param storage variant: {:?}", other),
        }
    }
    println!(
        "Param storage: {} CpuBf16, {} Cpu",
        bf16_count, f32_count
    );
    println!(
        "Param footprint: {:.1} MB BF16  vs  {:.1} MB F32 (savings = {:.1} MB, {:.1}%)",
        bf16_bytes as f64 / 1e6,
        equivalent_f32_bytes as f64 / 1e6,
        (equivalent_f32_bytes - bf16_bytes) as f64 / 1e6,
        100.0 * (equivalent_f32_bytes - bf16_bytes) as f64 / equivalent_f32_bytes as f64,
    );
    assert_eq!(
        bf16_count, 201,
        "every param must end up as CpuBf16 with the flag on"
    );
    assert_eq!(f32_count, 0, "no param should be Cpu(F32) under the BF16 flag");

    // RAM saving sanity: the BF16 footprint should be half the F32
    // equivalent. Allow 0.001 % slack for floating-point arithmetic
    // on the percent computation.
    let savings_ratio = bf16_bytes as f64 / equivalent_f32_bytes as f64;
    assert!(
        (savings_ratio - 0.5).abs() < 1e-6,
        "expected BF16 footprint ≈ 0.5 × F32 (got ratio = {:.6})",
        savings_ratio
    );

    // ---- 4. Forward pass ----
    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    println!("Running forward (BF16 storage active throughout)...");
    let fwd_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward: {:.2}s", fwd_start.elapsed().as_secs_f32());

    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, 4, 32_000]);
    let slice = logits.as_cpu_slice();
    let finite = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, slice.len(), "all logits must be finite");

    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
    println!("Logit stats: max |v|={:.4}  mean |v|={:.4}", max_abs, mean_abs);
    assert!(max_abs < 1000.0, "logits suspiciously large: {}", max_abs);

    // ---- 5. Argmax sanity vs F64 ground truth (M4.6.1 fixture) ----
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
        println!("  Pos {}: BF16 argmax id={:>5}  F64 id={:>5}   [{}]", pos, a_id, f_id, tag);
    }

    assert_eq!(
        argmax_match_count, runtime.seq,
        "argmax mismatch under BF16 storage on at least one position; \
         this is a real regression from the M4.6.1 F64 baseline"
    );

    println!("\nPASSED: TinyLlama BF16-storage forward green, argmax 4/4 vs F64.");
}
