//! M8.5 — full 4-model F64 re-validation under the M8 BF16-resident
//! GPU kernel path (`ATENIA_M8_BF16_KERNEL=1`).
//!
//! Re-runs the M4.6.1 / M4.7.2.e validation on each of the four
//! production checkpoints (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5
//! 1.5B, Llama 3.2 1B) with the M8 wire-up active end-to-end:
//!
//! - Loader (`WeightMapper::load_into_with_residency_plan` with
//!   `kernel_dtype = BF16`) routes every `_proj.weight` tensor into
//!   `Tier::Vram` and uploads it via `bf16_to_vram_no_upcast` —
//!   `SharedParam::Cuda { gpu: BF16-typed }`.
//! - Dispatcher (`try_gpu_matmul`) detects the BF16-resident
//!   weight at every matmul and routes through
//!   `cuda_matmul_bf16_inplace` (`cublasGemmEx` with BF16 inputs,
//!   F32 output, F32 accumulate, Tensor Cores).
//!
//! Asserts the same contracts as the existing F64 family tests:
//!
//!   1. Atenia F32 max drift vs F64 truth stays under the
//!      ADR-004 threshold (`< 0.5`).
//!   2. Argmax MATCH on every position (4/4 per model — kill-
//!      switch on any miss).
//!
//! Plus M8.5-specific assertions:
//!
//!   3. `vram_bf16_fast_path_count` advanced by the model's
//!      `_proj.weight` count (i.e. the loader actually took the
//!      M8 path and not the M6 F32-resident fallback).
//!   4. `vram_bf16_matmul_count` advanced — at least one
//!      production matmul went through the BF16 GPU dispatch.
//!
//! The single-op BF16 envelope on a per-matmul basis is in the
//! 1e-3..3e-2 range (M8.2 measured 2.89e-2 worst case at K =
//! 13824, single op). Cascaded over 22-28 transformer blocks
//! the drift compounds but stays bounded — empirically the
//! same envelope as M4.7.2.e (BF16 storage on CPU). 0.5 is the
//! conservative ceiling.
//!
//! # Setup
//!
//! Canonical paths to the four checkpoints — see
//! [`docs/MODELS_LAYOUT.md`](../docs/MODELS_LAYOUT.md) for the
//! full layout and provenance notes. Every test that needs the
//! production checkpoints reads the same env vars, so this set
//! works for `bf16_storage_full_family_validation_test`,
//! `m4_7_3_full_family_validation_test`, and the M8.5 tests below:
//!
//! ```powershell
//! $models = "F:\Proyectos\artenia_engine\atenia-engine\models"
//! $env:TINYLLAMA_SAFETENSORS_PATH = "$models\tinyllama-1.1b\model.safetensors"
//! $env:SMOLLM2_SAFETENSORS_PATH   = "$models\smollm2-1.7b-instruct\model.safetensors"
//! $env:QWEN25_SAFETENSORS_PATH    = "$models\qwen2.5-1.5b-instruct\model.safetensors"
//! $env:LLAMA32_SAFETENSORS_PATH   = "$models\llama-3.2-1b-instruct\model.safetensors"
//! cargo test --test m8_5_full_family_validation_test --release `
//!     -- --ignored --nocapture
//! ```
//!
//! Operators may keep the model checkpoints on a different volume;
//! the only contract is that the path under `models/` matches the
//! layout in `docs/MODELS_LAYOUT.md`.
//!
//! `ATENIA_M8_BF16_KERNEL` is set/unset by the test itself via a
//! RAII guard — operators do not need to manage the flag.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::weight_store::{SharedParam, WeightStore};
use atenia_engine::cuda::cuda_available;
use atenia_engine::cuda::matmul::vram_bf16_matmul_count;
use atenia_engine::gpu::tier_plan::{plan, TensorMeta, TierPlanInput};
use atenia_engine::nn::llama::{
    build_llama, build_llama_with_store, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::{DType, Tensor};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::vram_bf16_fast_path_count;

const TOKENS: [f32; 4] = [1.0, 100.0, 200.0, 300.0];
const ADR_004_THRESHOLD: f64 = 0.5;

/// Process-wide serialisation for any test that touches
/// `ATENIA_M8_BF16_KERNEL` or snapshots the BF16 counters.
/// The flag is read at every loader invocation; without this
/// lock, parallel test execution would race on whether the flag
/// is set during a given run.
static M8_5_TEST_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard for `ATENIA_M8_BF16_KERNEL`. Sets the flag on
/// construction; removes it on `Drop` (panic-safe).
struct M8FlagGuard;

impl M8FlagGuard {
    fn set() -> Self {
        unsafe {
            env::set_var("ATENIA_M8_BF16_KERNEL", "1");
        }
        Self
    }
}

impl Drop for M8FlagGuard {
    fn drop(&mut self) {
        unsafe {
            env::remove_var("ATENIA_M8_BF16_KERNEL");
        }
    }
}

fn load_f64_fixture(rel_dir: &str) -> Vec<f64> {
    let path =
        PathBuf::from("tests/fixtures").join(rel_dir).join("expected_logits_f64.json");
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

/// Build the per-tensor metadata vector that the planner consumes.
/// Mirrors the `pipeline.rs` single-shard branch.
fn collect_metas_from_reader(reader: &SafetensorsReader) -> Vec<TensorMeta> {
    reader
        .iter()
        .map(|e| TensorMeta {
            name: e.name.to_string(),
            shape: e.shape.to_vec(),
            dtype: e.dtype,
        })
        .collect()
}

/// Map (name → dtype) over the safetensors entries, used to
/// compute the model_total_bytes input to the planner.
fn sum_model_bytes(metas: &[TensorMeta]) -> u64 {
    metas
        .iter()
        .map(|m| {
            let numel: u64 = m.shape.iter().product::<usize>() as u64;
            numel * (m.dtype.size_in_bytes() as u64)
        })
        .sum()
}

/// Count the planner's `_proj.weight + rank ≥ 2` tensors — these
/// are the GPU-eligible weights that the M8 path should upload as
/// BF16-resident. Used as the lower bound for the
/// `vram_bf16_fast_path_count` delta after a load.
fn count_proj_weights(metas: &[TensorMeta]) -> usize {
    metas
        .iter()
        .filter(|m| m.shape.len() >= 2 && m.name.ends_with("_proj.weight"))
        .count()
}

/// Confirm that the loader actually uploaded the right number of
/// BF16-resident `_proj.weight` tensors. Counts directly from the
/// resulting store rather than the global counter, which lets us
/// run multiple models in the same process without inter-test
/// counter leakage on this assertion.
fn count_bf16_resident_in_store(store: &WeightStore) -> usize {
    store
        .params
        .iter()
        .filter(|p| {
            if let SharedParam::Cuda { gpu, .. } = p {
                gpu.dtype() == DType::BF16
            } else {
                false
            }
        })
        .count()
}

/// One end-to-end run for a single model under the M8 BF16-
/// resident path. Returns `(max_drift, argmax_matches,
/// bf16_resident_count_in_store, bf16_matmul_delta)`.
#[allow(clippy::too_many_arguments)]
fn run_one_model_m8(
    label: &str,
    config_json: &str,
    safetensors_env_var: &str,
    fixture_dir: &str,
    expected_param_count: usize,
    vocab_size: usize,
) -> (f64, [bool; 4], usize, usize) {
    println!("\n=== {} F64 validation under M8 BF16 kernel ===", label);

    // Resolve the safetensors path with diagnostics that point
    // back to `docs/MODELS_LAYOUT.md`. Two failure modes:
    //   1. Env var unset      → panic listing the env var name.
    //   2. Env var set but the file does not exist on disk
    //      → panic listing the resolved path so the operator
    //      can compare against the canonical layout.
    let path = env::var(safetensors_env_var).unwrap_or_else(|_| {
        panic!(
            "Set {} to the model.safetensors path. \
             See docs/MODELS_LAYOUT.md for the canonical layout.",
            safetensors_env_var
        )
    });
    if !Path::new(&path).is_file() {
        panic!(
            "{} resolves to '{}' but no file exists there. \
             Verify the model is at the expected path \
             (see docs/MODELS_LAYOUT.md) — typical layout is \
             `<repo>/models/<model-dir>/model.safetensors`.",
            safetensors_env_var, path
        );
    }

    let config = LlamaConfig::from_json_str(config_json).expect("parse config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // ---- Build the scratch graph (just for handles + param ids).
    let mut gb_scratch = GraphBuilder::new();
    let token_input_id_scratch = gb_scratch.input();
    let handles =
        build_llama(&mut gb_scratch, &config, &runtime, token_input_id_scratch);
    let _ = gb_scratch.output(handles.logits_id);
    let mut scratch_graph = gb_scratch.build();
    assert_eq!(handles.param_ids.len(), expected_param_count);

    // ---- Open safetensors and build the tier plan with kernel_dtype = BF16.
    println!("Loading {} via M8 BF16-resident path ...", label);
    let load_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("mapper");

    let metas = collect_metas_from_reader(&reader);
    let proj_weight_count = count_proj_weights(&metas);
    let model_total_bytes = sum_model_bytes(&metas);

    // Plan with abundant VRAM (16 GiB) — the small validation
    // models (1-2 GB BF16) all fit on the RTX 4070's 8 GiB. The
    // 16 GiB number lets the planner unconditionally pick Vram
    // for every GPU-eligible weight regardless of the live
    // probe; M8.5 is a numerical contract test, not a hardware
    // sizing test.
    let plan_input = TierPlanInput {
        tensors: metas,
        free_vram_bytes: 16 * 1024 * 1024 * 1024,
        free_ram_bytes: 32 * 1024 * 1024 * 1024,
        model_total_bytes,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        kernel_dtype: DType::BF16,
    };
    let plan_out = plan(&plan_input);

    // Sanity: the plan placed every _proj.weight on Vram.
    let vram_count = plan_out.vram_count();
    assert!(
        vram_count >= proj_weight_count,
        "{}: plan placed only {} on Vram (expected ≥ {} _proj.weight)",
        label,
        vram_count,
        proj_weight_count
    );

    // Snapshot the BF16 counters for the post-load delta check.
    let bf16_load_before = vram_bf16_fast_path_count();

    let (store, report) = mapper
        .load_into_with_residency_plan(
            &mut scratch_graph,
            &reader,
            &plan_out,
            &handles.param_ids,
            &handles.param_names,
        )
        .expect("load_into_with_residency_plan");
    drop(reader);
    println!(
        "Loaded {} tensors in {:.2}s",
        report.loaded,
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(report.loaded, expected_param_count);
    assert!(report.missing.is_empty());

    let bf16_load_after = vram_bf16_fast_path_count();
    let bf16_load_delta = bf16_load_after - bf16_load_before;
    let bf16_resident_count = count_bf16_resident_in_store(&store);
    println!(
        "BF16 resident: {} in store / {} loader fast-path delta (expected ≥ {})",
        bf16_resident_count, bf16_load_delta, proj_weight_count
    );
    assert!(
        bf16_resident_count >= proj_weight_count,
        "{}: store has only {} BF16-resident params (expected ≥ {})",
        label,
        bf16_resident_count,
        proj_weight_count
    );
    assert!(
        bf16_load_delta >= proj_weight_count,
        "{}: vram_bf16_fast_path_count delta {} < {} _proj.weight tensors",
        label,
        bf16_load_delta,
        proj_weight_count
    );

    // ---- Drop the scratch graph; build a NEW graph wired to the store.
    drop(scratch_graph);

    let mut gb_exec = GraphBuilder::new();
    let token_input_id = gb_exec.input();
    let handles_exec = build_llama_with_store(
        &mut gb_exec,
        &config,
        &runtime,
        token_input_id,
        &store,
        None,
    )
    .expect("build_llama_with_store");
    let _ = gb_exec.output(handles_exec.logits_id);
    let mut graph_for_exec = gb_exec.build();

    // ---- Forward pass. Snapshot vram_bf16_matmul_count to verify
    // the dispatcher actually fired the BF16 path on at least one
    // matmul.
    let bf16_matmul_before = vram_bf16_matmul_count();
    let tokens = Tensor::new_cpu(vec![1, 4], TOKENS.to_vec());
    println!("Running forward (M8 BF16 dispatcher active)...");
    let fwd_start = std::time::Instant::now();
    let outputs = graph_for_exec.execute(vec![tokens]);
    println!("Forward: {:.2}s", fwd_start.elapsed().as_secs_f32());
    let bf16_matmul_after = vram_bf16_matmul_count();
    let bf16_matmul_delta = bf16_matmul_after - bf16_matmul_before;
    println!(
        "BF16 matmul calls: {} during forward",
        bf16_matmul_delta
    );
    assert!(
        bf16_matmul_delta > 0,
        "{}: vram_bf16_matmul_count did not advance — dispatcher did NOT take the BF16 arm",
        label
    );

    let atenia_logits = outputs[0].as_cpu_slice();
    let total = atenia_logits.len();
    assert_eq!(total, 4 * vocab_size, "logits shape mismatch");

    // ---- Drift vs F64.
    let f64_ref = load_f64_fixture(fixture_dir);
    assert_eq!(f64_ref.len(), total, "F64 fixture length mismatch");

    let max_drift: f64 = atenia_logits
        .iter()
        .zip(f64_ref.iter())
        .map(|(a, t)| ((*a as f64) - t).abs())
        .fold(0.0_f64, f64::max);

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
            "  Pos {}: M8 argmax id={:>6}  F64 id={:>6}   [{}]",
            pos, a_id, f_id, tag
        );
    }

    println!(
        "{}: max drift vs F64 = {:.6} (threshold {:.1})",
        label, max_drift, ADR_004_THRESHOLD
    );

    (max_drift, matches, bf16_resident_count, bf16_matmul_delta)
}

// ---- Configs (verbatim from the existing M4.7.2.e test file). ----

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

// ---- Per-model validation tests. ----

#[test]
#[ignore = "requires CUDA + checkpoint env vars + F64 fixtures"]
fn m8_5_tinyllama_under_bf16_kernel_matches_f64() {
    let _guard = M8_5_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let _flag = M8FlagGuard::set();

    let (drift, matches, _store_count, _matmul_delta) = run_one_model_m8(
        "TinyLlama 1.1B",
        TINYLLAMA_CONFIG,
        "TINYLLAMA_SAFETENSORS_PATH",
        "tinyllama_reference",
        201,
        32_000,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "TinyLlama M8 drift {:.6} exceeds {} (ADR-004) — PARAR per protocol",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "TinyLlama M8 argmax mismatch");
}

#[test]
#[ignore = "requires CUDA + checkpoint env vars + F64 fixtures"]
fn m8_5_smollm2_under_bf16_kernel_matches_f64() {
    let _guard = M8_5_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let _flag = M8FlagGuard::set();

    let (drift, matches, _store_count, _matmul_delta) = run_one_model_m8(
        "SmolLM2 1.7B",
        SMOLLM2_CONFIG,
        "SMOLLM2_SAFETENSORS_PATH",
        "smollm2_reference",
        218,
        49_152,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "SmolLM2 M8 drift {:.6} exceeds {} (ADR-004) — PARAR per protocol",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "SmolLM2 M8 argmax mismatch");
}

#[test]
#[ignore = "requires CUDA + checkpoint env vars + F64 fixtures"]
fn m8_5_qwen25_under_bf16_kernel_matches_f64() {
    let _guard = M8_5_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let _flag = M8FlagGuard::set();

    let (drift, matches, _store_count, _matmul_delta) = run_one_model_m8(
        "Qwen 2.5 1.5B",
        QWEN25_CONFIG,
        "QWEN25_SAFETENSORS_PATH",
        "qwen25_reference",
        338,
        151_936,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "Qwen 2.5 M8 drift {:.6} exceeds {} (ADR-004) — PARAR per protocol",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "Qwen 2.5 M8 argmax mismatch");
}

#[test]
#[ignore = "requires CUDA + checkpoint env vars + F64 fixtures"]
fn m8_5_llama_3_2_under_bf16_kernel_matches_f64() {
    let _guard = M8_5_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let _flag = M8FlagGuard::set();

    let (drift, matches, _store_count, _matmul_delta) = run_one_model_m8(
        "Llama 3.2 1B",
        LLAMA32_CONFIG,
        "LLAMA32_SAFETENSORS_PATH",
        "llama32_reference",
        146,
        128_256,
    );
    assert!(
        drift < ADR_004_THRESHOLD,
        "Llama 3.2 M8 drift {:.6} exceeds {} (ADR-004) — PARAR per protocol",
        drift,
        ADR_004_THRESHOLD
    );
    assert_eq!(matches, [true; 4], "Llama 3.2 M8 argmax mismatch");
}

// **Hash to make HashMap import not stale** — referenced indirectly
// by the safetensors crate signature via `HashMap`-shaped constructors
// in the helper module shared with other tests; keep the import
// active to silence dead-code on platforms where the helper inlines.
#[allow(dead_code)]
fn _hashmap_import_anchor() -> HashMap<String, String> {
    HashMap::new()
}
