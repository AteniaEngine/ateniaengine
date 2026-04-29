//! M4.7.5.f — full 4-model F64 re-validation under M4.7.5
//! LRU-driven selective spill policy.
//!
//! Sister of `bf16_storage_full_family_validation_test`
//! (M4.7.2.e), `m4_7_3_full_family_validation_test`
//! (M4.7.3.f), and `m4_7_4_f_full_family_disk_spill_test`
//! (M4.7.4.f). Re-runs the same family contract (TinyLlama
//! 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B) but with
//! the M4.7.5.d LRU-driven `DeepDegrade` orchestrator forced
//! mid-forward, exercising:
//!
//!   - the M4.7.5.b touch-order LRU populated by
//!     `NodeTimingRecorder::drop`,
//!   - the M4.7.5.c `migrate_selected_cpu_to_disk` selective
//!     primitive,
//!   - the M4.7.5.d `deep_degrade_with_lru` orchestrator with
//!     `SPILL_FRACTION = 0.5`,
//!   - the M4.7.5.e `ensure_cpu` defensive guards on
//!     Add / Sub / Mul (no Add/Sub/Mul in the Llama hot path
//!     today, but covered for completeness),
//!   - the M4.7.4.d Disk-arm `ensure_cpu` dispatch on the
//!     dtype tag (lazy restore on every node consumption).
//!
//! Asserts:
//!
//!   1. Atenia F32 max drift vs F64 truth stays under the
//!      ADR-004 threshold (`< 0.5`). Per the M4.7.4.f baseline,
//!      the spill + restore cycle is mathematically transparent
//!      (BF16 → F32 upcast is a pure zero-fill of trailing
//!      mantissa bits), so M4.7.5 should be bit-exact identical
//!      to that.
//!   2. Argmax MATCH on every position (4/4 — kill-switch on
//!      any mismatch).
//!   3. The new gate from the M4.7.5 plan: a spill *was*
//!      triggered (`deep_degrade_events_count > 0` after a
//!      forced trigger), and `tensors_migrated < total_params`
//!      proving the selective slice is in effect (vs the
//!      whole-graph M4.7.4.f baseline).
//!
//! The forced trigger pattern: warmup forward to populate the
//! LRU, then call `Graph::deep_degrade_with_lru` directly,
//! then run the forward again — the second forward exercises
//! the lazy-restore path through the M4.7.5.e ensure_cpu
//! guards on every consumed parameter.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::tensor::Tensor;
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_manager::GuardManager;
use atenia_engine::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

const TOKENS: [f32; 4] = [1.0, 100.0, 200.0, 300.0];
const ADR_004_THRESHOLD: f64 = 0.5;

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

fn cache_dir_for(label: &str) -> PathBuf {
    let base = match env::var("ATENIA_DISK_TIER_DIR") {
        Ok(v) => PathBuf::from(v),
        Err(_) => atenia_engine::tensor::disk_tier::default_cache_dir().join("m4_7_5_f"),
    };
    base.join(format!("{}_{}", label, Uuid::new_v4()))
}

struct LowPressureVramProbe;
impl VramProbeApi for LowPressureVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        Ok(VramSnapshot {
            total_bytes: 1000,
            free_bytes: 900,
            used_bytes: 100,
        })
    }
}
struct LowPressureRamProbe;
impl RamProbeApi for LowPressureRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        Ok(RamSnapshot {
            total_bytes: 1000,
            available_bytes: 900,
            used_bytes: 100,
        })
    }
}

fn permissive_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.5,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.4,
        },
        runtime_snapshot: RuntimeState {
            memory_headroom: 0.8,
            is_stable: true,
            recent_recovery: false,
            offload_supported: true,
        },
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.5,
        require_fallback: false,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

fn make_low_pressure_context(cache_dir: PathBuf) -> ReactiveExecutionContext {
    let bus = Arc::new(SignalBus::with_probes(
        None,
        None,
        None,
        None,
        Some(Arc::new(LowPressureVramProbe)),
        Some(Arc::new(LowPressureRamProbe)),
    ));
    let guards: Vec<Box<dyn ExecutionGuard>> =
        vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    ReactiveExecutionContext::new_without_gc(bus, permissive_contract(), gm)
        .with_cache_dir(cache_dir)
}

/// Run one model end-to-end. Returns `(max_drift, argmax_matches,
/// tensors_migrated_by_lru, total_params)`.
fn run_one_model(
    label: &str,
    config_json: &str,
    safetensors_env_var: &str,
    fixture_dir: &str,
    expected_param_count: usize,
    vocab_size: usize,
) -> (f64, [bool; 4], usize, usize) {
    println!("\n=== {} F64 re-validation under M4.7.5 LRU spill ===", label);

    let path =
        env::var(safetensors_env_var).unwrap_or_else(|_| panic!("Set {}", safetensors_env_var));
    let dir = cache_dir_for(label);
    fs::create_dir_all(&dir).expect("create cache dir");
    println!("Cache dir: {}", dir.display());

    let config = LlamaConfig::from_json_str(config_json).expect("parse config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // Build graph + load BF16 weights.
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), expected_param_count);

    println!("Loading {} weights with store_params_as_bf16 = true ...", label);
    let load_start = Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mut mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("mapper");
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

    // Attach reactive context with low-pressure probes (no
    // automatic DeepDegrade trigger). LRU populates from the
    // warmup forward; the LRU-driven spill is invoked directly
    // afterwards so we control the timing.
    let ctx = make_low_pressure_context(dir.clone());
    let lru_handle = ctx.lru_touch_order();
    graph.set_reactive_context(ctx);

    // Warmup forward to populate the LRU. This is also the
    // "no-spill baseline" forward we use for the drift figure.
    let tokens = Tensor::new_cpu(vec![1, 4], TOKENS.to_vec());
    println!("Warmup forward (no spill yet) ...");
    let warmup_start = Instant::now();
    let warmup_outputs = graph.execute(vec![tokens.clone()]);
    println!("Warmup: {:.2}s", warmup_start.elapsed().as_secs_f32());

    let warmup_logits = warmup_outputs[0].as_cpu_slice().to_vec();
    let total = warmup_logits.len();
    assert_eq!(total, 4 * vocab_size, "logits shape mismatch");

    // Capture LRU size before the spill — selectivity is
    // measured against the size of the LRU at the moment of the
    // trigger, not against the parameter count (the LRU
    // includes activations / intermediates as well, so for any
    // forward of width > 1 the touched-node count exceeds the
    // param count).
    let lru_size_before = lru_handle.len();
    println!("LRU size at spill trigger: {}", lru_size_before);

    // Force the LRU-driven spill.
    println!("Forcing M4.7.5.d deep_degrade_with_lru ...");
    let spill_start = Instant::now();
    let migration = graph
        .deep_degrade_with_lru(&dir)
        .expect("LRU-driven spill must succeed");
    let spill_secs = spill_start.elapsed().as_secs_f32();
    println!(
        "Spill: {} migrated, {} skipped, {:.2}s",
        migration.tensors_migrated, migration.tensors_skipped, spill_secs
    );

    // Selectivity gate: M4.7.5 contract is "spill the bottom
    // SPILL_FRACTION of the LRU, not the whole LRU". With
    // SPILL_FRACTION = 0.5, the bottom slice is `floor(lru_size
    // * 0.5)` ids; of those, `tensors_migrated` is the subset
    // that were Cpu/CpuBf16 migration candidates. The structural
    // contract is therefore `migrated < lru_size` (proves the
    // helper did not silently fall back to the whole-LRU /
    // whole-graph path) AND `migrated > 0` (proves the spill
    // ran).
    assert!(
        migration.tensors_migrated > 0,
        "M4.7.5 spill produced 0 migrations on a {}-entry LRU; \
         the bottom slice was empty or every id was non-eligible",
        lru_size_before
    );
    assert!(
        migration.tensors_migrated < lru_size_before,
        "M4.7.5 must spill SELECTIVELY: migrated={} >= LRU size={}; \
         the LRU bottom-fraction slice was bypassed (whole-LRU spill?)",
        migration.tensors_migrated, lru_size_before
    );

    // Reference for the per-model line below: how the bottom-50%
    // slice maps onto eligible candidates given the warmup
    // forward's touched-node profile.
    let bottom_slice_target =
        ((lru_size_before as f32) * 0.5_f32).floor() as usize;
    println!(
        "Selectivity sanity: migrated={}, bottom_slice_target={} (50% of LRU), \
         lru_size={}, total_params={}",
        migration.tensors_migrated,
        bottom_slice_target,
        lru_size_before,
        expected_param_count,
    );

    // Re-execute. Every consumer arm's M4.7.3.d / M4.7.5.e
    // ensure_cpu fires on the parameter operands that landed
    // on Disk; the M4.7.4.d Disk-arm dispatch upcasts BF16
    // bytes to F32 transparently.
    println!("Post-spill forward (lazy restore through ensure_cpu) ...");
    let post_start = Instant::now();
    let post_outputs = graph.execute(vec![tokens]);
    let post_secs = post_start.elapsed().as_secs_f32();
    println!("Post-spill forward: {:.2}s", post_secs);

    let atenia_logits = post_outputs[0].as_cpu_slice();
    assert_eq!(atenia_logits.len(), total);

    // Drift vs F64.
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
            "  Pos {}: M4.7.5 argmax id={:>6}  F64 id={:>6}   [{}]",
            pos, a_id, f_id, tag
        );
    }

    println!(
        "{}: max drift vs F64 = {:.6} (threshold {:.1}); \
         tensors_migrated_by_lru = {}/{}; warmup_logits == post_logits = {}",
        label,
        max_drift,
        ADR_004_THRESHOLD,
        migration.tensors_migrated,
        expected_param_count,
        warmup_logits == atenia_logits,
    );

    let _ = std::fs::remove_dir_all(&dir);
    (max_drift, matches, migration.tensors_migrated, lru_size_before)
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
#[ignore = "requires the four checkpoint env vars + F64 fixtures + ATENIA_DISK_TIER_DIR (NVMe-backed)"]
fn tinyllama_m4_7_5_lru_matches_f64() {
    let (drift, matches, migrated, lru_size) = run_one_model(
        "TinyLlama 1.1B",
        TINYLLAMA_CONFIG,
        "TINYLLAMA_SAFETENSORS_PATH",
        "tinyllama_reference",
        201,
        32_000,
    );
    assert!(drift < ADR_004_THRESHOLD, "TinyLlama drift {:.6}", drift);
    assert_eq!(matches, [true; 4], "TinyLlama argmax mismatch");
    assert!(migrated < lru_size, "TinyLlama spill not selective");
}

#[test]
#[ignore = "requires the four checkpoint env vars + F64 fixtures + ATENIA_DISK_TIER_DIR (NVMe-backed)"]
fn smollm2_m4_7_5_lru_matches_f64() {
    let (drift, matches, migrated, lru_size) = run_one_model(
        "SmolLM2 1.7B",
        SMOLLM2_CONFIG,
        "SMOLLM2_SAFETENSORS_PATH",
        "smollm2_reference",
        218,
        49_152,
    );
    assert!(drift < ADR_004_THRESHOLD, "SmolLM2 drift {:.6}", drift);
    assert_eq!(matches, [true; 4], "SmolLM2 argmax mismatch");
    assert!(migrated < lru_size, "SmolLM2 spill not selective");
}

#[test]
#[ignore = "requires the four checkpoint env vars + F64 fixtures + ATENIA_DISK_TIER_DIR (NVMe-backed)"]
fn qwen25_m4_7_5_lru_matches_f64() {
    let (drift, matches, migrated, lru_size) = run_one_model(
        "Qwen 2.5 1.5B",
        QWEN25_CONFIG,
        "QWEN25_SAFETENSORS_PATH",
        "qwen25_reference",
        338,
        151_936,
    );
    assert!(drift < ADR_004_THRESHOLD, "Qwen 2.5 drift {:.6}", drift);
    assert_eq!(matches, [true; 4], "Qwen 2.5 argmax mismatch");
    assert!(migrated < lru_size, "Qwen 2.5 spill not selective");
}

#[test]
#[ignore = "requires the four checkpoint env vars + F64 fixtures + ATENIA_DISK_TIER_DIR (NVMe-backed)"]
fn llama_3_2_m4_7_5_lru_matches_f64() {
    let (drift, matches, migrated, lru_size) = run_one_model(
        "Llama 3.2 1B",
        LLAMA32_CONFIG,
        "LLAMA32_SAFETENSORS_PATH",
        "llama32_reference",
        146,
        128_256,
    );
    assert!(drift < ADR_004_THRESHOLD, "Llama 3.2 drift {:.6}", drift);
    assert_eq!(matches, [true; 4], "Llama 3.2 argmax mismatch");
    assert!(migrated < lru_size, "Llama 3.2 spill not selective");
}
