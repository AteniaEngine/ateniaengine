//! **CERTIFY-BREADTH-1** — CPU F32 vs F64 ground-truth validation harness for
//! the families that functionally validate but lack an ADR-004 numeric
//! certificate (Gemma 2, Gemma 3, Phi-3).
//!
//! Methodology (ADR-002 / ADR-004): assert Atenia's **CPU F32** forward against
//! a **PyTorch F64** reference fixture on the canonical 4-token sequence
//! `[1, 100, 200, 300]` — `max_abs_diff < 0.5` and per-position argmax match.
//! This is the exact primary metric the M4.6 Llama-family CPU tests use
//! (`llama_3_2_numerical_validation_test.rs`); the only new thing here is that
//! the forward is driven through the **adapter layer** so it is family-agnostic
//! (Gemma 2's SoftCap / dual-norm, Phi-3's fused QKV + LongRope, Gemma 3's
//! dual-RoPE are all exercised via their registered builders).
//!
//! ## Status (why the model tests are `#[ignore]`)
//!
//! These three families have **no committed F64 fixture yet** — generating one
//! requires a PyTorch F64 pass on the real multi-GB checkpoint (see
//! `tests/fixtures/generate_f64_reference.py`) with enough RAM
//! (Gemma-3-1B ~8 GiB, Gemma-2-2B ~21 GiB, Phi-3.5 ~30 GiB peak f64). Until the
//! fixture exists, the per-model test panics with an actionable message. The
//! certificate manifests keep `max_abs_diff_vs_f64: null` — **no number is
//! fabricated**. The *infrastructure* (this harness + the generation script) is
//! what CERTIFY-BREADTH-1 delivers; filling the drift is a follow-up run on
//! adequate hardware.
//!
//! ## Note on what this certifies vs the manifest `certified_mode`
//!
//! This harness certifies the **engine's F32 forward math** (CPU path) against
//! F64 truth — the ADR-004 *primary* claim. The manifest's
//! `drift_envelope.certified_mode` field specifically refers to the GPU TF32
//! kernel path; filling **that** field additionally needs the CUDA run. The
//! two are complementary; neither is relaxed.
//!
//! ## Run (once a fixture exists)
//! ```powershell
//! $env:GEMMA2_2B_DIR = "...\models\gemma-2-2b-it"
//! cargo test --test certify_breadth_f64_validation_test --release -- --ignored --nocapture
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::model_adapters::{
    model_metadata_from_parts, resolve_adapter, AteniaModelAdapter, ModelFormat,
};
use atenia_engine::nn::llama::config::LlamaConfig;
use atenia_engine::nn::llama::LlamaRuntime;
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::sharded_reader::ShardedSafetensorsReader;

/// Canonical fixture input — identical to every existing F64 fixture.
const TOKEN_IDS: [i64; 4] = [1, 100, 200, 300];
/// ADR-004 strict gate.
const ADR_004_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// Pure helpers (exercised by the CI unit tests below — no model required)
// ---------------------------------------------------------------------------

/// Worst per-element absolute drift of Atenia F32 vs the F64 reference.
fn max_abs_diff(atenia: &[f32], reference_f64: &[f64]) -> f64 {
    atenia
        .iter()
        .zip(reference_f64.iter())
        .map(|(a, t)| ((*a as f64) - *t).abs())
        .fold(0.0_f64, f64::max)
}

/// Per-position argmax agreement between Atenia and the F64 reference. The
/// logit buffers are `[seq * vocab]` row-major over `(position, vocab)`.
fn per_position_argmax_match(
    atenia: &[f32],
    reference_f64: &[f64],
    seq: usize,
    vocab: usize,
) -> Vec<bool> {
    (0..seq)
        .map(|pos| {
            let s = pos * vocab;
            let e = s + vocab;
            let a_arg = argmax_f32(&atenia[s..e]);
            let r_arg = argmax_f64(&reference_f64[s..e]);
            a_arg == r_arg
        })
        .collect()
}

fn argmax_f32(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

fn argmax_f64(v: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Load `tests/fixtures/<rel_dir>/expected_logits_f64.json` → flat `Vec<f64>`.
/// Panics with an actionable message (naming the generation script) when the
/// fixture is absent — the explicit "not certified yet" state.
fn load_f64_fixture(rel_dir: &str) -> Vec<f64> {
    let path = PathBuf::from("tests/fixtures")
        .join(rel_dir)
        .join("expected_logits_f64.json");
    let s = fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "F64 fixture missing: {}\n  Generate it with:\n    python \
             tests/fixtures/generate_f64_reference.py <model_dir> tests/fixtures/{}",
            path.display(),
            rel_dir
        )
    });
    let json: serde_json::Value = serde_json::from_str(&s).expect("malformed F64 fixture");
    json["values"]
        .as_array()
        .expect("`values` array")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

/// Read `architectures[0]` from a checkpoint `config.json`.
fn read_architecture0(config_path: &Path) -> Option<String> {
    let bytes = fs::read(config_path).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    v.get("architectures")?
        .as_array()?
        .first()?
        .as_str()
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Family-agnostic CPU F32 forward (drives the registered adapter)
// ---------------------------------------------------------------------------

/// Run Atenia's **CPU F32** forward on `TOKEN_IDS` for the checkpoint at
/// `model_dir`, through the registered family adapter. Returns
/// `(flat_logits[seq*vocab], seq, vocab)`. Handles single-file and sharded
/// safetensors. No CUDA — pure CPU graph execution.
fn atenia_cpu_logits(model_dir: &Path) -> (Vec<f32>, usize, usize) {
    let config_path = model_dir.join("config.json");
    let config = LlamaConfig::from_json_file(&config_path).expect("parse config.json");

    let arch = read_architecture0(&config_path);
    let metadata = model_metadata_from_parts(
        arch.as_deref(),
        config.model_type.as_deref(),
        ModelFormat::HfSafetensors,
    );
    let adapter: &dyn AteniaModelAdapter =
        resolve_adapter(&metadata).expect("architecture must resolve to a registered adapter");

    let seq = TOKEN_IDS.len();
    let runtime = LlamaRuntime { batch: 1, seq };

    // Build the adapter-specific graph.
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let scratch = adapter.build_scratch_graph(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(scratch.logits_id);
    let mut graph = gb.build();

    // Map + load weights in pure F32 (no BF16 storage → ADR-004 F32 path).
    let mut mapper = adapter
        .map_hf_weights(&config, &scratch.param_names, &scratch.param_ids)
        .expect("build HF weight mapper");
    mapper.set_store_params_as_bf16(false);

    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");
    let report = if index_path.exists() {
        let sharded = ShardedSafetensorsReader::open(&index_path).expect("open shard index");
        sharded.load_into(&mut graph, &mapper).expect("sharded load")
    } else {
        let reader = SafetensorsReader::open(&single_path).expect("open safetensors");
        mapper.load_into(&mut graph, &reader).expect("load")
    };
    assert!(
        report.missing.is_empty(),
        "missing tensors after load: {:?}",
        report.missing
    );

    let tokens_f32: Vec<f32> = TOKEN_IDS.iter().map(|&t| t as f32).collect();
    let tokens = Tensor::new_cpu(vec![1, seq], tokens_f32);
    let outputs = graph.execute(vec![tokens]);
    let logits = outputs[0].as_cpu_slice().to_vec();
    (logits, seq, config.vocab_size)
}

/// Shared body for the per-model `#[ignore]` validations.
fn run_validation(env_var: &str, fixture_dir: &str, label: &str) {
    let dir = env::var(env_var).unwrap_or_else(|_| {
        panic!("Set {env_var} to the {label} checkpoint directory (config.json + safetensors)")
    });
    let model_dir = PathBuf::from(&dir);
    assert!(
        model_dir.join("config.json").is_file(),
        "{env_var} = {dir}: no config.json there"
    );

    println!("\n=== {label} — Atenia F32 vs F64 ground truth (ADR-004) ===");
    let (atenia, seq, vocab) = atenia_cpu_logits(&model_dir);
    let reference = load_f64_fixture(fixture_dir);
    assert_eq!(
        atenia.len(),
        reference.len(),
        "{label}: logit length {} != fixture length {}",
        atenia.len(),
        reference.len()
    );

    let drift = max_abs_diff(&atenia, &reference);
    let matches = per_position_argmax_match(&atenia, &reference, seq, vocab);
    println!("{label}: max_abs_diff vs F64 = {drift:.6} (threshold {ADR_004_THRESHOLD})");
    println!("{label}: per-position argmax matches = {matches:?}");

    assert!(
        drift < ADR_004_THRESHOLD,
        "{label}: drift {drift:.6} exceeds ADR-004 threshold {ADR_004_THRESHOLD}"
    );
    assert!(
        matches.iter().all(|&m| m),
        "{label}: argmax mismatch at one or more positions: {matches:?}"
    );
    println!("{label}: CERTIFIED (CPU F32 vs F64) ✓");
}

// ---------------------------------------------------------------------------
// Per-model validations — `#[ignore]` until a real F64 fixture is generated.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs GEMMA2_2B_DIR + tests/fixtures/gemma2_2b_reference/expected_logits_f64.json (see generate_f64_reference.py)"]
fn gemma2_2b_atenia_f32_matches_f64() {
    run_validation("GEMMA2_2B_DIR", "gemma2_2b_reference", "Gemma-2-2B-IT");
}

#[test]
#[ignore = "needs GEMMA3_1B_DIR + tests/fixtures/gemma3_1b_reference/expected_logits_f64.json (see generate_f64_reference.py)"]
fn gemma3_1b_atenia_f32_matches_f64() {
    run_validation("GEMMA3_1B_DIR", "gemma3_1b_reference", "Gemma-3-1B-IT");
}

#[test]
#[ignore = "needs PHI35_MINI_DIR + tests/fixtures/phi35_mini_reference/expected_logits_f64.json (see generate_f64_reference.py)"]
fn phi35_mini_atenia_f32_matches_f64() {
    run_validation("PHI35_MINI_DIR", "phi35_mini_reference", "Phi-3.5-Mini");
}

// ---------------------------------------------------------------------------
// CI unit tests — pure logic, no model / fixture / CUDA.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn max_abs_diff_picks_worst_element() {
        let atenia = [1.0_f32, 2.0, 3.0];
        let reference = [1.0_f64, 2.5, 3.0];
        let d = max_abs_diff(&atenia, &reference);
        assert!((d - 0.5).abs() < 1e-9, "got {d}");
    }

    #[test]
    fn argmax_match_all_true_when_identical() {
        // seq=2, vocab=3
        let atenia = [0.1_f32, 0.9, 0.2, /*pos1*/ 0.5, 0.4, 0.6];
        let reference = [0.0_f64, 1.0, 0.0, /*pos1*/ 0.1, 0.0, 0.2];
        let m = per_position_argmax_match(&atenia, &reference, 2, 3);
        assert_eq!(m, vec![true, true]);
    }

    #[test]
    fn argmax_match_detects_a_flip() {
        // pos0: atenia argmax=2, reference argmax=0 → mismatch
        let atenia = [0.1_f32, 0.2, 0.9];
        let reference = [0.9_f64, 0.2, 0.1];
        let m = per_position_argmax_match(&atenia, &reference, 1, 3);
        assert_eq!(m, vec![false]);
    }

    #[test]
    fn fixture_parse_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "atenia_cb_fixture_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rel = dir.file_name().unwrap().to_str().unwrap().to_string();
        // load_f64_fixture roots at tests/fixtures, so write there.
        let fixtures_root = PathBuf::from("tests/fixtures").join(&rel);
        fs::create_dir_all(&fixtures_root).unwrap();
        fs::write(
            fixtures_root.join("expected_logits_f64.json"),
            r#"{"shape":[1,1,3],"values":[1.5,2.5,3.5],"dtype":"f64"}"#,
        )
        .unwrap();
        let v = load_f64_fixture(&rel);
        assert_eq!(v, vec![1.5, 2.5, 3.5]);
        let _ = fs::remove_dir_all(&fixtures_root);
    }

    #[test]
    fn missing_fixture_panics_with_actionable_message() {
        let result = std::panic::catch_unwind(|| {
            load_f64_fixture("definitely_does_not_exist_cb1");
        });
        assert!(result.is_err(), "missing fixture must panic");
    }

    #[test]
    fn read_architecture0_extracts_first() {
        let dir = std::env::temp_dir().join(format!("atenia_cb_arch_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("config.json");
        fs::write(&cfg, r#"{"architectures":["Gemma2ForCausalLM"],"model_type":"gemma2"}"#).unwrap();
        assert_eq!(read_architecture0(&cfg).as_deref(), Some("Gemma2ForCausalLM"));
        let _ = fs::remove_dir_all(&dir);
    }
}
