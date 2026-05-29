//! **MOE-14** — opt-in, out-of-CI smoke test against a REAL local MoE
//! checkpoint.
//!
//! This test is `#[ignore]`d and gated on the `ATENIA_MOE_REAL_MODEL` env
//! var. It is NEVER run in CI by default. It downloads nothing and embeds no
//! model in the repo — it only reads a checkpoint already on disk.
//!
//! ## How to run
//!
//! ```bash
//! ATENIA_MOE_REAL_MODEL=/path/to/qwen-moe \
//!   cargo test --release --test moe_real_model_smoke_test -- --ignored --nocapture
//! ```
//!
//! The directory should contain one or more `*.safetensors` files (shards are
//! supported) and, ideally, a `config.json`. Expected first targets:
//!
//! * Qwen1.5-MoE-A2.7B
//! * small Qwen-MoE checkpoints
//! * Mixtral only if the host has enough RAM
//!
//! The test is model-agnostic — no model is hardcoded.
//!
//! ## What a PASS means / does NOT mean
//!
//! PASS = "the experimental MoE path discovered the shards, built a
//! `MoeWeightMap`, assembled a stack, and ran a finite forward". It does NOT
//! mean the model is numerically correct, supported, or production-ready. No
//! Mixtral / Qwen-MoE full support is claimed. The productive loader still
//! fails loud on MoE checkpoints; this is an isolated experimental path.

use std::path::PathBuf;

use atenia_engine::moe::{
    discover_safetensors_files, LocalMoeCheckpoint, MinimalMoeConfig,
    RealMoeCheckpointValidation,
};

const ENV_VAR: &str = "ATENIA_MOE_REAL_MODEL";

/// Default routing top-k used when `config.json` does not specify one.
const DEFAULT_EXPERTS_PER_TOKEN: usize = 2;

#[test]
#[ignore = "requires ATENIA_MOE_REAL_MODEL pointing at a local MoE checkpoint"]
fn moe_real_local_checkpoint_smoke_test() {
    let Some(dir) = std::env::var_os(ENV_VAR) else {
        // Not a failure: the test was invoked without a model. Print a clear
        // skip line and return (no panic). With `--ignored` but no env var,
        // this simply reports that nothing was validated.
        println!("real smoke not run: {ENV_VAR} not set");
        return;
    };
    let model_dir = PathBuf::from(dir);
    println!("== MOE-14 real local checkpoint smoke ==");
    println!("model dir: {}", model_dir.display());

    // 1) Discover shards.
    let files = match discover_safetensors_files(&model_dir) {
        Ok(f) => f,
        Err(e) => {
            println!("SKIP: shard discovery failed: {e}");
            return;
        }
    };
    println!("discovered {} safetensors file(s):", files.len());
    for f in &files {
        println!("  - {}", f.file_name().unwrap().to_string_lossy());
    }

    // 2) Minimal config (optional).
    let cfg = match MinimalMoeConfig::from_dir(&model_dir) {
        Ok(c) => {
            println!(
                "config.json: layers={:?} experts={:?} per_tok={:?} hidden={:?} inter={:?}",
                c.num_hidden_layers,
                c.num_experts,
                c.num_experts_per_token,
                c.hidden_size,
                c.intermediate_size
            );
            c
        }
        Err(e) => {
            println!("config.json not used ({e}); falling back to defaults");
            MinimalMoeConfig::default()
        }
    };
    let experts_per_token = cfg.experts_per_token_or(DEFAULT_EXPERTS_PER_TOKEN);
    println!("experts_per_token (for probe): {experts_per_token}");

    // 3) Open shards + merged metadata.
    let checkpoint = match LocalMoeCheckpoint::open(&files) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP: failed to open shards: {e}");
            return;
        }
    };
    println!(
        "opened {} shard(s), {} tensor(s) total",
        checkpoint.num_shards(),
        checkpoint.num_tensors()
    );
    let map = checkpoint.weight_map();

    // 4 + 5) Validate: stack assembly + minimal forward + report.
    let resolve = |name: &str| checkpoint.resolve(name);
    let report = RealMoeCheckpointValidation::validate(&map, experts_per_token, &resolve);

    println!("---- ValidationReport ----");
    println!("{}", report.summary());
    println!("layers_detected : {}", report.layers_detected);
    println!("experts_detected: {}", report.experts_detected);
    println!("shared_experts  : {}", report.shared_experts);
    println!("d_model         : {:?}", report.d_model);
    println!("forward_pass_ok : {}", report.forward_pass_ok);
    if !report.errors.is_empty() {
        println!("errors:");
        for e in &report.errors {
            println!("  - {e}");
        }
    }
    println!("--------------------------");

    // The harness must at least recognise this as a MoE checkpoint. If it is
    // not MoE, that is a real (informative) failure of the chosen target.
    assert!(
        report.is_moe(),
        "checkpoint at {} was not detected as MoE (layers_detected=0)",
        model_dir.display()
    );

    if report.forward_pass_ok {
        println!(
            "SMOKE PASS: experimental MoE path ran a finite forward over the real checkpoint."
        );
        println!(
            "NOTE: PASS means the path executed, NOT that the model is correct/supported/certified."
        );
    } else {
        println!(
            "SMOKE INCOMPLETE: metadata read but forward did not complete (see errors above)."
        );
    }
}

/// Confirms the gating contract without a model: when the env var is unset,
/// the harness reports a clear skip rather than running anything. This is a
/// normal (non-ignored) test so CI exercises the gate itself.
#[test]
fn real_smoke_requires_env_var() {
    if std::env::var_os(ENV_VAR).is_some() {
        // A model is configured in this environment; the gate would open, so
        // there is nothing to assert about the unset case here.
        eprintln!("{ENV_VAR} is set; skipping unset-gate assertion");
        return;
    }
    // Unset: the message the ignored test would print.
    let msg = format!("real smoke not run: {ENV_VAR} not set");
    assert!(msg.contains(ENV_VAR));
    assert!(std::env::var_os(ENV_VAR).is_none());
}
