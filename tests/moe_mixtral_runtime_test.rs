//! **MOE-FULL-10** — integration test: the controlled productive Mixtral
//! runtime loads a real tiny Mixtral checkpoint (HF `config.json` + committed
//! `full_mixtral.safetensors`) and generates greedily to EOS, behind the
//! `ATENIA_EXPERIMENTAL_MOE=1` opt-in. Reuses the certified MoE pipeline
//! (family recognition, adapter validation, GQA tiling, residency + cache,
//! prefill/KV-cache/decode generation) — no experimental test helpers.
//!
//! Fixtures: `fixtures/moe/mixtral_tiny_config.json` (eos_token_id=20) +
//! `fixtures/moe/full_mixtral.safetensors` (MOE-FULL-6). No model downloaded.

use std::path::PathBuf;

use atenia_engine::moe::{classify_family, MixtralRuntime, MixtralRuntimeError, MoeFamily};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn config_path() -> PathBuf {
    fixture_dir().join("mixtral_tiny_config.json")
}
fn weights_path() -> PathBuf {
    fixture_dir().join("full_mixtral.safetensors")
}

/// End-to-end controlled path. One test owns the env var across both the
/// disabled and enabled branches so it cannot race sibling tests.
#[test]
fn controlled_mixtral_load_generate_eos() {
    // (a) Without the opt-in, the runtime refuses (fail-loud preserved).
    unsafe {
        std::env::remove_var("ATENIA_EXPERIMENTAL_MOE");
    }
    let err = MixtralRuntime::load_from_files(&config_path(), &weights_path()).unwrap_err();
    assert!(
        matches!(err, MixtralRuntimeError::OptInDisabled),
        "without opt-in the runtime must refuse (got {err:?})"
    );

    // (b) With the opt-in, it loads and generates.
    // SAFETY: this test owns the flag; no sibling test reads it.
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }

    let rt = MixtralRuntime::load_from_files(&config_path(), &weights_path())
        .expect("controlled Mixtral runtime must load with the opt-in");

    // Config wired from the real config.json.
    assert_eq!(rt.num_layers(), 2);
    assert_eq!(rt.eos_token_ids(), &[20]);
    // Residency + cache wired (and self-validated at load).
    assert_eq!(rt.residency().len(), 2);
    assert_eq!(rt.caches().len(), 2);

    // Generate: prompt [22,25,29] greedily yields [17,20,10,17]; eos=20 stops
    // the run at step 1 → [17,20]. This is load → generate → EOS end-to-end.
    let prompt = [22u32, 25, 29];
    let out = rt.generate(&prompt, 8);
    eprintln!("MIXTRAL RUNTIME generate({prompt:?}, 8) = {out:?}");
    assert_eq!(out, vec![17, 20], "controlled runtime must generate then stop at EOS");
    assert_eq!(*out.last().unwrap(), 20, "generation must terminate on the EOS token");
    assert!(out.len() < 8, "EOS must stop generation before max_new_tokens");

    // Determinism.
    let out2 = rt.generate(&prompt, 8);
    assert_eq!(out, out2, "controlled generation must be deterministic");
}

/// Dense checkpoints are unaffected — not classified as any MoE family.
#[test]
fn dense_is_not_a_mixtral_family() {
    let dense = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ];
    assert_eq!(classify_family(dense.into_iter()), None);
    let _ = MoeFamily::Mixtral; // type is reachable
}
