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

use atenia_engine::moe::{classify_family, MixtralRuntime, MoeFamily, NumericalMetrics};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn config_path() -> PathBuf {
    fixture_dir().join("mixtral_tiny_config.json")
}
fn weights_path() -> PathBuf {
    fixture_dir().join("full_mixtral.safetensors")
}

/// End-to-end controlled path: load → generate → EOS → deterministic. The
/// opt-in-disabled refusal is covered by the lib unit test `opt_in_disabled_refuses`
/// (in a process where no test sets the env var); here every test only *sets*
/// the flag, so sibling tests in this binary never race on it.
#[test]
fn controlled_mixtral_load_generate_eos() {
    // SAFETY: all tests in this file only set (never unset) the opt-in flag.
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

/// **MOE-FULL-11** — extended Mixtral validation: the productive runtime's
/// full-sequence forward logits match the HuggingFace f64 reference (not just
/// the greedy ids). Ties the runtime to the MOE-FULL-6 logit certification.
#[test]
fn mixtral_runtime_forward_matches_hf() {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
    let rt = MixtralRuntime::load_from_files(&config_path(), &weights_path()).unwrap();

    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("full_mixtral.json")).unwrap(),
    )
    .unwrap();
    let ids: Vec<u32> =
        j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let hf: Vec<f32> =
        j["hf_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();

    let got = rt.forward_logits(&ids);
    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!("MIXTRAL RUNTIME forward vs HF: max_abs_diff={:.3e}", m.max_abs_diff);
    assert!(m.max_abs_diff < 1e-3, "Mixtral runtime forward must match HF: {m:?}");
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
