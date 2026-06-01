//! **MOE-FULL-15** — SCALE certification: the controlled runtime is certified
//! end-to-end vs HuggingFace f64 on **topology-representative** fixtures that
//! mirror the real large-checkpoint structure (expert count, top-k routing, GQA
//! ratio, shared experts, MLA) at a reduced hidden dim. This certifies the
//! runtime handles the real topologies — NOT the multi-GB real weights (those
//! cannot be downloaded / committed; see `docs/HANDOFF_MOE_FULL_15.md`).
//!
//!  - `mixtral_scale`   — 8 experts, top-2, GQA 4:1 (the Mixtral-8x7B topology)
//!  - `qwen_scale`      — 16 experts, top-4, shared expert, GQA, qkv bias
//!  - `deepseek_scale`  — 16 routed, top-6, 2 shared experts, MLA
//!
//! Fixtures generated offline by `fixtures/moe/generate_scale_references.py`.

use std::path::PathBuf;

use atenia_engine::moe::{MoeFamily, MoeRuntime, NumericalMetrics};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn opt_in() {
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
    }
}

fn load(name: &str) -> MoeRuntime {
    opt_in();
    MoeRuntime::load_from_files(
        &fixture_dir().join(format!("{name}_config.json")),
        &fixture_dir().join(format!("{name}.safetensors")),
    )
    .unwrap_or_else(|e| panic!("{name} must load: {e}"))
}

fn sidecar(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join(format!("{name}.json"))).unwrap())
        .unwrap()
}

fn f32_vec(j: &serde_json::Value, k: &str) -> Vec<f32> {
    j[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn u32_vec(j: &serde_json::Value, k: &str) -> Vec<u32> {
    j[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}
fn argmax(r: &[f32]) -> usize {
    r.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

/// Certify one scale fixture: family, full-forward vs HF, per-position argmax,
/// generate → EOS, determinism. Returns the full-forward max_abs_diff.
fn certify_scale(name: &str, family: MoeFamily, drift_bound: f64) -> f64 {
    let rt = load(name);
    assert_eq!(rt.family(), family, "{name} family");
    let j = sidecar(name);
    let vocab = j["vocab_size"].as_u64().unwrap() as usize;
    let ids = u32_vec(&j, "input_ids");
    let hf = f32_vec(&j, "hf_logits");

    let got = rt.forward_logits(&ids);
    assert_eq!(got.len(), hf.len());
    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!("SCALE CERT {name} ({}) vs HF: max_abs_diff={:.3e}", family.name(), m.max_abs_diff);
    assert!(m.max_abs_diff < drift_bound, "{name} drift {:.3e} >= {drift_bound:.0e}", m.max_abs_diff);

    let seq = ids.len();
    for pos in 0..seq {
        assert_eq!(
            argmax(&got[pos * vocab..(pos + 1) * vocab]),
            argmax(&hf[pos * vocab..(pos + 1) * vocab]),
            "{name} argmax mismatch at {pos}"
        );
    }

    // generate → EOS. The emitted sequence equals the greedy prefix up to and
    // INCLUDING the first token equal to EOS (greedy is the full-recompute
    // reference; the KV-cache decode reproduces it, then stops on EOS).
    let greedy = u32_vec(&j, "greedy_ids");
    let eos = j["eos_token_id"].as_u64().unwrap() as u32;
    let first_eos = greedy.iter().position(|&t| t == eos).unwrap();
    let expected: Vec<u32> = greedy[..=first_eos].to_vec();
    let out = rt.generate(&ids, 8);
    eprintln!("SCALE CERT {name} generate = {out:?} (greedy {greedy:?}, eos {eos})");
    assert_eq!(out, expected, "{name} greedy/eos");
    assert_eq!(*out.last().unwrap(), eos);
    assert_eq!(out, rt.generate(&ids, 8), "{name} determinism");
    m.max_abs_diff
}

#[test]
fn mixtral_8x7b_topology_certifies() {
    // 8 experts, top-2, GQA 4:1 — the Mixtral-8x7B routing/attention topology.
    certify_scale("mixtral_scale", MoeFamily::Mixtral, 1e-3);
}

#[test]
fn qwen_moe_scale_topology_certifies() {
    // 16 experts, top-4, shared expert (sigmoid gate), GQA, qkv bias.
    certify_scale("qwen_scale", MoeFamily::QwenMoe, 1e-3);
}

#[test]
fn deepseek_scale_topology_certifies() {
    // 16 routed, top-6, 2 shared experts, MLA. The full-forward drift is f32-
    // vs-f64 (MoE-block dominated, more experts → larger accumulation); argmax /
    // greedy are exact. Documented bound (see handoff).
    let d = certify_scale("deepseek_scale", MoeFamily::DeepSeekMoe, 1e-2);
    eprintln!("DEEPSEEK SCALE full-forward drift = {d:.3e}");
}
