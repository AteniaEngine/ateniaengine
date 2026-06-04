//! **MLA-0** — the experimental MLA runtime reproduces a tiny **DeepSeek-V2-Lite-
//! like** checkpoint with the three features V2-Lite needs that MLA-0 adds:
//!   * **YaRN** rope_scaling (inv_freq reparam + mscale² on the attention scale,
//!     active at every position),
//!   * **`first_k_dense_replace=1`** (layer 0 is a dense SwiGLU MLP, 1.. are MoE),
//!   * **`norm_topk_prob=false`** (no top-k renorm; ungated shared expert).
//!
//! Fixture: `fixtures/moe/deepseek_v2lite_mla0.{safetensors,json}` +
//! `deepseek_v2lite_mla0_config.json`, generated offline from a real
//! `DeepseekV2ForCausalLM` (`.double()`). No model downloaded. This validates the
//! MLA-0 implementation against HuggingFace; it does NOT certify a real model.

use std::path::PathBuf;

use atenia_engine::moe::{MoeFamily, MoeRuntime};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn load() -> MoeRuntime {
    // SAFETY: single-threaded test; owns the opt-in env toggle.
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
    MoeRuntime::load_from_files(
        &fixture_dir().join("deepseek_v2lite_mla0_config.json"),
        &fixture_dir().join("deepseek_v2lite_mla0.safetensors"),
    )
    .expect("MLA-0 DeepSeek-V2-Lite-like runtime must load (dense-first + YaRN)")
}

fn sidecar() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("deepseek_v2lite_mla0.json")).unwrap(),
    )
    .unwrap()
}

fn f32_vec(j: &serde_json::Value, key: &str) -> Vec<f32> {
    j[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn u32_vec(j: &serde_json::Value, key: &str) -> Vec<u32> {
    j[key].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}
fn argmax(r: &[f32]) -> usize {
    r.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn mla0_v2lite_loads_with_dense_first_layer() {
    // `first_k_dense_replace=1` must not break assembly (the old code assumed
    // every layer is MoE and would fail to assemble the dense layer 0).
    let rt = load();
    assert_eq!(rt.family(), MoeFamily::DeepSeekMoe);
    assert_eq!(rt.num_layers(), 3);
}

#[test]
fn mla0_v2lite_full_forward_matches_hf() {
    let rt = load();
    let j = sidecar();
    let ids = u32_vec(&j, "input_ids");
    let vocab = j["vocab_size"].as_u64().unwrap() as usize;
    let hf = f32_vec(&j, "hf_logits");

    let got = rt.forward_logits(&ids);
    assert_eq!(got.len(), hf.len(), "logits dims");
    let m = max_abs(&got, &hf);
    eprintln!("MLA-0 DeepSeek-V2-Lite-like FULL FORWARD vs HF (YaRN + dense-first): max_abs_diff={m:.3e}");

    // ADR-004 gate (unchanged, not lowered). The f32-vs-f64 DeepSeek block drift
    // is ~1e-3 (documented); argmax must be exact.
    assert!(m < 0.5, "MLA-0 full forward over the ADR-004 gate: max_abs_diff={m:.3e}");
    for pos in 0..ids.len() {
        assert_eq!(
            argmax(&got[pos * vocab..(pos + 1) * vocab]),
            argmax(&hf[pos * vocab..(pos + 1) * vocab]),
            "argmax mismatch at position {pos} (YaRN / dense-first / routing convention wrong?)"
        );
    }
    // It also clears the tighter empirical bound the existing DeepSeek tests use.
    assert!(m < 5e-3, "MLA-0 drift larger than the DeepSeek empirical bound: {m:.3e}");
}

#[test]
fn mla0_v2lite_greedy_generates_to_eos() {
    let rt = load();
    let j = sidecar();
    let ids = u32_vec(&j, "input_ids");
    let greedy = u32_vec(&j, "greedy_ids");
    let eos = j["eos_token_id"].as_u64().unwrap() as u32;
    let first_eos = greedy.iter().position(|&t| t == eos).unwrap();
    let expected: Vec<u32> = greedy[..=first_eos].to_vec();
    let out = rt.generate(&ids, 8);
    eprintln!("MLA-0 generate = {out:?} (greedy {greedy:?}, eos {eos})");
    assert_eq!(out, expected, "greedy/EOS mismatch");
    assert_eq!(out, rt.generate(&ids, 8), "generation must be deterministic");
}
