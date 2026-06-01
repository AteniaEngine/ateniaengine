//! **MOE-FULL-12** — integration test: the controlled MoE runtime loads a real
//! tiny **DeepSeek-V2 (MLA)** checkpoint and (a) reproduces the layer-0 MLA
//! attention output, (b) reproduces the HuggingFace f64 full-transformer
//! logits, and (c) generates greedily to EOS — all through `MoeRuntime`, behind
//! the opt-in.
//!
//! Fixtures: `fixtures/moe/deepseek_full.{safetensors,json}` +
//! `deepseek_full_config.json`, generated offline by
//! `fixtures/moe/generate_deepseek_full_reference.py`. No model downloaded.

use std::path::PathBuf;

use atenia_engine::moe::{MoeFamily, MoeRuntime};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn enable_opt_in() {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
}

fn load() -> MoeRuntime {
    enable_opt_in();
    MoeRuntime::load_from_files(
        &fixture_dir().join("deepseek_full_config.json"),
        &fixture_dir().join("deepseek_full.safetensors"),
    )
    .expect("DeepSeek runtime must load with the opt-in")
}

fn sidecar() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("deepseek_full.json")).unwrap(),
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
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn deepseek_runtime_recognizes_mla_family() {
    let rt = load();
    assert_eq!(rt.family(), MoeFamily::DeepSeekMoe);
    assert_eq!(rt.num_layers(), 2);
}

#[test]
fn mla_attention_matches_hf_layer0() {
    // Certify the MLA attention in isolation: feed the HF-captured post-input-LN
    // hidden states and compare to the HF self-attention output.
    let rt = load();
    let j = sidecar();
    let hidden = j["hidden"].as_u64().unwrap() as usize;
    let attn_in = f32_vec(&j, "attn0_in");
    let attn_out = f32_vec(&j, "attn0_out");
    let seq = attn_in.len() / hidden;
    let normed: Vec<Vec<f32>> =
        (0..seq).map(|t| attn_in[t * hidden..(t + 1) * hidden].to_vec()).collect();

    let got = rt.debug_mla_attention(0, &normed).expect("MLA backend");
    let flat: Vec<f32> = got.into_iter().flatten().collect();
    let m = max_abs(&flat, &attn_out);
    eprintln!("MLA ATTENTION (layer 0) vs HF: max_abs_diff={m:.3e}");
    assert!(m < 1e-3, "MLA attention must match HF within 1e-3: max_abs_diff={m:.3e}");
}

#[test]
fn deepseek_full_forward_matches_hf_reference() {
    let rt = load();
    let j = sidecar();
    let vocab = j["vocab_size"].as_u64().unwrap() as usize;
    let ids = u32_vec(&j, "input_ids");
    let hf = f32_vec(&j, "hf_logits");
    let got = rt.forward_logits(&ids);
    assert_eq!(got.len(), hf.len());
    let m = max_abs(&got, &hf);
    eprintln!("DEEPSEEK FULL FORWARD vs HF: max_abs_diff={m:.3e}");
    // Drift is dominated by the f32-vs-f64 MoE block (~2e-4/layer, see
    // MOE-FULL-11) accumulated over 2 layers + lm_head; the MLA attention itself
    // matches to ~1e-5. Argmax (the next-token signal) is unaffected — see the
    // per-position check below and the exact greedy match in the generation
    // test. Documented bound: < 5e-3.
    assert!(m < 5e-3, "DeepSeek full forward drift too large: max_abs_diff={m:.3e}");

    let seq = ids.len();
    for pos in 0..seq {
        let a = &got[pos * vocab..(pos + 1) * vocab];
        let b = &hf[pos * vocab..(pos + 1) * vocab];
        let am = a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let bm = b.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        assert_eq!(am, bm, "DeepSeek argmax mismatch at position {pos}");
    }
}

#[test]
fn deepseek_generates_to_eos_deterministically() {
    let rt = load();
    let j = sidecar();
    let ids = u32_vec(&j, "input_ids");
    let greedy = u32_vec(&j, "greedy_ids");
    let eos = j["eos_token_id"].as_u64().unwrap() as u32;

    let out = rt.generate(&ids, 8);
    eprintln!("DEEPSEEK generate = {out:?} (greedy ref {greedy:?}, eos {eos})");
    // greedy [g0, eos, ...] → KV-cache decode stops at eos: [g0, eos].
    assert_eq!(out, vec![greedy[0], greedy[1]]);
    assert_eq!(*out.last().unwrap(), eos, "must terminate on EOS");
    assert!(out.len() < 8);
    assert_eq!(out, rt.generate(&ids, 8), "deterministic");
}
