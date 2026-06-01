//! **MOE-FULL-11** — integration test: the controlled MoE runtime loads a real
//! tiny **Qwen2-MoE** checkpoint (GQA, Q/K/V attention bias, packed experts,
//! shared expert with sigmoid gate, `norm_topk_prob=false`) and (a) reproduces
//! the HuggingFace f64 full-transformer logits and (b) generates greedily to
//! EOS — all through the productive `MoeRuntime`, behind the opt-in.
//!
//! Fixtures: `fixtures/moe/qwen_moe_tiny.{safetensors,json}` +
//! `qwen_moe_tiny_config.json`, generated offline by
//! `fixtures/moe/generate_qwen_moe_reference.py`. No model downloaded.

use std::path::PathBuf;

use atenia_engine::moe::{MoeFamily, MoeRuntime, NumericalMetrics};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn enable_opt_in() {
    // SAFETY: this test file's only env reader; never unset concurrently.
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
}

fn load_runtime() -> MoeRuntime {
    enable_opt_in();
    MoeRuntime::load_from_files(
        &fixture_dir().join("qwen_moe_tiny_config.json"),
        &fixture_dir().join("qwen_moe_tiny.safetensors"),
    )
    .expect("Qwen-MoE runtime must load with the opt-in")
}

fn sidecar() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("qwen_moe_tiny.json")).unwrap(),
    )
    .unwrap()
}

#[test]
fn qwen_runtime_recognizes_family_and_config() {
    let rt = load_runtime();
    assert_eq!(rt.family(), MoeFamily::QwenMoe);
    assert_eq!(rt.num_layers(), 2);
    let j = sidecar();
    assert_eq!(rt.eos_token_ids(), &[j["eos_token_id"].as_u64().unwrap() as u32]);
    assert_eq!(rt.residency().len(), 2);
}

#[test]
fn qwen_full_forward_matches_hf_reference() {
    let rt = load_runtime();
    let j = sidecar();
    let vocab = j["vocab_size"].as_u64().unwrap() as usize;
    let ids: Vec<u32> =
        j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let hf: Vec<f32> =
        j["hf_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();

    let got = rt.forward_logits(&ids);
    assert_eq!(got.len(), hf.len());
    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!(
        "QWEN-MOE FULL FORWARD vs HF: max_abs_diff={:.3e} mean_abs_diff={:.3e} rmse={:.3e}",
        m.max_abs_diff, m.mean_abs_diff, m.rmse
    );
    assert!(m.max_abs_diff < 1e-3, "Qwen-MoE full forward must match HF within 1e-3: {m:?}");

    let seq = ids.len();
    for pos in 0..seq {
        let a = &got[pos * vocab..(pos + 1) * vocab];
        let b = &hf[pos * vocab..(pos + 1) * vocab];
        let am = a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let bm = b.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        assert_eq!(am, bm, "Qwen-MoE argmax mismatch at position {pos}");
    }
}

#[test]
fn qwen_runtime_generates_to_eos_deterministically() {
    let rt = load_runtime();
    let j = sidecar();
    let ids: Vec<u32> =
        j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let greedy: Vec<u32> =
        j["greedy_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let eos = j["eos_token_id"].as_u64().unwrap() as u32;

    let out = rt.generate(&ids, 8);
    eprintln!("QWEN-MOE generate = {out:?} (greedy ref {greedy:?}, eos {eos})");
    // Greedy [g0, eos, ...] → runtime stops at eos: [g0, eos].
    assert_eq!(out, vec![greedy[0], greedy[1]]);
    assert_eq!(*out.last().unwrap(), eos, "must terminate on EOS");
    assert!(out.len() < 8, "EOS must stop before max_new_tokens");
    // Determinism.
    assert_eq!(out, rt.generate(&ids, 8));
}
