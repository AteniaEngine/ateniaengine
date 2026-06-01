//! **MOE-FULL-13** — production certification campaign: expand **end-to-end
//! runtime** coverage for the Mixtral family to three real on-disk layouts
//! (packed-MHA already certified in MOE-FULL-10; classic-MHA and packed-GQA
//! here), plus robustness for corrupt router / MLA shapes.
//!
//! Fixtures:
//!  - `mixtral_classic.{safetensors}` + `mixtral_classic_config.json` — the
//!    committed packed `full_mixtral` weights re-packed into the CLASSIC
//!    `block_sparse_moe.experts.{e}.w1/w3/w2` layout (same numbers → same HF
//!    reference `full_mixtral.json`).
//!  - `gqa_mixtral.{safetensors,json}` + `gqa_mixtral_config.json` +
//!    `gqa_mixtral_gen.json` — the MOE-FULL-9 GQA Mixtral (n_kv=2), now driven
//!    through the productive runtime.
//! No model downloaded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use atenia_engine::moe::{MoeFamily, MoeRuntime, MoeRuntimeError, NumericalMetrics};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn opt_in() {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
}

fn json(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join(name)).unwrap()).unwrap()
}

fn f32_vec(j: &serde_json::Value, key: &str) -> Vec<f32> {
    j[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn u32_vec(j: &serde_json::Value, key: &str) -> Vec<u32> {
    j[key].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

// ---- Mixtral CLASSIC layout, end-to-end through the runtime ----

#[test]
fn mixtral_classic_runtime_matches_hf_and_generates() {
    opt_in();
    let rt = MoeRuntime::load_from_files(
        &fixture_dir().join("mixtral_classic_config.json"),
        &fixture_dir().join("mixtral_classic.safetensors"),
    )
    .expect("classic Mixtral must load");
    assert_eq!(rt.family(), MoeFamily::Mixtral);

    // Same weights as full_mixtral → same HF f64 logits.
    let j = json("full_mixtral.json");
    let ids = u32_vec(&j, "input_ids");
    let hf = f32_vec(&j, "hf_logits");
    let vocab = j["vocab_size"].as_u64().unwrap() as usize;
    let got = rt.forward_logits(&ids);
    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!("MIXTRAL CLASSIC runtime vs HF: max_abs_diff={:.3e}", m.max_abs_diff);
    assert!(m.max_abs_diff < 1e-3, "classic Mixtral must match HF: {m:?}");
    let seq = ids.len();
    for pos in 0..seq {
        assert_eq!(
            argmax(&got[pos * vocab..(pos + 1) * vocab]),
            argmax(&hf[pos * vocab..(pos + 1) * vocab]),
            "classic argmax mismatch at {pos}"
        );
    }

    // Greedy [17,20,...] from prompt [22,25,29]; eos=20 → [17,20].
    let g = json("full_mixtral_gen.json");
    let prompt = u32_vec(&g, "prompt_ids");
    let out = rt.generate(&prompt, 8);
    eprintln!("MIXTRAL CLASSIC generate = {out:?}");
    assert_eq!(out, vec![17, 20]);
    assert_eq!(out, rt.generate(&prompt, 8), "deterministic");
}

// ---- Mixtral packed GQA (n_kv=2), end-to-end through the runtime ----

#[test]
fn mixtral_gqa_runtime_matches_hf_and_generates() {
    opt_in();
    let rt = MoeRuntime::load_from_files(
        &fixture_dir().join("gqa_mixtral_config.json"),
        &fixture_dir().join("gqa_mixtral.safetensors"),
    )
    .expect("GQA Mixtral must load");
    assert_eq!(rt.family(), MoeFamily::Mixtral);
    assert_eq!(rt.num_layers(), 2);

    let j = json("gqa_mixtral.json");
    let ids = u32_vec(&j, "input_ids");
    let hf = f32_vec(&j, "hf_logits");
    let vocab = j["vocab_size"].as_u64().unwrap() as usize;
    let got = rt.forward_logits(&ids);
    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!("MIXTRAL GQA runtime vs HF: max_abs_diff={:.3e}", m.max_abs_diff);
    assert!(m.max_abs_diff < 1e-3, "GQA Mixtral must match HF: {m:?}");
    let seq = ids.len();
    for pos in 0..seq {
        assert_eq!(
            argmax(&got[pos * vocab..(pos + 1) * vocab]),
            argmax(&hf[pos * vocab..(pos + 1) * vocab]),
            "GQA argmax mismatch at {pos}"
        );
    }

    let g = json("gqa_mixtral_gen.json");
    let greedy = u32_vec(&g, "greedy_ids");
    let eos = g["eos_token_id"].as_u64().unwrap() as u32;
    let prompt: Vec<u32> = ids[..3].to_vec();
    let out = rt.generate(&prompt, 8);
    eprintln!("MIXTRAL GQA generate = {out:?} (greedy {greedy:?}, eos {eos})");
    assert_eq!(out, vec![greedy[0], greedy[1]]);
    assert_eq!(*out.last().unwrap(), eos);
}

// ---- Robustness (FASE 5): corrupt router / MLA shapes → clear errors ----

fn write_st(path: &Path, tensors: &[(&str, Vec<usize>)]) {
    let buffers: Vec<Vec<u8>> =
        tensors.iter().map(|(_, s)| vec![0u8; s.iter().product::<usize>() * 4]).collect();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        views.insert((*name).to_string(), TensorView::new(StDtype::F32, shape.clone(), &buffers[i]).unwrap());
    }
    std::fs::write(path, safetensors::serialize(&views, &None).unwrap()).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atenia_moe13_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn corrupt_router_shape_reports_error() {
    opt_in();
    // Classic Mixtral whose router declares 5 experts but only 4 expert sets
    // exist → a clear error (router/expert count mismatch), not a panic.
    let dm = 8;
    let dff = 16;
    let mut t: Vec<(&str, Vec<usize>)> = vec![
        ("model.embed_tokens.weight", vec![12, dm]),
        ("lm_head.weight", vec![12, dm]),
        ("model.norm.weight", vec![dm]),
        ("model.layers.0.input_layernorm.weight", vec![dm]),
        ("model.layers.0.post_attention_layernorm.weight", vec![dm]),
        ("model.layers.0.self_attn.q_proj.weight", vec![dm, dm]),
        ("model.layers.0.self_attn.k_proj.weight", vec![dm, dm]),
        ("model.layers.0.self_attn.v_proj.weight", vec![dm, dm]),
        ("model.layers.0.self_attn.o_proj.weight", vec![dm, dm]),
        ("model.layers.0.block_sparse_moe.gate.weight", vec![5, dm]), // 5 != 4
    ];
    for e in 0..4 {
        let w1: &'static str = Box::leak(format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight").into_boxed_str());
        let w3: &'static str = Box::leak(format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight").into_boxed_str());
        let w2: &'static str = Box::leak(format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight").into_boxed_str());
        t.push((w1, vec![dff, dm]));
        t.push((w3, vec![dff, dm]));
        t.push((w2, vec![dm, dff]));
    }
    let w = tmp("badrouter.safetensors");
    write_st(&w, &t);
    let c = tmp("badrouter_config.json");
    std::fs::write(&c, r#"{"vocab_size":12,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
        "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"num_local_experts":4,
        "num_experts_per_tok":2,"rope_theta":10000.0,"rms_norm_eps":1e-5,"eos_token_id":0}"#).unwrap();
    let err = MoeRuntime::load_from_files(&c, &w).unwrap_err();
    assert!(
        matches!(err, MoeRuntimeError::Load(_) | MoeRuntimeError::ConfigInconsistent(_)),
        "expected a clear error, got {err:?}"
    );
}

#[test]
fn corrupt_mla_shape_reports_error() {
    opt_in();
    // Load the real DeepSeek weights but with a config that lies about
    // kv_lora_rank → the MLA shape check rejects it (no silent panic).
    let bad_cfg = tmp("badmla_config.json");
    let good = std::fs::read_to_string(fixture_dir().join("deepseek_full_config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&good).unwrap();
    v["kv_lora_rank"] = serde_json::json!(16); // real is 8
    std::fs::write(&bad_cfg, v.to_string()).unwrap();
    let err = MoeRuntime::load_from_files(&bad_cfg, &fixture_dir().join("deepseek_full.safetensors"))
        .unwrap_err();
    assert!(
        matches!(err, MoeRuntimeError::Load(_)),
        "MLA shape mismatch must be a clear Load error, got {err:?}"
    );
}
