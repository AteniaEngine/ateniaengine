//! **MOE-FULL-7** — integration test: the experimental MoE generation path
//! (prefill + KV cache + incremental decode) reproduces an offline HuggingFace
//! `MixtralForCausalLM` f64 **greedy** reference.
//!
//! Weights: the same committed tiny Mixtral as MOE-FULL-6
//! (`fixtures/moe/full_mixtral.safetensors`). Greedy reference:
//! `fixtures/moe/full_mixtral_gen.json` (prompt, generated ids, per-step f64
//! logits), produced offline by `fixtures/moe/generate_decode_reference.py`.
//! No model downloaded, no HF at test time, no productive runtime.

use std::path::PathBuf;

use atenia_engine::moe::full_forward::{TinyDecoderWeights, TinyMixtralConfig, TinyMixtralWeights};
use atenia_engine::moe::generate::generate_greedy_tiny;
use atenia_engine::moe::{detect_moe, MoeLayerConfig, MoeWeightMap, RealMoeLayer};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn read_json(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(fixture_dir().join(name)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn usize_f(j: &serde_json::Value, k: &str) -> usize {
    j[k].as_u64().unwrap() as usize
}

fn load_weights() -> TinyMixtralWeights {
    let j = read_json("full_mixtral.json");
    let hidden = usize_f(&j, "hidden_size");
    let n_layers = usize_f(&j, "num_hidden_layers");
    let n_experts = usize_f(&j, "num_local_experts");
    let topk = usize_f(&j, "num_experts_per_tok");
    let d_ff = usize_f(&j, "intermediate_size");

    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    let get = |name: &str| {
        reader.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).to_vec_f32().unwrap()
    };
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());

    let layers: Vec<TinyDecoderWeights> = (0..n_layers)
        .map(|l| {
            let p = format!("model.layers.{l}");
            let moe_cfg = MoeLayerConfig::new(n_experts, topk, false, hidden, d_ff).unwrap();
            let moe = RealMoeLayer::assemble(&map, l, moe_cfg, &resolve).unwrap();
            TinyDecoderWeights {
                input_ln: get(&format!("{p}.input_layernorm.weight")),
                w_q: get(&format!("{p}.self_attn.q_proj.weight")),
                w_k: get(&format!("{p}.self_attn.k_proj.weight")),
                w_v: get(&format!("{p}.self_attn.v_proj.weight")),
                w_o: get(&format!("{p}.self_attn.o_proj.weight")),
                post_ln: get(&format!("{p}.post_attention_layernorm.weight")),
                attn_bias: None,
                moe,
            }
        })
        .collect();

    TinyMixtralWeights {
        config: TinyMixtralConfig {
            vocab_size: usize_f(&j, "vocab_size"),
            hidden_size: hidden,
            num_hidden_layers: n_layers,
            num_attention_heads: usize_f(&j, "num_attention_heads"),
            head_dim: usize_f(&j, "head_dim"),
            rope_theta: j["rope_theta"].as_f64().unwrap() as u32,
        },
        embed_tokens: get("model.embed_tokens.weight"),
        layers,
        final_norm: get("model.norm.weight"),
        lm_head: get("lm_head.weight"),
        rms_eps: j["rms_norm_eps"].as_f64().unwrap() as f32,
    }
}

fn load_reference() -> (Vec<u32>, usize, Vec<u32>, Vec<f32>) {
    let g = read_json("full_mixtral_gen.json");
    let prompt: Vec<u32> =
        g["prompt_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let max_new = usize_f(&g, "max_new_tokens");
    let gen_ids: Vec<u32> =
        g["generated_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let step_logits: Vec<f32> =
        g["step_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    (prompt, max_new, gen_ids, step_logits)
}

#[test]
fn decode_generation_runs_and_shapes() {
    let w = load_weights();
    let (prompt, max_new, _, _) = load_reference();
    let out = generate_greedy_tiny(&w, &prompt, max_new);
    assert_eq!(out.tokens.len(), max_new);
    assert_eq!(out.step_logits.len(), max_new);
    let vocab = w.config.vocab_size;
    assert!(out.step_logits.iter().all(|r| r.len() == vocab));
    assert!(out.step_logits.iter().all(|r| r.iter().all(|v| v.is_finite())));
}

#[test]
fn decode_generation_is_deterministic() {
    let w = load_weights();
    let (prompt, max_new, _, _) = load_reference();
    let a = generate_greedy_tiny(&w, &prompt, max_new);
    let b = generate_greedy_tiny(&w, &prompt, max_new);
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.step_logits, b.step_logits);
}

#[test]
fn decode_generated_ids_match_hf_greedy() {
    let w = load_weights();
    let (prompt, max_new, ref_ids, _) = load_reference();
    let out = generate_greedy_tiny(&w, &prompt, max_new);
    assert_eq!(
        out.tokens, ref_ids,
        "Atenia greedy ids must match HF f64 greedy: {:?} vs {:?}",
        out.tokens, ref_ids
    );
}

#[test]
fn decode_step_logits_match_hf_reference() {
    let w = load_weights();
    let vocab = w.config.vocab_size;
    let (prompt, max_new, _, ref_logits) = load_reference();
    let out = generate_greedy_tiny(&w, &prompt, max_new);

    let got: Vec<f32> = out.step_logits.iter().flatten().copied().collect();
    assert_eq!(got.len(), ref_logits.len());

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (a, b) in got.iter().zip(ref_logits.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = (sum_abs / got.len() as f64) as f32;
    eprintln!("DECODE vs HF: max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e}");
    assert!(
        max_abs < 1e-3,
        "decode per-step logits must match HF within 1e-3: max_abs_diff={max_abs:.3e}"
    );

    // Per-step argmax (next-token prediction equivalence).
    for i in 0..max_new {
        let a = &got[i * vocab..(i + 1) * vocab];
        let b = &ref_logits[i * vocab..(i + 1) * vocab];
        let am = a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let bm = b.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        assert_eq!(am, bm, "argmax mismatch at decode step {i}");
    }
}

#[test]
fn fail_loud_still_active() {
    // Real MoE checkpoints still classify as MoE (the loader fail-loud guard
    // keys off this detection — unchanged by MOE-FULL-7).
    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe);
}
