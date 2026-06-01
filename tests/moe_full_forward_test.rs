//! **MOE-FULL-6** — integration test: the experimental tiny full MoE
//! transformer forward (embeddings → 2 decoder layers w/ real attention +
//! RoPE + causal mask + MoE block → final norm → lm_head → logits) matches
//! an offline HuggingFace `MixtralForCausalLM` f64 reference.
//!
//! Fixture: `fixtures/moe/full_mixtral.{safetensors,json}` (~84 KB), a real
//! tiny Mixtral (MHA, no GQA; vocab 48, hidden 32, 2 layers, 4 experts top-2),
//! generated offline by `fixtures/moe/generate_full_forward_reference.py`. No
//! model downloaded, no HF at test time, no generation, no KV cache.

use std::path::PathBuf;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::full_forward::{
    build_tiny_mixtral_graph, TinyDecoderWeights, TinyMixtralConfig, TinyMixtralWeights,
};
use atenia_engine::moe::{
    detect_moe, MoeLayerConfig, MoeWeightMap, NumericalMetrics, RealMoeLayer,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn read_json() -> serde_json::Value {
    let text = std::fs::read_to_string(fixture_dir().join("full_mixtral.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn usize_f(j: &serde_json::Value, k: &str) -> usize {
    j[k].as_u64().unwrap() as usize
}

/// Build the `TinyMixtralWeights` from the committed full_mixtral fixture.
fn load_weights() -> (TinyMixtralWeights, usize, Vec<f32>, Vec<f32>) {
    let j = read_json();
    let vocab = usize_f(&j, "vocab_size");
    let hidden = usize_f(&j, "hidden_size");
    let n_layers = usize_f(&j, "num_hidden_layers");
    let n_heads = usize_f(&j, "num_attention_heads");
    let head_dim = usize_f(&j, "head_dim");
    let n_experts = usize_f(&j, "num_local_experts");
    let topk = usize_f(&j, "num_experts_per_tok");
    let rope_theta = j["rope_theta"].as_f64().unwrap() as u32;
    let rms_eps = j["rms_norm_eps"].as_f64().unwrap() as f32;
    let seq = usize_f(&j, "seq");
    let input_ids: Vec<f32> =
        j["input_ids"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let hf_logits: Vec<f32> =
        j["hf_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();

    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    let get = |name: &str| {
        reader
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
            .to_vec_f32()
            .unwrap()
    };
    // Expert FFN size from config (the fixture stores experts packed, so
    // map.expert(0,0) is None — use the declared intermediate_size).
    let d_ff = usize_f(&j, "intermediate_size");
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
                moe,
            }
        })
        .collect();

    let w = TinyMixtralWeights {
        config: TinyMixtralConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_hidden_layers: n_layers,
            num_attention_heads: n_heads,
            head_dim,
            rope_theta,
        },
        embed_tokens: get("model.embed_tokens.weight"),
        layers,
        final_norm: get("model.norm.weight"),
        lm_head: get("lm_head.weight"),
        rms_eps,
    };
    (w, seq, input_ids, hf_logits)
}

fn run_forward(w: TinyMixtralWeights, seq: usize, tokens: &[f32]) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let tok = gb.input();
    let logits = build_tiny_mixtral_graph(&mut gb, tok, seq, w);
    gb.output(logits);
    let mut g = gb.build();
    let t = Tensor::new_cpu(vec![1, seq], tokens.to_vec());
    g.execute(vec![t])[0].as_cpu_slice().to_vec()
}

#[test]
fn tiny_full_forward_builds() {
    let (w, seq, ids, _hf) = load_weights();
    assert_eq!(w.layers.len(), 2);
    let out = run_forward(w, seq, &ids);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn tiny_full_forward_logits_shape() {
    let (w, seq, ids, hf) = load_weights();
    let vocab = w.config.vocab_size;
    let out = run_forward(w, seq, &ids);
    assert_eq!(out.len(), seq * vocab);
    assert_eq!(out.len(), hf.len());
}

#[test]
fn tiny_full_forward_is_deterministic() {
    let (w1, seq, ids, _) = load_weights();
    let (w2, _, _, _) = load_weights();
    let a = run_forward(w1, seq, &ids);
    let b = run_forward(w2, seq, &ids);
    assert_eq!(a, b);
}

#[test]
fn tiny_full_forward_matches_hf_reference() {
    let (w, seq, ids, hf) = load_weights();
    let vocab = w.config.vocab_size;
    let got = run_forward(w, seq, &ids);

    // Whole-sequence logit comparison (f32 graph vs f64 HF reference).
    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!(
        "FULL FORWARD vs HF: max_abs_diff={:.3e} mean_abs_diff={:.3e} rmse={:.3e}",
        m.max_abs_diff, m.mean_abs_diff, m.rmse
    );
    assert!(
        m.max_abs_diff < 1e-3,
        "tiny full forward logits must match HF within 1e-3: {m:?}"
    );

    // Per-position argmax must match (next-token prediction equivalence).
    for pos in 0..seq {
        let a = &got[pos * vocab..(pos + 1) * vocab];
        let b = &hf[pos * vocab..(pos + 1) * vocab];
        let am = a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let bm = b.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        assert_eq!(am, bm, "argmax mismatch at position {pos}");
    }
}

#[test]
fn causal_mask_changes_future_visibility() {
    // Changing only the last input token must not alter earlier positions'
    // logits (causality), but must alter the last position's logits.
    let (w1, seq, ids, _) = load_weights();
    let vocab = w1.config.vocab_size;
    let mut ids2 = ids.clone();
    ids2[seq - 1] = ((ids2[seq - 1] as usize + 1) % vocab) as f32;
    let (w2, _, _, _) = load_weights();
    let l1 = run_forward(w1, seq, &ids);
    let l2 = run_forward(w2, seq, &ids2);
    for pos in 0..(seq - 1) {
        for v in 0..vocab {
            let i = pos * vocab + v;
            assert!((l1[i] - l2[i]).abs() < 1e-4, "causal violated at pos {pos}");
        }
    }
    let last = (seq - 1) * vocab;
    assert!((0..vocab).any(|v| (l1[last + v] - l2[last + v]).abs() > 1e-5));
}

#[test]
fn fail_loud_still_active() {
    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe);
}

#[test]
fn dense_models_still_load() {
    let dense = vec![
        ("model.embed_tokens.weight", vec![16usize, 4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
    ];
    let names: Vec<&str> = dense.iter().map(|(n, _)| *n).collect();
    assert!(!detect_moe(names.into_iter()).is_moe);
    assert!(MoeWeightMap::from_tensors(dense.iter().map(|(n, s)| (*n, s.clone()))).is_empty());
}
