//! **MOE-FULL-9** — integration test: GQA support in the experimental MoE full
//! forward. Loads a real tiny **GQA** Mixtral (`num_key_value_heads = 2 !=
//! num_attention_heads = 4`), tiles the K/V projection weights to MHA shape
//! (`moe::gqa::to_mha_kv`), reuses the MOE-FULL-6 MHA graph unchanged, and
//! checks the logits match an offline HuggingFace f64 reference.
//!
//! Fixture: `fixtures/moe/gqa_mixtral.{safetensors,json}`, generated offline by
//! `fixtures/moe/generate_gqa_reference.py`. No model downloaded, no HF at test
//! time, no productive runtime.

use std::path::PathBuf;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::full_forward::{
    build_tiny_mixtral_graph, MoeBlock, TinyDecoderWeights, TinyMixtralConfig, TinyMixtralWeights,
};
use atenia_engine::moe::gqa::{kv_groups, to_mha_kv};
use atenia_engine::moe::{MoeLayerConfig, MoeWeightMap, NumericalMetrics, RealMoeLayer};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn usize_f(j: &serde_json::Value, k: &str) -> usize {
    j[k].as_u64().unwrap() as usize
}

fn load() -> (TinyMixtralWeights, usize, Vec<f32>, Vec<f32>, usize, usize) {
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("gqa_mixtral.json")).unwrap(),
    )
    .unwrap();
    let vocab = usize_f(&j, "vocab_size");
    let hidden = usize_f(&j, "hidden_size");
    let n_layers = usize_f(&j, "num_hidden_layers");
    let n_heads = usize_f(&j, "num_attention_heads");
    let n_kv_heads = usize_f(&j, "num_key_value_heads");
    let head_dim = usize_f(&j, "head_dim");
    let n_experts = usize_f(&j, "num_local_experts");
    let topk = usize_f(&j, "num_experts_per_tok");
    let d_ff = usize_f(&j, "intermediate_size");
    let rope_theta = j["rope_theta"].as_f64().unwrap() as u32;
    let rms_eps = j["rms_norm_eps"].as_f64().unwrap() as f32;
    let seq = usize_f(&j, "seq");
    let input_ids: Vec<f32> =
        j["input_ids"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let hf_logits: Vec<f32> =
        j["hf_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();

    let reader = SafetensorsReader::open(&fixture_dir().join("gqa_mixtral.safetensors")).unwrap();
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
            // GQA: tile K and V projection weights to MHA shape so the
            // certified MHA attention graph can be reused unchanged.
            let w_k_raw = get(&format!("{p}.self_attn.k_proj.weight")); // [n_kv*hd, d_model]
            let w_v_raw = get(&format!("{p}.self_attn.v_proj.weight"));
            let w_k = to_mha_kv(&w_k_raw, n_kv_heads, n_heads, head_dim, hidden).unwrap();
            let w_v = to_mha_kv(&w_v_raw, n_kv_heads, n_heads, head_dim, hidden).unwrap();
            TinyDecoderWeights {
                input_ln: get(&format!("{p}.input_layernorm.weight")),
                w_q: get(&format!("{p}.self_attn.q_proj.weight")),
                w_k,
                w_v,
                w_o: get(&format!("{p}.self_attn.o_proj.weight")),
                post_ln: get(&format!("{p}.post_attention_layernorm.weight")),
                attn_bias: None,
                moe: MoeBlock::Owned(moe),
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
    (w, seq, input_ids, hf_logits, n_heads, n_kv_heads)
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
fn fixture_is_genuinely_gqa() {
    let (_, _, _, _, n_heads, n_kv_heads) = load();
    assert_ne!(n_heads, n_kv_heads, "fixture must be GQA (n_heads != n_kv_heads)");
    assert_eq!(kv_groups(n_heads, n_kv_heads).unwrap(), 2);
}

#[test]
fn gqa_full_forward_matches_hf_reference() {
    let (w, seq, ids, hf, _, _) = load();
    let vocab = w.config.vocab_size;
    let got = run_forward(w, seq, &ids);
    assert_eq!(got.len(), hf.len());

    let m = NumericalMetrics::compute(&got, &hf).unwrap();
    eprintln!(
        "GQA FULL FORWARD vs HF: max_abs_diff={:.3e} mean_abs_diff={:.3e} rmse={:.3e}",
        m.max_abs_diff, m.mean_abs_diff, m.rmse
    );
    assert!(
        m.max_abs_diff < 1e-3,
        "GQA full forward logits must match HF within 1e-3: {m:?}"
    );
    for pos in 0..seq {
        let a = &got[pos * vocab..(pos + 1) * vocab];
        let b = &hf[pos * vocab..(pos + 1) * vocab];
        let am = a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let bm = b.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        assert_eq!(am, bm, "GQA argmax mismatch at position {pos}");
    }
}

#[test]
fn gqa_forward_is_deterministic() {
    let (w1, seq, ids, _, _, _) = load();
    let (w2, _, _, _, _, _) = load();
    let a = run_forward(w1, seq, &ids);
    let b = run_forward(w2, seq, &ids);
    assert_eq!(a, b);
}
