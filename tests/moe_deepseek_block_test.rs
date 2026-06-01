//! **MOE-FULL-11** — integration test: the DeepSeek-MoE **MoE block** is
//! numerically certified against HuggingFace f64. DeepSeek-V2/V3 use MLA
//! attention (out of scope for the experimental runtime), so this certifies the
//! family-distinguishing component — the MoE block (router + packed routed
//! experts + shared expert) — not an end-to-end forward.
//!
//! The HF reference is generated with simple routing (greedy, n_group=1,
//! routed_scaling_factor=1.0, softmax, norm_topk_prob=True) so the block reduces
//! to the certified top-k softmax + renormalise + ungated-shared convention,
//! which `RealMoeLayer::forward_auto` (Atenia convention) reproduces.
//!
//! Fixtures: `fixtures/moe/deepseek_block.{safetensors,json}`, generated offline
//! by `fixtures/moe/generate_deepseek_block_reference.py`. No model downloaded.

use std::path::PathBuf;

use atenia_engine::moe::{
    classify_family, MoeFamily, MoeLayerConfig, MoeWeightMap, RealMoeLayer,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn sidecar() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("deepseek_block.json")).unwrap(),
    )
    .unwrap()
}

#[test]
fn deepseek_block_classifies_as_deepseek_family() {
    // DeepSeek is recognised by its plural `shared_experts` marker (MOE-FULL-12).
    let reader = SafetensorsReader::open(&fixture_dir().join("deepseek_block.safetensors")).unwrap();
    let names: Vec<String> = reader.iter().map(|e| e.name.to_string()).collect();
    assert_eq!(classify_family(names.iter().map(|s| s.as_str())), Some(MoeFamily::DeepSeekMoe));
}

#[test]
fn deepseek_moe_block_matches_hf_reference() {
    let j = sidecar();
    let hidden = j["hidden"].as_u64().unwrap() as usize;
    let d_ff = j["moe_intermediate_size"].as_u64().unwrap() as usize;
    let n_experts = j["n_routed_experts"].as_u64().unwrap() as usize;
    let topk = j["num_experts_per_tok"].as_u64().unwrap() as usize;
    let probe: Vec<f32> =
        j["probe"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let hf_out: Vec<f32> =
        j["mlp_output"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();

    let reader = SafetensorsReader::open(&fixture_dir().join("deepseek_block.safetensors")).unwrap();
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());

    // has_shared_expert = true (DeepSeek shared_experts).
    let cfg = MoeLayerConfig::new(n_experts, topk, true, hidden, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve)
        .expect("DeepSeek MoE block must assemble (packed routed + shared expert)");
    assert!(layer.has_shared_expert());

    let got = layer.forward_auto(&probe).expect("forward_auto");
    assert_eq!(got.len(), hidden);

    let max_abs = got.iter().zip(hf_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let mean_abs = got.iter().zip(hf_out.iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>()
        / got.len() as f64;
    eprintln!("DEEPSEEK MoE BLOCK vs HF: max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e}");
    assert!(
        max_abs < 1e-3,
        "DeepSeek MoE block must match HF within 1e-3: max_abs_diff={max_abs:.3e}"
    );
}
