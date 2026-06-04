//! **MLA-1 (C1+C2)** — DeepSeek-V2-Lite certification by decomposition (ADR-007
//! C1 per-expert + C2 router), on the **real** weights, reusing the Qwen
//! MOE-CERT-2-ext harness. The expert/router tensor names are identical to
//! Qwen-MoE (`mlp.gate.weight`, `mlp.experts.{e}.{gate,up,down}_proj`), so the
//! only DeepSeek specifics are: **dense-first layer 0 is skipped** (it has no
//! experts), 64 experts / top-6, `norm_topk_prob=false` (the top-k *set* is
//! convention-independent), 2 ungated shared experts (covered by C3, not here).
//!
//! Read-only, test-only: it exercises `RealMoeLayer::assemble` + per-expert
//! `forward` + `top_k_routing` (no runtime/loader/numerics/MLA change). C3 (MLA
//! attention) is covered by the DeepSeek MLA cert + MLA-0, not here.
//!
//! `#[ignore]` + env `DEEPSEEK_V2_LITE_DIR`. Reproduce the reference + run:
//!   python fixtures/moe/generate_deepseek_v2lite_decomposition_reference.py <dir> fixtures/moe
//!   DEEPSEEK_V2_LITE_DIR=models/DeepSeek-V2-Lite cargo test \
//!     --test moe_mla1_deepseek_decomposition_test --release -- \
//!     --ignored mla1_real_deepseek_v2lite_c1_c2 --nocapture

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use atenia_engine::moe::{top_k_routing, MoeLayerConfig, MoeWeightMap, RealMoeLayer};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

const ADR_004_GATE: f32 = 0.5;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

/// f64 router logits `W[n,d]·x` — for the routing margin (the router is f64 on
/// every Atenia policy, so this mirrors Atenia's own routing arithmetic).
fn router_logits_f64(w_router: &[f32], n: usize, d: usize, x: &[f32]) -> Vec<f64> {
    (0..n)
        .map(|r| {
            let base = r * d;
            let mut acc = 0.0_f64;
            for c in 0..d {
                acc += (w_router[base + c] as f64) * (x[c] as f64);
            }
            acc
        })
        .collect()
}

/// Routing margin: gap between the k-th and (k+1)-th largest logit.
fn routing_margin(logits: &[f64], k: usize) -> f64 {
    let mut v = logits.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    v[k - 1] - v[k]
}

/// Shard file(s) holding `model.layers.<layer>.mlp.*` (a layer may span two).
fn mlp_shards_for_layer(model_dir: &Path, layer: usize) -> Vec<PathBuf> {
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let prefix = format!("model.layers.{layer}.mlp.");
    let mut shards: BTreeSet<String> = BTreeSet::new();
    for (name, shard) in index["weight_map"].as_object().unwrap() {
        if name.starts_with(&prefix) {
            shards.insert(shard.as_str().unwrap().to_string());
        }
    }
    shards.into_iter().map(|s| model_dir.join(s)).collect()
}

#[test]
#[ignore = "needs the real ~31 GB DeepSeek-V2-Lite checkpoint via DEEPSEEK_V2_LITE_DIR"]
fn mla1_real_deepseek_v2lite_c1_c2() {
    let model_dir = PathBuf::from(
        std::env::var("DEEPSEEK_V2_LITE_DIR").expect("set DEEPSEEK_V2_LITE_DIR to the checkout"),
    );

    // --- committed f64 reference (one expert at a time, MoE layers only) ---
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("deepseek_v2lite_decomp_ref.json")).unwrap(),
    )
    .unwrap();
    let n_moe = meta["num_moe_layers"].as_u64().unwrap() as usize;
    let n_experts = meta["num_experts"].as_u64().unwrap() as usize;
    let k = meta["experts_per_token"].as_u64().unwrap() as usize;
    let d_model = meta["d_model"].as_u64().unwrap() as usize;
    let d_ff = meta["d_ff"].as_u64().unwrap() as usize;
    let per_layer = meta["per_layer"].as_array().unwrap();

    let refr =
        SafetensorsReader::open(&fixture_dir().join("deepseek_v2lite_decomp_ref.safetensors")).unwrap();
    let input = refr.get("input").unwrap().to_vec_f32().unwrap();
    let expert_outputs = refr.get("expert_outputs").unwrap().to_vec_f32().unwrap();
    assert_eq!(input.len(), d_model);
    assert_eq!(expert_outputs.len(), n_moe * n_experts * d_model);

    let mut global_worst = 0.0_f32;
    let (mut worst_layer, mut worst_expert) = (0usize, 0usize);
    let mut min_margin = f64::INFINITY;
    let mut min_margin_layer = 0usize;
    let mut c1_failures: Vec<(usize, usize, f32)> = Vec::new();
    let mut c2_failures: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut experts_checked = 0usize;

    for mi in 0..n_moe {
        let real_layer = per_layer[mi]["layer"].as_u64().unwrap() as usize;

        // Open whichever shard(s) hold this layer's mlp; build a combined map.
        let shard_paths = mlp_shards_for_layer(&model_dir, real_layer);
        let readers: Vec<SafetensorsReader> =
            shard_paths.iter().map(|p| SafetensorsReader::open(p).unwrap()).collect();
        let prefix = format!("model.layers.{real_layer}.mlp.");
        let names_shapes: Vec<(String, Vec<usize>)> = readers
            .iter()
            .flat_map(|r| r.iter())
            .filter(|e| e.name.starts_with(&prefix))
            .map(|e| (e.name.to_string(), e.shape.to_vec()))
            .collect();
        let map = MoeWeightMap::from_tensors(names_shapes.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve =
            |name: &str| readers.iter().find_map(|r| r.get(name).and_then(|e| e.to_vec_f32().ok()));
        let cfg = MoeLayerConfig::new(n_experts, k, true, d_model, d_ff).unwrap();
        let layer = RealMoeLayer::assemble(&map, real_layer, cfg, &resolve).unwrap();
        assert_eq!(layer.num_experts(), n_experts, "layer {real_layer}: resolved expert count");

        // --- C1: per-expert parity, exhaustive ---
        let mut layer_worst = 0.0_f32;
        for e in 0..n_experts {
            let out = layer.routed.experts[e].forward(&input).unwrap();
            let base = (mi * n_experts + e) * d_model;
            let want = &expert_outputs[base..base + d_model];
            let diff = max_abs_diff(&out, want);
            experts_checked += 1;
            if diff > layer_worst {
                layer_worst = diff;
            }
            if diff > global_worst {
                global_worst = diff;
                worst_layer = real_layer;
                worst_expert = e;
            }
            if !(diff < ADR_004_GATE) {
                c1_failures.push((real_layer, e, diff));
            }
        }

        // --- C2: router top-k SET EQUALITY (hard gate) + margin ---
        let ref_topk: Vec<usize> = per_layer[mi]["topk_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let weights = layer.routed.route(&input).unwrap().weights;
        let atenia_topk = top_k_routing(&weights, k).unwrap().indices;
        let logits = router_logits_f64(&layer.routed.w_router, n_experts, d_model, &input);
        let margin = routing_margin(&logits, k);
        if margin < min_margin {
            min_margin = margin;
            min_margin_layer = real_layer;
        }
        if atenia_topk != ref_topk {
            c2_failures.push((real_layer, atenia_topk.clone(), ref_topk.clone()));
        }

        println!(
            "MLA1 moe {mi:2} (layer {real_layer:2}): C1 worst={layer_worst:.3e} | C2 topk={atenia_topk:?} margin={margin:.6}"
        );
    }

    println!(
        "MLA1-C1C2 SUMMARY: MoE_layers={n_moe} experts_checked={experts_checked} | \
         C1 global worst max_abs_diff={global_worst:.3e} (layer {worst_layer}, expert {worst_expert}) | \
         C2 min routing_margin={min_margin:.6} (layer {min_margin_layer}) | \
         C1 failures={} C2 failures={}",
        c1_failures.len(),
        c2_failures.len()
    );

    assert!(c1_failures.is_empty(), "C1 FAILED (not certifying): {c1_failures:?}");
    assert!(c2_failures.is_empty(), "C2 FAILED — top-k set differs (hard gate; not certifying): {c2_failures:?}");
    assert_eq!(experts_checked, n_moe * n_experts, "must check every expert of every MoE layer");

    println!(
        "MLA1 RESULT: DeepSeek-V2-Lite C1+C2 PASS across all {n_moe} MoE layers ({experts_checked} experts, \
         worst {global_worst:.3e} < {ADR_004_GATE}; all top-k sets match; min margin {min_margin:.4}). \
         C3 (MLA attention) via DeepSeek MLA cert + MLA-0; C4/C5 pending. -> ADR-007 PARTIAL L1."
    );
}
