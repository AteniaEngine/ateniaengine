//! **MIXTRAL-CERT-1 (C1+C2)** — Mixtral-8x7B-v0.1 certification by decomposition
//! (ADR-007 C1 per-expert + C2 router), on the **real** weights, reusing the Qwen
//! MOE-CERT-2-ext / DeepSeek MLA-1 harness. Mixtral specifics: router/experts live
//! under `block_sparse_moe` (classic `experts.{e}.{w1,w3,w2}`), **all 32 layers are
//! MoE** (no dense-first), **8 experts / top-2**, **no shared expert**, renormalised
//! top-k (the top-k *set* is convention-independent).
//!
//! Read-only, test-only: exercises `RealMoeLayer::assemble` + per-expert `forward` +
//! `top_k_routing` (no runtime/loader/numerics change). C3 (attention) and C4/C5 are
//! not here.
//!
//! `#[ignore]` + env `MIXTRAL_DIR`. Reproduce the reference + run:
//!   python fixtures/moe/generate_mixtral_decomposition_reference.py models/Mixtral-8x7B-v0.1 fixtures/moe
//!   MIXTRAL_DIR=models/Mixtral-8x7B-v0.1 cargo test \
//!     --test moe_mixtral_decomposition_test --release -- \
//!     --ignored mixtral_real_c1_c2 --nocapture

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

fn routing_margin(logits: &[f64], k: usize) -> f64 {
    let mut v = logits.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    v[k - 1] - v[k]
}

/// Shard file(s) holding `model.layers.<layer>.block_sparse_moe.*` (may span two).
fn moe_shards_for_layer(model_dir: &Path, layer: usize) -> Vec<PathBuf> {
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let prefix = format!("model.layers.{layer}.block_sparse_moe.");
    let mut shards: BTreeSet<String> = BTreeSet::new();
    for (name, shard) in index["weight_map"].as_object().unwrap() {
        if name.starts_with(&prefix) {
            shards.insert(shard.as_str().unwrap().to_string());
        }
    }
    shards.into_iter().map(|s| model_dir.join(s)).collect()
}

#[test]
#[ignore = "needs the real ~87 GB Mixtral-8x7B-v0.1 checkpoint via MIXTRAL_DIR"]
fn mixtral_real_c1_c2() {
    let model_dir =
        PathBuf::from(std::env::var("MIXTRAL_DIR").expect("set MIXTRAL_DIR to the checkout"));

    // --- committed f64 reference (one expert at a time, all layers) ---
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("mixtral_decomp_ref.json")).unwrap(),
    )
    .unwrap();
    let n_moe = meta["num_moe_layers"].as_u64().unwrap() as usize;
    let n_experts = meta["num_experts"].as_u64().unwrap() as usize;
    let k = meta["experts_per_token"].as_u64().unwrap() as usize;
    let d_model = meta["d_model"].as_u64().unwrap() as usize;
    let d_ff = meta["d_ff"].as_u64().unwrap() as usize;
    let per_layer = meta["per_layer"].as_array().unwrap();

    let refr = SafetensorsReader::open(&fixture_dir().join("mixtral_decomp_ref.safetensors")).unwrap();
    let input = refr.get("input").unwrap().to_vec_f32().unwrap();
    let expert_outputs = refr.get("expert_outputs").unwrap().to_vec_f32().unwrap();
    assert_eq!(input.len(), d_model);
    assert_eq!(expert_outputs.len(), n_moe * n_experts * d_model);

    // **Resumable checkpoints** — this run reads ~87 GB from an HDD (~1.5 h) and the
    // environment reaps long background processes (~60-70 min). Each processed layer
    // is written to a checkpoint file immediately; a re-run skips done layers and
    // continues, so the cert completes across windows without losing work or
    // re-reading already-validated layers. Dir override via MIXTRAL_C1C2_CKPT.
    let ckpt_dir = std::env::var("MIXTRAL_C1C2_CKPT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mixtral_c1c2_ckpt"));
    std::fs::create_dir_all(&ckpt_dir).unwrap();
    let ckpt_path = |layer: usize| ckpt_dir.join(format!("layer_{layer:02}.txt"));

    for mi in 0..n_moe {
        let real_layer = per_layer[mi]["layer"].as_u64().unwrap() as usize;
        if ckpt_path(real_layer).exists() {
            println!("MIXTRAL layer {real_layer:2}: SKIP (checkpoint exists)");
            continue;
        }

        let shard_paths = moe_shards_for_layer(&model_dir, real_layer);
        let readers: Vec<SafetensorsReader> =
            shard_paths.iter().map(|p| SafetensorsReader::open(p).unwrap()).collect();
        let prefix = format!("model.layers.{real_layer}.block_sparse_moe.");
        let names_shapes: Vec<(String, Vec<usize>)> = readers
            .iter()
            .flat_map(|r| r.iter())
            .filter(|e| e.name.starts_with(&prefix))
            .map(|e| (e.name.to_string(), e.shape.to_vec()))
            .collect();
        let map = MoeWeightMap::from_tensors(names_shapes.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve =
            |name: &str| readers.iter().find_map(|r| r.get(name).and_then(|e| e.to_vec_f32().ok()));
        // Mixtral: no shared expert.
        let cfg = MoeLayerConfig::new(n_experts, k, false, d_model, d_ff).unwrap();
        let layer = RealMoeLayer::assemble(&map, real_layer, cfg, &resolve).unwrap();
        assert_eq!(layer.num_experts(), n_experts, "layer {real_layer}: resolved expert count");

        // --- C1: per-expert parity, exhaustive ---
        let mut layer_worst = 0.0_f32;
        let mut layer_worst_expert = 0usize;
        let mut layer_c1_fail = 0usize;
        for e in 0..n_experts {
            let out = layer.routed.experts[e].forward(&input).unwrap();
            let base = (mi * n_experts + e) * d_model;
            let want = &expert_outputs[base..base + d_model];
            let diff = max_abs_diff(&out, want);
            if diff > layer_worst {
                layer_worst = diff;
                layer_worst_expert = e;
            }
            if !(diff < ADR_004_GATE) {
                layer_c1_fail += 1;
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
        let mut a_sorted = atenia_topk.clone();
        a_sorted.sort_unstable();
        let c2_match = a_sorted == ref_topk;

        // Checkpoint this layer atomically (worst_diff worst_expert c1_fail margin c2_match).
        let line = format!("{layer_worst:.9e} {layer_worst_expert} {layer_c1_fail} {margin:.12e} {}",
            if c2_match { 1 } else { 0 });
        let tmp = ckpt_path(real_layer).with_extension("tmp");
        std::fs::write(&tmp, &line).unwrap();
        std::fs::rename(&tmp, ckpt_path(real_layer)).unwrap();

        println!(
            "MIXTRAL layer {real_layer:2}: C1 worst={layer_worst:.3e} (exp {layer_worst_expert}) | \
             C2 topk={atenia_topk:?} match={c2_match} margin={margin:.6} [saved]"
        );
    }

    // --- Aggregate from per-layer checkpoints (resumable) ---
    let mut global_worst = 0.0_f32;
    let (mut worst_layer, mut worst_expert) = (0usize, 0usize);
    let mut min_margin = f64::INFINITY;
    let mut min_margin_layer = 0usize;
    let mut c1_fail_total = 0usize;
    let mut c2_failures: Vec<usize> = Vec::new();
    let mut done = 0usize;
    for mi in 0..n_moe {
        let real_layer = per_layer[mi]["layer"].as_u64().unwrap() as usize;
        let p = ckpt_path(real_layer);
        if !p.exists() {
            continue;
        }
        done += 1;
        let s = std::fs::read_to_string(&p).unwrap();
        let f: Vec<&str> = s.split_whitespace().collect();
        let worst: f32 = f[0].parse().unwrap();
        let wexp: usize = f[1].parse().unwrap();
        let c1f: usize = f[2].parse().unwrap();
        let margin: f64 = f[3].parse().unwrap();
        let c2m: u8 = f[4].parse().unwrap();
        if worst > global_worst {
            global_worst = worst;
            worst_layer = real_layer;
            worst_expert = wexp;
        }
        if margin < min_margin {
            min_margin = margin;
            min_margin_layer = real_layer;
        }
        c1_fail_total += c1f;
        if c2m == 0 {
            c2_failures.push(real_layer);
        }
    }

    if done < n_moe {
        println!(
            "MIXTRAL-C1C2 INCOMPLETE: {done}/{n_moe} layers checkpointed (re-run to continue; \
             checkpoints in {ckpt_dir:?}). Partial: C1 worst={global_worst:.3e} C1_fail={c1_fail_total} \
             C2_fail={c2_failures:?} min_margin={min_margin:.6}"
        );
        return;
    }

    let experts_checked = n_moe * n_experts;
    println!(
        "MIXTRAL-C1C2 SUMMARY: layers={n_moe} experts_checked={experts_checked} | \
         C1 global worst max_abs_diff={global_worst:.3e} (layer {worst_layer}, expert {worst_expert}) | \
         C2 min routing_margin={min_margin:.6} (layer {min_margin_layer}) | \
         C1 failures={c1_fail_total} C2 failures={}",
        c2_failures.len()
    );

    assert_eq!(c1_fail_total, 0, "C1 FAILED (not certifying): {c1_fail_total} experts >= {ADR_004_GATE}");
    assert!(
        c2_failures.is_empty(),
        "C2 FAILED — top-k set differs (hard gate; not certifying) at layers: {c2_failures:?}"
    );

    println!(
        "MIXTRAL RESULT: Mixtral-8x7B-v0.1 C1+C2 PASS across all {n_moe} layers ({experts_checked} experts, \
         worst {global_worst:.3e} < {ADR_004_GATE}; all top-k sets match; min margin {min_margin:.4}). \
         C3 (attention) + C4/C5 pending. -> ADR-007 toward L1."
    );
}
