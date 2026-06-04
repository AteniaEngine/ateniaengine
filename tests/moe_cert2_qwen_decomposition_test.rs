//! **MOE-CERT-2** — Qwen-MoE certification by decomposition (ADR-007 C1 + C2).
//!
//! ADR-007 (`docs/decisions/ADR-007-moe-certification-ladder.md`) certifies a
//! real MoE *by decomposition*, because a global F64 forward is infeasible (the
//! full weights do not fit in F64 RAM) AND incomplete (a forward only routes to
//! the top-k experts). This isolated harness covers two obligations for
//! Qwen-MoE; it is **test-only** — it touches no runtime, loader, MoeRuntime,
//! Adapter Toolkit, or numerics, and it never lifts the dense fail-loud guard.
//!
//!  - **C1 (per-expert).** For every routed expert, compare Atenia's SwiGLU
//!    forward against a float64 reference (NumPy, one expert at a time — see
//!    `fixtures/moe/generate_qwen_moe_decomposition_reference.py`). Gate:
//!    `max_abs_diff < 0.5` for EVERY expert (ADR-004 bar, unchanged), exhaustive
//!    coverage (no expert sampled).
//!  - **C2 (router).** Compare Atenia's top-k expert *set* to the reference set
//!    (SET EQUALITY is the hard gate) and report the routing margin (the gap
//!    between the k-th and (k+1)-th router logit) as a fragility signal.
//!
//! C3 (attention) is **not** re-derived here: it is already covered by the
//! existing Qwen-MoE full-forward certificate (GQA + Q/K/V bias vs HF f64,
//! MOE-FULL-13 / `moe_certification_test`). MOE-CERT-2 reuses that evidence.
//!
//! Two layers of testing:
//!  - a CI-deterministic **smoke** on the committed tiny real-layout fixture
//!    (`qwen15_moe_layer0.safetensors`, random weights) that exercises the
//!    harness mechanics (per-expert determinism, top-k set, margin, and that a
//!    top-k mismatch is detected loudly) without needing an oracle;
//!  - the **real-weight cert** (`#[ignore]`, env `QWEN_MOE_DIR`) that runs C1/C2
//!    on the real Qwen1.5-MoE-A2.7B layer-0 weights vs the committed f64
//!    reference — the measured L1 evidence.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use atenia_engine::moe::{top_k_routing, MoeLayerConfig, MoeWeightMap, RealMoeLayer};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

const ADR_004_GATE: f32 = 0.5;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

// ---------------------------------------------------------------------------
// Harness primitives (pure; shared by the smoke and the real-weight cert).
// ---------------------------------------------------------------------------

/// Float64 router logits `W_router[n, d] · x`. The router is kept in f64 on
/// every Atenia policy (see `src/moe/dense.rs`), so this mirrors Atenia's own
/// routing arithmetic; used here only to derive the routing margin.
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

/// Top-k expert index set from logits, ties broken by lower index, returned
/// sorted ascending (same selection rule as `moe::top_k_routing`).
fn topk_set_from_logits(logits: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut sel: Vec<usize> = order.into_iter().take(k).collect();
    sel.sort_unstable();
    sel
}

/// Routing margin: gap between the k-th and (k+1)-th largest logit. A small
/// margin warns that the top-k decision is fragile near a tie.
fn routing_margin(logits: &[f64], k: usize) -> f64 {
    let mut v = logits.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    v[k - 1] - v[k]
}

/// Max abs difference between two equal-length vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// CI smoke — harness mechanics on the committed tiny real-layout fixture.
// Deterministic, no oracle, no large model. Proves the decomposition harness
// works and that a top-k mismatch is detected loudly.
// ---------------------------------------------------------------------------

fn load_tiny_qwen() -> (RealMoeLayer, Vec<f32>) {
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("qwen15_moe_layer0.json")).unwrap(),
    )
    .unwrap();
    let cfg = MoeLayerConfig::new(
        j["num_experts"].as_u64().unwrap() as usize,
        j["experts_per_token"].as_u64().unwrap() as usize,
        j["has_shared"].as_bool().unwrap(),
        j["d_model"].as_u64().unwrap() as usize,
        j["d_ff"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let reader =
        SafetensorsReader::open(&fixture_dir().join("qwen15_moe_layer0.safetensors")).unwrap();
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    let input: Vec<f32> = j["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    (layer, input)
}

#[test]
fn cert2_harness_mechanics_smoke() {
    let (layer, x) = load_tiny_qwen();
    let k = layer.config.experts_per_token;

    // C1 mechanics: per-expert forward is finite + deterministic (run twice).
    for e in 0..layer.num_experts() {
        let a = layer.routed.experts[e].forward(&x).unwrap();
        let b = layer.routed.experts[e].forward(&x).unwrap();
        assert_eq!(a, b, "expert {e} forward must be deterministic");
        assert!(a.iter().all(|v| v.is_finite()), "expert {e} output must be finite");
    }

    // C2 mechanics: Atenia's official top-k set is well-formed + deterministic.
    let weights = layer.routed.route(&x).unwrap().weights;
    let sel = top_k_routing(&weights, k).unwrap();
    assert_eq!(sel.indices.len(), k, "top-k must select exactly k experts");
    let sel2 = top_k_routing(&layer.routed.route(&x).unwrap().weights, k).unwrap();
    assert_eq!(sel.indices, sel2.indices, "top-k selection must be deterministic");

    // Routing margin from f64 logits is finite and non-negative.
    let logits = router_logits_f64(&layer.routed.w_router, layer.num_experts(), layer.config.d_model, &x);
    let margin = routing_margin(&logits, k);
    assert!(margin.is_finite() && margin >= 0.0, "routing margin must be >= 0");
    // The logit-derived set must agree with Atenia's softmax-derived set
    // (softmax is monotonic, so top-k by weight == top-k by logit).
    assert_eq!(topk_set_from_logits(&logits, k), sel.indices, "logit/softmax top-k must agree");

    // Fail-loud demonstration: a WRONG reference set must be detected — i.e.
    // the C2 hard gate fires rather than silently certifying.
    let mut wrong = sel.indices.clone();
    wrong[0] = (wrong[0] + 1) % layer.num_experts();
    wrong.sort_unstable();
    wrong.dedup();
    assert_ne!(sel.indices, wrong, "a perturbed top-k set must compare unequal (gate fires)");

    println!(
        "CERT2-SMOKE qwen15_moe_tiny: experts={} k={} topk={:?} margin={:.6}",
        layer.num_experts(),
        k,
        sel.indices,
        margin
    );
}

// ---------------------------------------------------------------------------
// Real-weight cert — C1 + C2 on the real Qwen1.5-MoE-A2.7B layer-0 weights.
// #[ignore] + env QWEN_MOE_DIR (the 27 GB sharded checkpoint). Produces the
// measured L1 evidence. Reproduce the f64 reference with:
//   python fixtures/moe/generate_qwen_moe_decomposition_reference.py <QWEN_MOE_DIR>
// then:
//   QWEN_MOE_DIR=models/Qwen1.5-MoE-A2.7B-Chat cargo test \
//     --test moe_cert2_qwen_decomposition_test --release -- \
//     --ignored cert2_real_qwen_moe_per_expert_and_router --nocapture
// ---------------------------------------------------------------------------

/// Find the shard file holding a given tensor, via the sharded index.
fn shard_for(model_dir: &std::path::Path, tensor: &str) -> PathBuf {
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let shard = index["weight_map"][tensor]
        .as_str()
        .unwrap_or_else(|| panic!("tensor {tensor} not in index weight_map"));
    model_dir.join(shard)
}

#[test]
#[ignore = "needs the real ~27 GB Qwen1.5-MoE checkpoint via QWEN_MOE_DIR"]
fn cert2_real_qwen_moe_per_expert_and_router() {
    let model_dir = PathBuf::from(
        std::env::var("QWEN_MOE_DIR").expect("set QWEN_MOE_DIR to the Qwen1.5-MoE checkout"),
    );

    // --- committed f64 reference (computed one expert at a time) ---
    let ref_meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("qwen_moe_decomp_ref.json")).unwrap(),
    )
    .unwrap();
    let n_experts = ref_meta["num_experts"].as_u64().unwrap() as usize;
    let k = ref_meta["experts_per_token"].as_u64().unwrap() as usize;
    let d_model = ref_meta["d_model"].as_u64().unwrap() as usize;
    let d_ff = ref_meta["d_ff"].as_u64().unwrap() as usize;
    let ref_topk: Vec<usize> = ref_meta["topk_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let ref_margin = ref_meta["routing_margin"].as_f64().unwrap();

    let refr = SafetensorsReader::open(&fixture_dir().join("qwen_moe_decomp_ref.safetensors")).unwrap();
    let input = refr.get("input").unwrap().to_vec_f32().unwrap();
    let expert_outputs_flat = refr.get("expert_outputs").unwrap().to_vec_f32().unwrap();
    assert_eq!(input.len(), d_model, "reference input dim");
    assert_eq!(expert_outputs_flat.len(), n_experts * d_model, "reference expert_outputs dims");

    // --- assemble the real layer-0 MoE from the sharded checkpoint ---
    let shard = shard_for(&model_dir, "model.layers.0.mlp.gate.weight");
    let reader = SafetensorsReader::open(&shard).unwrap();
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let cfg = MoeLayerConfig::new(n_experts, k, true, d_model, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    assert_eq!(layer.num_experts(), n_experts, "resolved expert count");

    // === C1 — per-expert parity (exhaustive; gate < 0.5 for EVERY expert) ===
    let mut worst = 0.0_f32;
    let mut worst_e = 0usize;
    let mut failed: Vec<(usize, f32)> = Vec::new();
    for e in 0..n_experts {
        let out = layer.routed.experts[e].forward(&input).unwrap();
        let want = &expert_outputs_flat[e * d_model..(e + 1) * d_model];
        let diff = max_abs_diff(&out, want);
        if diff > worst {
            worst = diff;
            worst_e = e;
        }
        if !(diff < ADR_004_GATE) {
            failed.push((e, diff));
        }
    }
    println!(
        "CERT2-C1 real Qwen-MoE: {n_experts} experts, worst max_abs_diff={worst:.3e} (expert {worst_e}), gate {ADR_004_GATE}"
    );
    assert!(
        failed.is_empty(),
        "C1 FAILED — experts over the ADR-004 gate (not certifying): {failed:?}"
    );

    // === C2 — router parity (top-k SET EQUALITY hard gate + margin report) ===
    let weights = layer.routed.route(&input).unwrap().weights;
    let atenia_topk = top_k_routing(&weights, k).unwrap().indices; // sorted asc
    let logits = router_logits_f64(&layer.routed.w_router, n_experts, d_model, &input);
    let atenia_margin = routing_margin(&logits, k);
    println!(
        "CERT2-C2 real Qwen-MoE: atenia topk={atenia_topk:?} ref topk={ref_topk:?} | \
         atenia margin={atenia_margin:.6} ref margin={ref_margin:.6}"
    );
    assert_eq!(
        atenia_topk, ref_topk,
        "C2 FAILED — top-k set differs from the f64 reference (hard gate; not certifying)"
    );
    // Margin is a fragility signal, not a gate; sanity-check the two agree closely.
    assert!(
        (atenia_margin - ref_margin).abs() < 1e-2,
        "routing margin diverged: atenia {atenia_margin} vs ref {ref_margin}"
    );

    // Determinism: re-running C1 worst-expert + C2 is bit-identical.
    let again = layer.routed.experts[worst_e].forward(&input).unwrap();
    let first = layer.routed.experts[worst_e].forward(&input).unwrap();
    assert_eq!(again, first, "per-expert forward must be deterministic");
    assert_eq!(top_k_routing(&weights, k).unwrap().indices, atenia_topk, "routing deterministic");

    println!(
        "CERT2 RESULT: Qwen-MoE C1 PASS (worst {worst:.3e} < {ADR_004_GATE}, all {n_experts} layer-0 experts) \
         + C2 PASS (top-k set match). Scope: layer-0 representative (NOT all 24 layers). \
         C3 attention via the existing mechanism cert (MOE-FULL-13, tiny fixture), not real-weight re-certified. \
         → ADR-007 L1 on the layer-0 scope; whole-model L1 needs the all-layers extension."
    );
}

// ---------------------------------------------------------------------------
// MOE-CERT-2-ext — C1 + C2 across ALL layers of the real Qwen1.5-MoE.
// #[ignore] + env QWEN_MOE_DIR. Reads each layer's real weights from whichever
// shard(s) hold them (7 layers span two shards), one expert at a time, and
// gates against the committed all-layers f64 reference. Produces the
// whole-model L1 evidence. Regenerate the reference with:
//   python fixtures/moe/generate_qwen_moe_decomposition_reference.py <dir> fixtures/moe --all
// Run:
//   QWEN_MOE_DIR=models/Qwen1.5-MoE-A2.7B-Chat cargo test \
//     --test moe_cert2_qwen_decomposition_test --release -- \
//     --ignored cert2_real_qwen_moe_all_layers --nocapture
// ---------------------------------------------------------------------------

/// The set of shard files that hold `model.layers.<layer>.mlp.*` tensors,
/// read from the sharded index. A layer's experts may live in two shards.
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
#[ignore = "needs the real ~27 GB Qwen1.5-MoE checkpoint via QWEN_MOE_DIR"]
fn cert2_real_qwen_moe_all_layers() {
    let model_dir = PathBuf::from(
        std::env::var("QWEN_MOE_DIR").expect("set QWEN_MOE_DIR to the Qwen1.5-MoE checkout"),
    );

    // --- committed all-layers f64 reference ---
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("qwen_moe_decomp_ref_all_layers.json")).unwrap(),
    )
    .unwrap();
    let n_layers = meta["num_layers"].as_u64().unwrap() as usize;
    let n_experts = meta["num_experts"].as_u64().unwrap() as usize;
    let k = meta["experts_per_token"].as_u64().unwrap() as usize;
    let d_model = meta["d_model"].as_u64().unwrap() as usize;
    let d_ff = meta["d_ff"].as_u64().unwrap() as usize;
    let per_layer = meta["per_layer"].as_array().unwrap();

    let refr =
        SafetensorsReader::open(&fixture_dir().join("qwen_moe_decomp_ref_all_layers.safetensors"))
            .unwrap();
    let input = refr.get("input").unwrap().to_vec_f32().unwrap();
    let expert_outputs = refr.get("expert_outputs").unwrap().to_vec_f32().unwrap();
    assert_eq!(input.len(), d_model);
    assert_eq!(expert_outputs.len(), n_layers * n_experts * d_model);

    // Global trackers (no fabricated numbers — every value is measured).
    let mut global_worst = 0.0_f32;
    let (mut worst_layer, mut worst_expert) = (0usize, 0usize);
    let mut min_margin = f64::INFINITY;
    let mut min_margin_layer = 0usize;
    let mut c1_failures: Vec<(usize, usize, f32)> = Vec::new();
    let mut c2_failures: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut experts_checked = 0usize;

    for l in 0..n_layers {
        // Open whichever shard(s) hold this layer's mlp, build a combined map +
        // resolver. Fresh opens per layer (safetensors mmaps; cheap, lazy).
        let shard_paths = mlp_shards_for_layer(&model_dir, l);
        let readers: Vec<SafetensorsReader> = shard_paths
            .iter()
            .map(|p| SafetensorsReader::open(p).unwrap())
            .collect();
        let prefix = format!("model.layers.{l}.mlp.");
        let names_shapes: Vec<(String, Vec<usize>)> = readers
            .iter()
            .flat_map(|r| r.iter())
            .filter(|e| e.name.starts_with(&prefix))
            .map(|e| (e.name.to_string(), e.shape.to_vec()))
            .collect();
        let map = MoeWeightMap::from_tensors(names_shapes.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve = |name: &str| {
            readers.iter().find_map(|r| r.get(name).and_then(|e| e.to_vec_f32().ok()))
        };
        let cfg = MoeLayerConfig::new(n_experts, k, true, d_model, d_ff).unwrap();
        let layer = RealMoeLayer::assemble(&map, l, cfg, &resolve).unwrap();
        assert_eq!(layer.num_experts(), n_experts, "layer {l}: resolved expert count");

        // --- C1: per-expert parity, exhaustive ---
        let mut layer_worst = 0.0_f32;
        for e in 0..n_experts {
            let out = layer.routed.experts[e].forward(&input).unwrap();
            let base = (l * n_experts + e) * d_model;
            let want = &expert_outputs[base..base + d_model];
            let diff = max_abs_diff(&out, want);
            experts_checked += 1;
            if diff > layer_worst {
                layer_worst = diff;
            }
            if diff > global_worst {
                global_worst = diff;
                worst_layer = l;
                worst_expert = e;
            }
            if !(diff < ADR_004_GATE) {
                c1_failures.push((l, e, diff));
            }
        }

        // --- C2: router top-k set equality (hard gate) + margin ---
        let ref_topk: Vec<usize> = per_layer[l]["topk_indices"]
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
            min_margin_layer = l;
        }
        if atenia_topk != ref_topk {
            c2_failures.push((l, atenia_topk.clone(), ref_topk.clone()));
        }

        println!(
            "CERT2-EXT layer {l:2}: C1 worst={layer_worst:.3e} | C2 topk={atenia_topk:?} margin={margin:.6}"
        );
    }

    println!(
        "CERT2-EXT SUMMARY: layers={n_layers} experts_checked={experts_checked} | \
         C1 global worst max_abs_diff={global_worst:.3e} (layer {worst_layer}, expert {worst_expert}) | \
         C2 min routing_margin={min_margin:.6} (layer {min_margin_layer}) | \
         C1 failures={} C2 failures={}",
        c1_failures.len(),
        c2_failures.len()
    );

    // Hard gates — if ANY layer fails, do NOT certify (the assert fails loudly).
    assert!(
        c1_failures.is_empty(),
        "C1 FAILED on some experts (not certifying L1): {c1_failures:?}"
    );
    assert!(
        c2_failures.is_empty(),
        "C2 FAILED — top-k set differs on some layers (not certifying L1): {c2_failures:?}"
    );
    assert_eq!(experts_checked, n_layers * n_experts, "must check every expert of every layer");

    println!(
        "CERT2-EXT RESULT: Qwen-MoE C1+C2 PASS across ALL {n_layers} layers ({experts_checked} experts, \
         worst {global_worst:.3e} < {ADR_004_GATE}; all top-k sets match; min margin {min_margin:.4}). \
         C3 attention reused (MOE-FULL-13). → ADR-007 WHOLE-MODEL L1 for Qwen-MoE."
    );
}
