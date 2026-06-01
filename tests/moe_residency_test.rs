//! **MOE-FULL-8** — integration test: experimental tiered expert residency.
//!
//! Two things are demonstrated:
//!
//!  1. **Compatibility with MOE-FULL-6/7.** A `ResidentExpertLayer` built from
//!     the *real* committed tiny Mixtral fixture (`full_mixtral.safetensors`,
//!     layer 0) reproduces `RealMoeLayer::forward_auto` bit-for-bit on the
//!     NVMe tier — so the residency mechanism is a drop-in for the certified
//!     MoE block, not a different computation.
//!
//!  2. **Large-scenario residency.** A synthetic 128-expert MoE placed on the
//!     NVMe tier holds ~router-only bytes in host RAM (experts cost zero RAM
//!     until requested), and each forward materialises only the top-k experts
//!     — without downloading any large model.
//!
//! CPU + NVMe only. No model downloaded, no productive runtime, fail-loud
//! intact.

use std::path::PathBuf;

use atenia_engine::moe::residency::{ExpertTier, ResidentExpertLayer};
use atenia_engine::moe::{detect_moe, MoeLayerConfig, MoeWeightMap, RealMoeLayer};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as u32;
        out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

/// Assemble layer 0 of the committed tiny Mixtral fixture as a `RealMoeLayer`.
fn real_layer_from_fixture() -> (RealMoeLayer, usize) {
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("full_mixtral.json")).unwrap(),
    )
    .unwrap();
    let hidden = j["hidden_size"].as_u64().unwrap() as usize;
    let d_ff = j["intermediate_size"].as_u64().unwrap() as usize;
    let n_experts = j["num_local_experts"].as_u64().unwrap() as usize;
    let topk = j["num_experts_per_tok"].as_u64().unwrap() as usize;

    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let cfg = MoeLayerConfig::new(n_experts, topk, false, hidden, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    (layer, hidden)
}

/// Build a synthetic Mixtral-style `RealMoeLayer` with `n` experts.
fn synthetic_real(n: usize, d_model: usize, d_ff: usize) -> RealMoeLayer {
    use std::collections::HashMap;
    let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
    let mut store: HashMap<String, Vec<f32>> = HashMap::new();
    let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
    ns.push((router.clone(), vec![n, d_model]));
    store.insert(router, seeded(1, n * d_model));
    for e in 0..n {
        let base = 100 + e as u64;
        let g = format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight");
        let u = format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight");
        let d = format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight");
        ns.push((g.clone(), vec![d_ff, d_model]));
        ns.push((u.clone(), vec![d_ff, d_model]));
        ns.push((d.clone(), vec![d_model, d_ff]));
        store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
        store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
        store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
    }
    let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
    let resolve = move |name: &str| store.get(name).cloned();
    let cfg = MoeLayerConfig::new(n, 2, false, d_model, d_ff).unwrap();
    RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap()
}

#[test]
fn residency_matches_real_fixture_layer_on_nvme() {
    let (real, hidden) = real_layer_from_fixture();
    let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
    for s in 0..6u64 {
        let x = seeded(7000 + s, hidden);
        let (got, info) = res.forward(&x).unwrap();
        let want = real.forward_auto(&x).unwrap();
        assert_eq!(got, want, "NVMe-tier residency must equal the certified RealMoeLayer (seed {s})");
        assert_eq!(info.materialized_experts.len(), real.config.experts_per_token);
    }
}

#[test]
fn large_scenario_residency_keeps_ram_flat() {
    // 128 experts, modest dims — a Mixtral-class expert count without a large
    // download. On NVMe the experts cost zero host RAM until requested.
    let n = 128;
    let d_model = 64;
    let d_ff = 128;
    let real = synthetic_real(n, d_model, d_ff);
    let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();

    let resident = res.resident_ram_bytes();
    let full = res.full_materialization_bytes();
    let per_expert = (d_ff * d_model * 2 + d_model * d_ff) * 4;

    eprintln!(
        "RESIDENCY (128 experts, NVMe): resident_ram={} bytes  full_materialization={} bytes  saving={:.1}x  per_expert={} bytes",
        resident,
        full,
        full as f64 / resident as f64,
        per_expert,
    );

    // Host RAM holds only the router (n * d_model * 4); experts are on NVMe.
    assert_eq!(resident, n * d_model * 4);
    // Full materialisation is dominated by the 128 experts → orders of
    // magnitude larger than the resident (router-only) footprint.
    assert!(full as f64 / resident as f64 > 100.0, "expected >100x residency saving");

    // Each forward materialises exactly top-k (=2) experts, regardless of N.
    let mut seen_sets = std::collections::HashSet::new();
    for s in 0..8u64 {
        let x = seeded(2000 + s, d_model);
        let (out, info) = res.forward(&x).unwrap();
        assert_eq!(info.materialized_experts.len(), 2);
        assert_eq!(info.materialized_bytes, 2 * per_expert);
        assert!(out.iter().all(|v| v.is_finite()));
        seen_sets.insert(info.materialized_experts.clone());
    }
    // Different tokens route to different experts → the residency layer is
    // genuinely selecting, not pinning a fixed pair.
    assert!(seen_sets.len() > 1, "expected routing to vary across tokens");
}

#[test]
fn large_scenario_nvme_matches_ram_tier() {
    // Correctness at scale: the NVMe tier and the RAM tier produce identical
    // output for the same 128-expert layer.
    let real = synthetic_real(128, 32, 64);
    let ram = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Ram).unwrap();
    let disk = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
    for s in 0..4u64 {
        let x = seeded(3000 + s, 32);
        let (a, _) = ram.forward(&x).unwrap();
        let (b, _) = disk.forward(&x).unwrap();
        assert_eq!(a, b, "NVMe and RAM tiers must agree (seed {s})");
    }
}

#[test]
fn fail_loud_still_active() {
    // The loader fail-loud guard is unchanged by MOE-FULL-8.
    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe);
}
