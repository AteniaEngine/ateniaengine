//! **MOE-PERF-2-VALIDATION** — real before/after measurement of the PERF-2 expert
//! cache (auto-size + bf16-resident), using **existing** instrumentation only
//! (`CacheStats` + `ExpertCache::resident_bytes*`). No runtime change, no new
//! telemetry. `#[ignore]` so CI never runs it (it is a measurement, not a gate);
//! run on demand:
//!   cargo test --release --test moe_perf2_cache_validation -- --ignored --nocapture
//!
//! It builds a Mixtral-style disk-tier MoE layer and replays a multi-token decode,
//! measuring — for several cache capacities × {f32, bf16} — the cache hits /
//! misses (= tier reads = rematerializations) / evictions / resident bytes. This
//! isolates the cache mechanism (the thing PERF-2 changed) on a small, safe model;
//! the 87 GB real-weight forward is mapped analytically (see
//! `docs/HANDOFF_MOE_PERF_2_VALIDATION.md`).

use atenia_engine::moe::data_plane::MoeWeightMap;
use atenia_engine::moe::layer::{MoeLayerConfig, RealMoeLayer};
use atenia_engine::moe::residency::{ExpertCache, ExpertTier, ResidentExpertLayer};

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // bf16-representable (low 16 bits zero) — what the real bf16 tier yields.
            let u = (state >> 11) as u32;
            let v = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f32::from_bits(v.to_bits() & 0xFFFF_0000)
        })
        .collect()
}

/// Mixtral-style real MoE layer: `n` routed experts, no shared, bf16-representable.
fn build(n: usize, d_model: usize, d_ff: usize) -> RealMoeLayer {
    let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
    let mut store: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
    ns.push((router.clone(), vec![n, d_model]));
    store.insert(router, seeded(1, n * d_model));
    for e in 0..n {
        let base = 100 + e as u64;
        for (role, shape, seed) in [
            (format!("w1"), vec![d_ff, d_model], base * 10 + 1),
            (format!("w3"), vec![d_ff, d_model], base * 10 + 2),
            (format!("w2"), vec![d_model, d_ff], base * 10 + 3),
        ] {
            let name = format!("model.layers.0.block_sparse_moe.experts.{e}.{role}.weight");
            ns.push((name.clone(), shape.clone()));
            store.insert(name, seeded(seed, shape.iter().product()));
        }
    }
    let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
    let resolve = move |name: &str| store.get(name).cloned();
    let cfg = MoeLayerConfig::new(n, 2, false, d_model, d_ff).unwrap();
    RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap()
}

#[test]
#[ignore = "measurement (cache stats), not a gate — run with --ignored --nocapture"]
fn perf2_cache_before_after() {
    // Mixtral-like expert count; small dims so it is fast + safe.
    let (n, d_model, d_ff, tokens) = (32usize, 64usize, 128usize, 24usize);
    let real = build(n, d_model, d_ff);
    let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
    let per_expert_f32 = (d_ff * d_model * 2 + d_model * d_ff) * 4;

    eprintln!(
        "=== MOE-PERF-2-VALIDATION: {n} experts, top-2, {tokens}-token decode, per-expert {} KiB f32 ===",
        per_expert_f32 / 1024
    );
    eprintln!(
        "{:<22} | {:>6} {:>6} {:>9} | {:>10} {:>10} {:>9}",
        "config", "hits", "misses", "evictions", "resident", "f32-equiv", "saved"
    );

    // Baseline = cap 1, f32 (the MIXTRAL-CERT-3 setting `ATENIA_MOE_EXPERT_CACHE=1`).
    // After    = the same cap with bf16, and the larger caps bf16 makes affordable.
    for (label, cap, bf16) in [
        ("BEFORE cap=1 f32", 1usize, false),
        ("cap=1 bf16", 1, true),
        ("cap=4 bf16", 4, true),
        ("cap=8 bf16", 8, true),
        ("cap=all(32) bf16", n, true),
        ("cap=all(32) f32", n, false),
    ] {
        let mut cache = ExpertCache::new(cap);
        cache.set_bf16_resident(bf16);
        for t in 0..tokens as u64 {
            let x = seeded(10_000 + t, d_model);
            let _ = res.forward_cached(&mut cache, &x).unwrap();
        }
        let s = cache.stats();
        eprintln!(
            "{label:<22} | {:>6} {:>6} {:>9} | {:>9} B {:>8} B {:>7} B",
            s.hits,
            s.misses,
            s.evictions,
            cache.resident_bytes(),
            cache.resident_bytes_f32_equiv(),
            cache.resident_bytes_saved(),
        );
    }
    eprintln!(
        "NOTE: misses = tier reads = rematerializations. bf16 halves `resident` at every \
         capacity (the RAM win, realized everywhere). A LARGER cache (which bf16 makes \
         affordable) cuts MISSES (the runtime win) — but only when RAM allows cap>1. On a \
         32 GB host the auto-size picks cap=1, so misses are unchanged there (case B)."
    );
}
