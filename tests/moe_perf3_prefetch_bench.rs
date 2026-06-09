//! **MOE-PERF-3** — measurement of the parallel expert prefetch, using existing
//! instrumentation only (`CacheStats`). `#[ignore]` (timing, not a gate):
//!   cargo test --release --test moe_perf3_prefetch_bench -- --ignored --nocapture
//!
//! Compares prefetch OFF vs ON on a disk-tier MoE layer with a wide top-k (more
//! reads to overlap). Reports misses (unchanged), the SUM of per-expert read time
//! (`resolve_nanos`) vs the overlapped wall time (`prefetch_wall_nanos`), and the
//! decode wall-clock. Prefetch changes only the order/concurrency of tier reads —
//! the forward output is bit-exact (asserted in `moe::residency` unit tests).

use std::time::Instant;

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
            let u = (state >> 11) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn build(n: usize, top_k: usize, d_model: usize, d_ff: usize) -> RealMoeLayer {
    let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
    let mut store: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
    ns.push((router.clone(), vec![n, d_model]));
    store.insert(router, seeded(1, n * d_model));
    for e in 0..n {
        let base = 100 + e as u64;
        for (role, shape, seed) in [
            ("w1", vec![d_ff, d_model], base * 10 + 1),
            ("w3", vec![d_ff, d_model], base * 10 + 2),
            ("w2", vec![d_model, d_ff], base * 10 + 3),
        ] {
            let name = format!("model.layers.0.block_sparse_moe.experts.{e}.{role}.weight");
            ns.push((name.clone(), shape.clone()));
            store.insert(name, seeded(seed, shape.iter().product()));
        }
    }
    let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
    let resolve = move |name: &str| store.get(name).cloned();
    let cfg = MoeLayerConfig::new(n, top_k, false, d_model, d_ff).unwrap();
    RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap()
}

#[test]
#[ignore = "prefetch measurement (timing), not a gate — run with --ignored --nocapture"]
fn perf3_prefetch_measurement() {
    // Wide top-k (6) so a forward reads 6 experts — overlap is visible. cap=1 (the
    // RAM-constrained case PERF-3 targets), large-ish experts for slower reads.
    let (n, top_k, d_model, d_ff, tokens) = (16usize, 6usize, 256usize, 512usize, 16usize);
    let real = build(n, top_k, d_model, d_ff);
    let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();

    eprintln!(
        "=== MOE-PERF-3: {n} experts, top-{top_k}, cap=1, {tokens}-token decode, per-expert {} KiB ===",
        (d_ff * d_model * 3 * 4) / 1024
    );
    for (label, prefetch) in [("prefetch OFF", false), ("prefetch ON ", true)] {
        let mut cache = ExpertCache::new(1);
        cache.set_prefetch(prefetch);
        let t = Instant::now();
        for tk in 0..tokens as u64 {
            let x = seeded(9000 + tk, d_model);
            let _ = res.forward_cached(&mut cache, &x).unwrap();
        }
        let wall_ms = t.elapsed().as_secs_f64() * 1e3;
        let s = cache.stats();
        let overlap_saved_ms = (s.resolve_nanos.saturating_sub(s.prefetch_wall_nanos)) as f64 / 1e6;
        eprintln!(
            "{label} | decode={wall_ms:8.2} ms | misses={:>4} parallel_prefetches={:>4} | \
             Σ read={:8.2} ms  wall read={:8.2} ms  overlap_saved={overlap_saved_ms:8.2} ms",
            s.misses,
            s.parallel_prefetches,
            s.resolve_nanos as f64 / 1e6,
            s.prefetch_wall_nanos as f64 / 1e6,
        );
    }
    eprintln!(
        "NOTE: misses are identical (prefetch changes read ORDER, not count). With prefetch \
         ON, `wall read` < `Σ read` = the NVMe read latency overlapped under the existing rayon \
         pool. The gain scales with read latency (large real experts) and top-k."
    );
}
