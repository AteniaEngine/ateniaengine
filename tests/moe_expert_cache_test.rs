//! **MOE-PROD-3** — expert-cache integration on the disk-tier MoE node.
//!
//! MOE-PROD-2 streams experts to NVMe; without a cache the disk-tier node
//! re-reads the routed top-k experts from NVMe **every token**. MOE-PROD-3
//! threads a per-layer LRU [`ExpertCache`] through the node so a repeated
//! expert is served from RAM. This test proves, on the committed tiny Mixtral
//! fixture, that:
//!
//!   1. the disk-tier node actually exercises the cache (hits accrue on
//!      repeated experts within a multi-token forward), and
//!   2. correctness is unchanged (covered bit-exactly by
//!      `moe_residency_tier_test` and the `cached_forward_matches_uncached`
//!      unit test) — here we additionally check generation still reaches EOS.

use std::path::PathBuf;

use atenia_engine::moe::graph_op::aggregate_resident_cache_stats;
use atenia_engine::moe::runtime::MixtralRuntime;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

#[test]
fn disk_tier_node_uses_expert_cache() {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        // Default capacity (2*top_k) would already cache; pin it explicitly so
        // the test is independent of the default.
        std::env::set_var("ATENIA_MOE_EXPERT_CACHE", "4");
    }

    let rt = MixtralRuntime::load_from_files(
        &fixture_dir().join("mixtral_tiny_config.json"),
        &fixtures_weights(),
    )
    .expect("disk-tier load");

    // Multi-token forward: 7 prompt rows × top-2 of 4 experts per layer ⇒
    // experts repeat across rows ⇒ the cache must register hits.
    let prompt = [22u32, 25, 29, 22, 25, 29, 22];
    let _ = rt.forward_logits(&prompt);
    let out = rt.generate(&[22, 25, 29], 8);
    assert_eq!(out, vec![17, 20], "disk-tier+cache generation still stops at EOS");

    let s = aggregate_resident_cache_stats();
    // The cache is process-global and only grows; after a real disk-tier
    // forward there must be both cold misses and warm hits.
    assert!(s.misses > 0, "expected cold-start tier reads (misses), got {s:?}");
    assert!(s.hits > 0, "expected cache hits on repeated experts, got {s:?}");

    unsafe {
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_MOE_EXPERT_CACHE");
    }
}

fn fixtures_weights() -> PathBuf {
    fixture_dir().join("full_mixtral.safetensors")
}
