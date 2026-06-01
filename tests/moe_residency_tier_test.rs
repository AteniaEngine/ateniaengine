//! **MOE-PROD-2** — expert residency tier equivalence.
//!
//! The controlled MoE runtime can hold the routed experts either RAM-f32
//! (default) or NVMe-backed (`ATENIA_MOE_EXPERT_TIER=disk`), so a large real
//! MoE (e.g. Qwen1.5-MoE-A2.7B) loads without keeping all experts as f32 in
//! RAM. The disk tier routes through the certified `ResidentExpertLayer`
//! (bit-identical to `RealMoeLayer::forward_auto`, MOE-FULL-8).
//!
//! This test proves the two tiers produce **bit-for-bit identical** logits and
//! generation on the committed tiny Mixtral fixture — no download, no large
//! fixture. Single test, sequential, so the process-global env var is not raced.

use std::path::PathBuf;

use atenia_engine::moe::runtime::MixtralRuntime;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

const PROMPT: &[u32] = &[22, 25, 29];

#[test]
fn disk_tier_equals_ram_tier_bit_for_bit() {
    let config = fixture_dir().join("mixtral_tiny_config.json");
    let weights = fixture_dir().join("full_mixtral.safetensors");

    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }

    // ---- RAM tier (default) ----
    unsafe {
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
    }
    let ram = MixtralRuntime::load_from_files(&config, &weights).expect("RAM-tier load");
    let ram_logits = ram.forward_logits(PROMPT);
    let ram_gen = ram.generate(PROMPT, 8);
    assert_eq!(ram.residency().len(), 2, "two layers resident");

    // ---- Disk tier (NVMe-backed experts) ----
    unsafe {
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
    }
    let disk = MixtralRuntime::load_from_files(&config, &weights).expect("disk-tier load");
    let disk_logits = disk.forward_logits(PROMPT);
    let disk_gen = disk.generate(PROMPT, 8);
    assert_eq!(disk.residency().len(), 2, "two layers resident (disk)");

    unsafe {
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
    }

    // ---- Bit-for-bit equivalence ----
    assert_eq!(ram_logits.len(), disk_logits.len(), "logit length mismatch");
    let max_abs = ram_logits
        .iter()
        .zip(disk_logits.iter())
        .fold(0.0_f32, |a, (x, y)| a.max((x - y).abs()));
    assert_eq!(
        max_abs, 0.0,
        "disk-tier logits must be bit-identical to RAM-tier (max_abs_diff = {max_abs:.3e})"
    );
    assert_eq!(disk_gen, ram_gen, "disk-tier generation must match RAM-tier");
    // Sanity: the fixture stops at EOS=20 after one generated token.
    assert_eq!(ram_gen, vec![17, 20], "fixture greedy path");
}
