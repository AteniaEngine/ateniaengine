//! **MOE-PRODUCT-1** — the productive `generate` routing decision, taken
//! **through the declarative resolver bridge** ([`MoeSpecResolver::route`]), on
//! real fixture checkpoints: a dense checkpoint routes to the dense path; a
//! Mixtral / Qwen-MoE checkpoint is detected and gated behind the explicit
//! opt-in; everything else fails loud. This mirrors the MOE-INTEGRATE-2
//! `decide_route` test but exercises the resolver-backed route the CLI now uses,
//! proving they agree on the productively-routable families.
//!
//! Read-only: `diagnose_moe` + `MoeSpecResolver::route` (no generation, no
//! runtime load). One sequential test so the global `ATENIA_ENABLE_MOE` env is
//! never toggled under a concurrent assertion.

use std::path::PathBuf;

use atenia_engine::adapter_toolkit::MoeSpecResolver;
use atenia_engine::moe::family::MoeFamily;
use atenia_engine::moe::{decide_route, diagnose_moe, MoeRoute};

fn manifest_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn dir_with(fixture_rel: &str, label: &str) -> PathBuf {
    let src = manifest_path(fixture_rel);
    assert!(src.is_file(), "fixture missing: {}", src.display());
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("atenia_prod1_{label}_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(&src, dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn productive_route_uses_resolver_and_gates_on_opt_in() {
    // SAFETY: single-threaded test; it owns the env toggle.
    unsafe { std::env::remove_var("ATENIA_ENABLE_MOE") };
    unsafe { std::env::remove_var("ATENIA_EXPERIMENTAL_MOE") };

    // --- Dense → Dense (unchanged path). ---
    let dense = dir_with("tests/fixtures/pytorch_bin/tiny_reference.safetensors", "dense");
    let d = diagnose_moe(&dense);
    assert!(!d.is_moe);
    assert_eq!(MoeSpecResolver::route(&d), MoeRoute::Dense);
    assert_eq!(MoeSpecResolver::route(&d), decide_route(&d)); // agrees with the primitive

    // --- Qwen-MoE → NeedsOptIn without the opt-in. ---
    let qwen = dir_with("fixtures/moe/qwen_moe_tiny.safetensors", "qwen");
    let dq = diagnose_moe(&qwen);
    assert_eq!(dq.family, Some(MoeFamily::QwenMoe));
    assert_eq!(
        MoeSpecResolver::route(&dq),
        MoeRoute::NeedsOptIn { family: MoeFamily::QwenMoe }
    );
    assert_eq!(MoeSpecResolver::route(&dq), decide_route(&dq));

    // --- Mixtral → NeedsOptIn without the opt-in. ---
    let mix = dir_with("fixtures/moe/mixtral_classic.safetensors", "mixtral");
    let dm = diagnose_moe(&mix);
    assert_eq!(dm.family, Some(MoeFamily::Mixtral));
    assert_eq!(
        MoeSpecResolver::route(&dm),
        MoeRoute::NeedsOptIn { family: MoeFamily::Mixtral }
    );

    // --- DeepSeek-V2-Lite shape (MLA, no Q-LoRA, no V3 marker) → NeedsOptIn
    //     without the opt-in (MOE-PRODUCT-2). ---
    let ds = dir_with("fixtures/moe/deepseek_scale.safetensors", "deepseek");
    let dd = diagnose_moe(&ds);
    assert_eq!(dd.family, Some(MoeFamily::DeepSeekMoe));
    assert!(dd.unsupported_variant.is_none(), "V2-Lite shape must not be an unsupported variant");
    assert_eq!(
        MoeSpecResolver::route(&dd),
        MoeRoute::NeedsOptIn { family: MoeFamily::DeepSeekMoe }
    );

    // --- With the opt-in → runnable families route to the MoE runtime. ---
    unsafe { std::env::set_var("ATENIA_ENABLE_MOE", "1") };
    let dq2 = diagnose_moe(&qwen);
    assert!(dq2.opt_in_set);
    assert_eq!(
        MoeSpecResolver::route(&dq2),
        MoeRoute::RunMoe { family: MoeFamily::QwenMoe }
    );
    // DeepSeek-V2-Lite now routes to the MoE runtime with the opt-in.
    assert_eq!(
        MoeSpecResolver::route(&diagnose_moe(&ds)),
        MoeRoute::RunMoe { family: MoeFamily::DeepSeekMoe }
    );
    // The resolver-backed route agrees with the primitive on the runnable set.
    assert_eq!(MoeSpecResolver::route(&dq2), decide_route(&dq2));
    unsafe { std::env::remove_var("ATENIA_ENABLE_MOE") };

    let _ = std::fs::remove_dir_all(&dense);
    let _ = std::fs::remove_dir_all(&qwen);
    let _ = std::fs::remove_dir_all(&mix);
    let _ = std::fs::remove_dir_all(&ds);
}
