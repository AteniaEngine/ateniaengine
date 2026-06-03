//! **MOE-INTEGRATE-2** — end-to-end routing decision on real fixture
//! checkpoints: a dense checkpoint routes to the dense path; a Mixtral /
//! Qwen-MoE checkpoint is detected and gated behind the explicit opt-in.
//!
//! Read-only: it exercises `diagnose_moe` + `decide_route` (no generation, no
//! runtime load). One sequential test so the global `ATENIA_ENABLE_MOE` env is
//! never toggled under a concurrent assertion.

use std::path::PathBuf;

use atenia_engine::moe::family::MoeFamily;
use atenia_engine::moe::{decide_route, diagnose_moe, MoeRoute};

fn manifest_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Copy a fixture safetensors into a fresh temp dir as `model.safetensors`
/// (what `diagnose_moe` reads). Returns the dir.
fn dir_with(fixture_rel: &str, label: &str) -> PathBuf {
    let src = manifest_path(fixture_rel);
    assert!(src.is_file(), "fixture missing: {}", src.display());
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("atenia_mi2_{label}_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(&src, dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn routing_detects_moe_and_gates_on_opt_in() {
    // Ensure the opt-in starts unset for the no-opt-in assertions.
    // SAFETY: single-threaded test; this test owns the env toggle.
    unsafe { std::env::remove_var("ATENIA_ENABLE_MOE") };
    unsafe { std::env::remove_var("ATENIA_EXPERIMENTAL_MOE") };

    // --- Dense checkpoint → Dense (unchanged path). ---
    let dense = dir_with("tests/fixtures/pytorch_bin/tiny_reference.safetensors", "dense");
    let d = diagnose_moe(&dense);
    assert!(!d.is_moe, "dense fixture must not be MoE");
    assert_eq!(decide_route(&d), MoeRoute::Dense);

    // --- Qwen-MoE → detected; no opt-in → NeedsOptIn (fail loud). ---
    let qwen = dir_with("fixtures/moe/qwen_moe_tiny.safetensors", "qwen");
    let dq = diagnose_moe(&qwen);
    assert!(dq.is_moe, "qwen fixture must be MoE");
    assert_eq!(dq.family, Some(MoeFamily::QwenMoe));
    assert!(!dq.opt_in_set);
    assert_eq!(
        decide_route(&dq),
        MoeRoute::NeedsOptIn { family: MoeFamily::QwenMoe }
    );

    // --- Mixtral → detected; no opt-in → NeedsOptIn. ---
    let mix = dir_with("fixtures/moe/mixtral_classic.safetensors", "mixtral");
    let dm = diagnose_moe(&mix);
    assert!(dm.is_moe, "mixtral fixture must be MoE");
    assert_eq!(dm.family, Some(MoeFamily::Mixtral));
    assert_eq!(
        decide_route(&dm),
        MoeRoute::NeedsOptIn { family: MoeFamily::Mixtral }
    );

    // --- With the opt-in set → a runnable family routes to the MoE runtime. ---
    unsafe { std::env::set_var("ATENIA_ENABLE_MOE", "1") };
    let dq2 = diagnose_moe(&qwen);
    assert!(dq2.opt_in_set, "opt-in must be observed");
    assert_eq!(
        decide_route(&dq2),
        MoeRoute::RunMoe { family: MoeFamily::QwenMoe }
    );
    unsafe { std::env::remove_var("ATENIA_ENABLE_MOE") };

    let _ = std::fs::remove_dir_all(&dense);
    let _ = std::fs::remove_dir_all(&qwen);
    let _ = std::fs::remove_dir_all(&mix);
}
