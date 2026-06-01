//! **MOE-FULL-14** — CLI smoke for `atenia moe-generate` (controlled, opt-in
//! MoE generation). Builds a tiny Mixtral model dir from the committed fixtures
//! and drives the real binary. The dense `generate` path is untouched.

use std::path::PathBuf;
use std::process::Command;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn mixtral_dir(label: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("atenia_cli_moe_{}_{}", std::process::id(), label));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixture_dir().join("mixtral_tiny_config.json"), d.join("config.json")).unwrap();
    std::fs::copy(fixture_dir().join("full_mixtral.safetensors"), d.join("model.safetensors"))
        .unwrap();
    d
}

#[test]
fn moe_generate_runs_with_opt_in_flag() {
    let dir = mixtral_dir("run");
    let out = Command::new(env!("CARGO_BIN_EXE_atenia"))
        .args([
            "moe-generate",
            "--model",
            dir.to_str().unwrap(),
            "--prompt-ids",
            "22,25,29",
            "--max-new",
            "8",
            "--experimental-moe",
        ])
        .env_remove("ATENIA_ENABLE_MOE")
        .env_remove("ATENIA_EXPERIMENTAL_MOE")
        .output()
        .expect("run atenia moe-generate");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "expected exit 0, stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(stdout.trim(), "17,20", "controlled MoE generate ids");
}

#[test]
fn moe_generate_refuses_without_opt_in() {
    let dir = mixtral_dir("refuse");
    let out = Command::new(env!("CARGO_BIN_EXE_atenia"))
        .args(["moe-generate", "--model", dir.to_str().unwrap(), "--prompt-ids", "22,25,29"])
        .env_remove("ATENIA_ENABLE_MOE")
        .env_remove("ATENIA_EXPERIMENTAL_MOE")
        .output()
        .expect("run atenia moe-generate");
    assert!(!out.status.success(), "must refuse without the opt-in");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ATENIA_ENABLE_MOE"), "actionable hint missing: {stderr}");
    assert!(stderr.contains("Mixtral"), "family missing: {stderr}");
}
