//! CLI diagnostics integration tests (CLI-3).
//!
//! Exercises `atenia doctor`, `atenia diagnose` and `atenia
//! capabilities` as subprocesses: exit codes, stdout/stderr
//! separation, JSON output, and CLI-2 logging integration.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn atenia_bin() -> &'static str {
    env!("CARGO_BIN_EXE_atenia")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atenia_diag_test_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Output {
    let out = Command::new(atenia_bin())
        .args(args)
        .output()
        .expect("spawn atenia");
    Output {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// A minimal but valid HuggingFace model directory: a Llama-family
/// config plus a (presence-only) tokenizer.json.
fn minimal_model_dir(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    fs::write(
        dir.join("config.json"),
        r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama",
            "num_attention_heads":32,"num_key_value_heads":8,"eos_token_id":2}"#,
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    dir
}

#[test]
fn doctor_exits_zero() {
    let out = run(&["doctor"]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stdout.contains("system diagnostics"),
        "stdout:\n{}",
        out.stdout
    );
}

#[test]
fn doctor_json_is_valid_json() {
    let out = run(&["doctor", "--json"]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    let parsed: serde_json::Value =
        serde_json::from_str(&out.stdout).expect("doctor --json must emit valid JSON");
    assert!(parsed.get("checks").is_some(), "JSON missing `checks`");
}

#[test]
fn capabilities_exits_zero_and_lists_families() {
    let out = run(&["capabilities"]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(out.stdout.contains("llama"), "stdout:\n{}", out.stdout);
    // Honest about what is out of scope.
    assert!(
        out.stdout.contains("falcon (classic"),
        "capabilities must list classic Falcon as unsupported:\n{}",
        out.stdout
    );
}

#[test]
fn capabilities_json_is_valid_json() {
    let out = run(&["capabilities", "--json"]);
    assert_eq!(out.code, 0);
    let parsed: serde_json::Value =
        serde_json::from_str(&out.stdout).expect("valid JSON");
    assert!(parsed.get("supported_families").is_some());
    assert!(parsed.get("unsupported_architectures").is_some());
}

#[test]
fn diagnose_missing_path_is_io_not_found() {
    let missing = std::env::temp_dir().join("atenia_diagnose_missing_xyz");
    let _ = fs::remove_dir_all(&missing);

    let out = run(&["diagnose", "--model", missing.to_str().unwrap()]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("error[E-IO-NOT-FOUND]:"),
        "stderr:\n{}",
        out.stderr
    );
}

#[test]
fn diagnose_valid_minimal_model_exits_zero() {
    let dir = minimal_model_dir("ok");
    let out = run(&["diagnose", "--model", dir.to_str().unwrap()]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stdout.contains("ready to generate"),
        "stdout:\n{}",
        out.stdout
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diagnose_unsupported_model_exits_two() {
    // Classic Falcon: exists but is out of scope.
    let dir = temp_dir("falcon");
    fs::write(
        dir.join("config.json"),
        r#"{"architectures":["FalconForCausalLM"],"model_type":"falcon"}"#,
    )
    .unwrap();
    let out = run(&["diagnose", "--model", dir.to_str().unwrap()]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert!(
        out.stdout.contains("[fail]"),
        "diagnosis must show a failing check:\n{}",
        out.stdout
    );
    assert!(out.stdout.contains("NOT ready"), "stdout:\n{}", out.stdout);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diagnose_json_reports_ready_flag() {
    let dir = minimal_model_dir("json");
    let out = run(&["diagnose", "--model", dir.to_str().unwrap(), "--json"]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    let parsed: serde_json::Value =
        serde_json::from_str(&out.stdout).expect("valid JSON");
    assert_eq!(parsed.get("ready").and_then(|v| v.as_bool()), Some(true));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diagnostics_stdout_is_clean_of_logs() {
    let out = run(&["--debug", "doctor"]);
    assert_eq!(out.code, 0);
    // Logs (even at --debug) never reach stdout.
    assert!(!out.stdout.contains("[INFO]"));
    assert!(!out.stdout.contains("[DEBUG]"));
}

#[test]
fn doctor_respects_verbose_and_quiet() {
    let verbose = run(&["--verbose", "doctor"]);
    assert!(
        verbose.stderr.contains("[INFO]"),
        "verbose stderr:\n{}",
        verbose.stderr
    );
    let quiet = run(&["--quiet", "doctor"]);
    assert!(
        !quiet.stderr.contains("[INFO]"),
        "quiet stderr:\n{}",
        quiet.stderr
    );
}
