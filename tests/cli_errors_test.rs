//! CLI error-layer integration tests (CLI-1).
//!
//! These exercise the real `atenia` binary as a subprocess and
//! assert the CLI-1 contract: human errors carry a stable
//! `error[E-*]` code, the `What happened:` / `How to fix:`
//! sections are present, exit codes follow the unified scheme,
//! and stdout / stderr stay separated (results on stdout, errors
//! on stderr).
//!
//! No real model weights are used — every fixture is a tiny
//! temporary file or directory.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Path to the `atenia` binary, provided by Cargo for integration
/// tests.
fn atenia_bin() -> &'static str {
    env!("CARGO_BIN_EXE_atenia")
}

/// Create a unique temporary directory for one test.
fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atenia_cli_test_{tag}_{}_{}",
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

/// Assert the rendered error carries the mandatory human sections.
fn assert_human_error(stderr: &str, expected_code: &str) {
    assert!(
        stderr.contains(&format!("error[{expected_code}]:")),
        "expected error code {expected_code} in stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("What happened:"),
        "missing 'What happened:' section:\n{stderr}"
    );
    assert!(
        stderr.contains("How to fix:"),
        "missing 'How to fix:' section:\n{stderr}"
    );
}

#[test]
fn load_invalid_yaml_is_invalid_spec_exit_2() {
    let dir = temp_dir("invalid_yaml");
    let file = dir.join("bad.yaml");
    // Malformed YAML — an unterminated flow sequence.
    fs::write(&file, "family: [unterminated\n").unwrap();

    let out = run(&["load", file.to_str().unwrap()]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert_human_error(&out.stderr, "E-ADAPTER-INVALID-SPEC");
    assert!(
        out.stdout.is_empty(),
        "stdout must stay empty on error:\n{}",
        out.stdout
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn load_unknown_family_fails_exit_2() {
    let dir = temp_dir("unknown_family");
    let file = dir.join("falcon.yaml");
    fs::write(&file, "family: falcon\n").unwrap();

    let out = run(&["load", file.to_str().unwrap()]);
    // `atenia load` validates first; an unknown family surfaces as
    // a validation failure (E-ADAPTER-INVALID-SPEC). Either way it
    // is an adapter fault and exits 2.
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("error[E-ADAPTER-INVALID-SPEC]:")
            || out.stderr.contains("error[E-ADAPTER-UNSUPPORTED-ARCHITECTURE]:"),
        "expected an adapter error code:\n{}",
        out.stderr
    );
    assert!(out.stderr.contains("falcon"), "raw reason preserved");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn generate_missing_model_dir_is_io_not_found_exit_2() {
    let missing = std::env::temp_dir().join("atenia_definitely_missing_dir_xyz");
    let _ = fs::remove_dir_all(&missing);

    let out = run(&[
        "generate",
        "--model",
        missing.to_str().unwrap(),
        "--prompt",
        "Hello",
    ]);
    // A non-existent --model path is user input → exit 2.
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert_human_error(&out.stderr, "E-IO-NOT-FOUND");
}

#[test]
fn generate_missing_config_json_is_config_missing_exit_2() {
    let dir = temp_dir("no_config");
    // Empty directory: exists, but has no config.json and no .gguf.
    let out = run(&[
        "generate",
        "--model",
        dir.to_str().unwrap(),
        "--prompt",
        "Hello",
    ]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert_human_error(&out.stderr, "E-CONFIG-MISSING");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn inspect_classic_falcon_is_unsupported_architecture_exit_2() {
    let dir = temp_dir("classic_falcon");
    // Minimal synthetic classic-Falcon config — architecture only.
    fs::write(
        dir.join("config.json"),
        r#"{"architectures":["FalconForCausalLM"],"model_type":"falcon"}"#,
    )
    .unwrap();

    let out = run(&["inspect", dir.to_str().unwrap()]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert_human_error(&out.stderr, "E-ADAPTER-UNSUPPORTED-ARCHITECTURE");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn inspect_missing_dir_fails_exit_2() {
    let missing = std::env::temp_dir().join("atenia_inspect_missing_xyz");
    let _ = fs::remove_dir_all(&missing);

    let out = run(&["inspect", missing.to_str().unwrap()]);
    // Inspecting a non-existent directory fails loud; `inspect`
    // re-words the I/O case for the model-directory context.
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert_human_error(&out.stderr, "E-ADAPTER-INSPECT-FAILED");
}

#[test]
fn load_success_writes_report_to_stdout_exit_0() {
    // The shipped example must load cleanly; the report goes to
    // stdout, and no error code appears anywhere.
    let out = run(&["load", "config/adapters/llama.yaml"]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stdout.contains("Adapter Toolkit v2"),
        "report must be on stdout:\n{}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("error["),
        "stdout must not contain an error line:\n{}",
        out.stdout
    );
    assert!(
        !out.stderr.contains("error["),
        "a successful run must not print an error to stderr:\n{}",
        out.stderr
    );
}

#[test]
fn errors_go_to_stderr_not_stdout() {
    // Cross-check the stdout/stderr contract on a failing command.
    let dir = temp_dir("stderr_check");
    let file = dir.join("bad.yaml");
    fs::write(&file, "family: [unterminated\n").unwrap();

    let out = run(&["load", file.to_str().unwrap()]);
    assert!(
        out.stdout.is_empty(),
        "stdout must be empty on failure:\n{}",
        out.stdout
    );
    assert!(
        out.stderr.contains("error[E-"),
        "the human error must be on stderr:\n{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rendered_error_format_is_consistent() {
    let dir = temp_dir("format");
    let file = dir.join("bad.yaml");
    fs::write(&file, "family: [unterminated\n").unwrap();

    let out = run(&["load", file.to_str().unwrap()]);
    let s = &out.stderr;
    // The full CLI-1 error shape.
    assert!(s.contains("error[E-ADAPTER-INVALID-SPEC]:"), "{s}");
    assert!(s.contains("What happened:"), "{s}");
    assert!(s.contains("How to fix:"), "{s}");
    assert!(s.contains("Technical details:"), "{s}");
    let _ = fs::remove_dir_all(&dir);
}
