//! CLI interactive-chat integration tests (CLI-4).
//!
//! Drives `atenia chat` as a subprocess with a scripted stdin.
//! The model is a fake (empty) directory: the pipeline is
//! lazy-loaded only on a real message, so every slash command and
//! the EOF path are exercised without a real checkpoint.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn atenia_bin() -> &'static str {
    env!("CARGO_BIN_EXE_atenia")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atenia_chat_test_{tag}_{}_{}",
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

/// Run `atenia chat` with `args`, feeding `stdin_script` on stdin.
fn run_chat(args: &[&str], stdin_script: &str) -> Output {
    let mut child = Command::new(atenia_bin())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn atenia chat");
    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(stdin_script.as_bytes())
        .expect("write stdin");
    // stdin dropped here → EOF delivered to the child.
    let out = child.wait_with_output().expect("wait atenia chat");
    Output {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn chat_starts_and_exits_cleanly() {
    let dir = temp_dir("start");
    let out = run_chat(&["chat", "--model", dir.to_str().unwrap()], "/exit\n");
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_handles_eof_gracefully() {
    // Empty stdin → immediate EOF (the Ctrl+D path).
    let dir = temp_dir("eof");
    let out = run_chat(&["chat", "--model", dir.to_str().unwrap()], "");
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_empty_input_does_not_panic() {
    let dir = temp_dir("empty");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "\n   \n\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(!out.stderr.contains("panic"), "stderr:\n{}", out.stderr);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_reset_clears_history() {
    let dir = temp_dir("reset");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/reset\n/history\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    // After /reset the /history dump (stdout) reports an empty
    // history.
    assert!(
        out.stdout.contains("history is empty"),
        "stdout:\n{}",
        out.stdout
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_stdout_is_clean_for_command_only_session() {
    // A session that only issues slash commands produces nothing
    // on stdout except an explicit /history dump — never logs or
    // banners.
    let dir = temp_dir("clean");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/help\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stdout.is_empty(),
        "stdout must be empty (help goes to stderr):\n{}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("[INFO]") && !out.stdout.contains("[DEBUG]"),
        "no logs on stdout"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_unknown_command_is_reported() {
    let dir = temp_dir("unknown");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/bogus\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("unknown command"),
        "stderr:\n{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_missing_model_is_io_not_found() {
    let missing = std::env::temp_dir().join("atenia_chat_missing_xyz");
    let _ = fs::remove_dir_all(&missing);
    let out = run_chat(&["chat", "--model", missing.to_str().unwrap()], "/exit\n");
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("error[E-IO-NOT-FOUND]:"),
        "stderr:\n{}",
        out.stderr
    );
}

#[test]
fn chat_model_pointing_at_a_file_is_invalid_args() {
    let dir = temp_dir("file");
    let file = dir.join("adapter.yaml");
    fs::write(&file, "family: llama\n").unwrap();
    let out = run_chat(&["chat", "--model", file.to_str().unwrap()], "/exit\n");
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("error[E-CLI-INVALID-ARGS]:"),
        "stderr:\n{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn chat_verbose_logs_go_to_stderr() {
    let dir = temp_dir("verbose");
    let out = run_chat(
        &["--verbose", "chat", "--model", dir.to_str().unwrap()],
        "/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("[INFO]"),
        "verbose stderr must carry INFO logs:\n{}",
        out.stderr
    );
    assert!(out.stdout.is_empty(), "stdout:\n{}", out.stdout);
    let _ = fs::remove_dir_all(&dir);
}
