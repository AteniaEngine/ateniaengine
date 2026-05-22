//! CLI UX integration tests (CLI-5).
//!
//! Asserts the user-experience refinements: the `You> ` chat
//! prompt, the `/clear` alias, the `/help` command list, clean
//! stdout, verbose logging on stderr, and well-formed output.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn atenia_bin() -> &'static str {
    env!("CARGO_BIN_EXE_atenia")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atenia_ux_test_{tag}_{}_{}",
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
    let out = child.wait_with_output().expect("wait atenia chat");
    Output {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn chat_prompt_is_you() {
    let dir = temp_dir("prompt");
    let out = run_chat(&["chat", "--model", dir.to_str().unwrap()], "/exit\n");
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("You> "),
        "expected the `You> ` turn prompt:\n{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn clear_is_an_alias_for_reset() {
    let dir = temp_dir("clear");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/clear\n/history\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    // `/clear` must be recognised (not "unknown command") and must
    // clear the history.
    assert!(
        !out.stderr.contains("unknown command"),
        "`/clear` must be a known command:\n{}",
        out.stderr
    );
    assert!(
        out.stdout.contains("history is empty"),
        "stdout:\n{}",
        out.stdout
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn help_lists_all_commands() {
    let dir = temp_dir("help");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/help\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    for cmd in ["/exit", "/reset", "/clear", "/history", "/help"] {
        assert!(
            out.stderr.contains(cmd),
            "/help must list {cmd}:\n{}",
            out.stderr
        );
    }
    // Help is chrome → stderr, never stdout.
    assert!(out.stdout.is_empty(), "stdout:\n{}", out.stdout);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn generate_error_keeps_stdout_clean() {
    let missing = std::env::temp_dir().join("atenia_ux_gen_missing_xyz");
    let _ = fs::remove_dir_all(&missing);
    let out = run(&[
        "generate",
        "--model",
        missing.to_str().unwrap(),
        "--prompt",
        "Hello",
    ]);
    assert_ne!(out.code, 0);
    assert!(
        out.stdout.is_empty(),
        "generate must keep stdout clean on error:\n{}",
        out.stdout
    );
    assert!(out.stderr.contains("error[E-"), "stderr:\n{}", out.stderr);
}

#[test]
fn verbose_chat_logs_to_stderr() {
    let dir = temp_dir("verbose");
    let out = run_chat(
        &["--verbose", "chat", "--model", dir.to_str().unwrap()],
        "/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(out.stderr.contains("[INFO]"), "stderr:\n{}", out.stderr);
    assert!(out.stdout.is_empty(), "stdout:\n{}", out.stdout);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn command_only_session_shows_no_thinking_indicator() {
    // No real message → no generation → the "Thinking ..."
    // indicator must not appear.
    let dir = temp_dir("nothink");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/help\n/history\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        !out.stderr.contains("Thinking"),
        "no generation → no Thinking indicator:\n{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn empty_history_is_a_formatted_message() {
    let dir = temp_dir("hist");
    let out = run_chat(
        &["chat", "--model", dir.to_str().unwrap()],
        "/history\n/exit\n",
    );
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    // A friendly message, not a crude empty dump.
    assert_eq!(out.stdout.trim(), "(history is empty)");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn capabilities_output_is_clean_and_complete() {
    // A non-interactive command: clean stdout, no logs, exit 0.
    let out = run(&["capabilities"]);
    assert_eq!(out.code, 0);
    assert!(out.stdout.contains("Atenia Engine — capabilities"));
    assert!(!out.stdout.contains("[INFO]"));
}
