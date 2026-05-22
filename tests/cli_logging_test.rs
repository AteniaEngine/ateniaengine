//! CLI logging integration tests (CLI-2).
//!
//! Exercises the real `atenia` binary as a subprocess and asserts
//! the CLI-2 contract: log levels are controlled by the global
//! flags, `--quiet` silences non-critical logs, `--verbose` shows
//! progress, `--log-file` captures the run with its trace id, and
//! stdout is never contaminated by log lines.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn atenia_bin() -> &'static str {
    env!("CARGO_BIN_EXE_atenia")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atenia_log_test_{tag}_{}_{}",
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

const EXAMPLE: &str = "config/adapters/llama.yaml";

#[test]
fn default_run_prints_no_log_lines() {
    // A successful default run must not print INFO/DEBUG/TRACE log
    // lines — a non-technical user sees a clean result.
    let out = run(&["load", EXAMPLE]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(!out.stderr.contains("[INFO]"), "stderr:\n{}", out.stderr);
    assert!(!out.stderr.contains("[DEBUG]"), "stderr:\n{}", out.stderr);
    assert!(!out.stderr.contains("[TRACE]"), "stderr:\n{}", out.stderr);
}

#[test]
fn verbose_prints_info_logs() {
    let out = run(&["--verbose", "load", EXAMPLE]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("[INFO]"),
        "expected INFO logs under --verbose:\n{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("command start: load"),
        "stderr:\n{}",
        out.stderr
    );
}

#[test]
fn debug_prints_debug_logs() {
    let out = run(&["--debug", "load", EXAMPLE]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("[DEBUG]"),
        "expected DEBUG logs under --debug:\n{}",
        out.stderr
    );
}

#[test]
fn log_level_flag_overrides_verbosity_flags() {
    // Explicit --log-level wins: even with --quiet present, an
    // explicit `debug` level emits DEBUG logs.
    let out = run(&["--quiet", "--log-level", "debug", "load", EXAMPLE]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("[DEBUG]"),
        "explicit --log-level must win over --quiet:\n{}",
        out.stderr
    );
}

#[test]
fn quiet_suppresses_non_critical_logs() {
    let out = run(&["--quiet", "--verbose", "load", EXAMPLE]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    // --quiet wins over --verbose: no INFO logs.
    assert!(
        !out.stderr.contains("[INFO]"),
        "--quiet must suppress INFO logs:\n{}",
        out.stderr
    );
}

#[test]
fn stdout_is_never_contaminated_by_logs() {
    let out = run(&["--debug", "load", EXAMPLE]);
    assert_eq!(out.code, 0);
    // The report is on stdout; log lines never are.
    assert!(out.stdout.contains("Adapter Toolkit v2"));
    assert!(!out.stdout.contains("[INFO]"));
    assert!(!out.stdout.contains("[DEBUG]"));
}

#[test]
fn log_file_is_created_and_records_trace_id() {
    let dir = temp_dir("logfile");
    let log = dir.join("nested").join("run.log");
    let out = run(&[
        "--verbose",
        "--log-file",
        log.to_str().unwrap(),
        "--trace-id",
        "manual-test-123",
        "load",
        EXAMPLE,
    ]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(log.is_file(), "log file must be created (with parent dirs)");
    let contents = fs::read_to_string(&log).expect("read log file");
    assert!(
        contents.contains("manual-test-123"),
        "log file must record the trace id:\n{contents}"
    );
    assert!(
        contents.contains("command start: load"),
        "log file must record the info logs:\n{contents}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn error_footer_shows_trace_id_and_log_file() {
    let dir = temp_dir("errfooter");
    let log = dir.join("error.log");
    let bad = dir.join("bad.yaml");
    fs::write(&bad, "family: [unterminated\n").unwrap();

    let out = run(&[
        "--log-file",
        log.to_str().unwrap(),
        "--trace-id",
        "err-trace-9",
        "load",
        bad.to_str().unwrap(),
    ]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    // The human error keeps its CLI-1 shape...
    assert!(out.stderr.contains("error[E-ADAPTER-INVALID-SPEC]:"));
    // ...and CLI-2 adds the Trace footer.
    assert!(
        out.stderr.contains("Trace:"),
        "error must show a Trace footer:\n{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("err-trace-9"),
        "error footer must show the trace id:\n{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("log file:"),
        "error footer must show the log file:\n{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn human_error_still_reaches_stderr_with_logging_on() {
    let dir = temp_dir("errstderr");
    let bad = dir.join("bad.yaml");
    fs::write(&bad, "family: [unterminated\n").unwrap();

    let out = run(&["--verbose", "load", bad.to_str().unwrap()]);
    assert_eq!(out.code, 2);
    assert!(out.stdout.is_empty(), "stdout:\n{}", out.stdout);
    assert!(out.stderr.contains("error[E-"), "stderr:\n{}", out.stderr);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unknown_log_level_is_a_clear_error() {
    let out = run(&["--log-level", "shout", "load", EXAMPLE]);
    assert_eq!(out.code, 2, "stderr:\n{}", out.stderr);
    assert!(
        out.stderr.contains("error[E-CLI-INVALID-ARGS]:"),
        "stderr:\n{}",
        out.stderr
    );
}

#[test]
fn global_flags_work_after_the_subcommand() {
    // `global = true` lets the flag appear after the subcommand.
    let out = run(&["load", EXAMPLE, "--verbose"]);
    assert_eq!(out.code, 0, "stderr:\n{}", out.stderr);
    assert!(out.stderr.contains("[INFO]"), "stderr:\n{}", out.stderr);
}
