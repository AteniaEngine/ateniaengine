//! Integration tests for `atenia quickstart` (CLI-7).
//!
//! The plan-mode tests drive the binary as a subprocess so we
//! actually exercise the clap parser. The `--download` path is
//! driven through `run_quickstart_with` with a `FakeFetcher`, the
//! same pattern CLI-6 uses, so no real network call ever happens
//! in CI.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use atenia_engine::cli::download::{catalog, fetch::test_support::FakeFetcher};
use atenia_engine::cli::quickstart::{run_quickstart_with, QuickstartArgs, DEFAULT_MODEL};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_atenia"))
}

fn args(model: &str) -> QuickstartArgs {
    QuickstartArgs {
        download: false,
        model: model.into(),
        dir: None,
        no_suggest: false,
    }
}

#[test]
fn quickstart_default_exits_0() {
    let out = bin().arg("quickstart").output().expect("run atenia");
    assert!(out.status.success(), "exit={:?}", out.status.code());
}

#[test]
fn quickstart_default_mentions_download_diagnose_chat() {
    let out = bin().arg("quickstart").output().expect("run atenia");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The four-step plan must literally name the three companion
    // subcommands so a copy-paste flow is possible.
    assert!(stderr.contains("atenia doctor"));
    assert!(stderr.contains("atenia download"));
    assert!(stderr.contains("atenia diagnose"));
    assert!(stderr.contains("atenia chat"));
    // The default alias has to be present and labeled "recommended".
    assert!(stderr.contains(DEFAULT_MODEL));
    assert!(stderr.contains("recommended model"));
}

#[test]
fn quickstart_custom_model_mentions_selected_alias() {
    let out = bin()
        .args(["quickstart", "--model", "tinyllama"])
        .output()
        .expect("run atenia");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("tinyllama"));
    assert!(stderr.contains("atenia download tinyllama"));
    // The default smollm2-135m should not leak into the plan when
    // the user picked something else.
    assert!(
        !stderr.contains("atenia download smollm2-135m"),
        "selected alias must not coexist with the default in the plan"
    );
}

#[test]
fn quickstart_unknown_model_exits_2() {
    let out = bin()
        .args(["quickstart", "--model", "definitely-not-real"])
        .output()
        .expect("run atenia");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("E-DOWNLOAD-UNKNOWN-MODEL"));
}

#[test]
fn quickstart_no_suggest_suppresses_next_commands() {
    let out = bin()
        .args(["quickstart", "--no-suggest"])
        .output()
        .expect("run atenia");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The plan itself still prints — only the trailing
    // "Want the shortest path?" / "Tip:" lines should vanish.
    assert!(stderr.contains("Step 1"));
    assert!(!stderr.contains("Want the shortest path?"));
    assert!(!stderr.contains("Tip:"));
}

#[test]
fn quickstart_download_uses_download_module_with_mock_fetcher() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");

    let entry = catalog::find("smollm2-135m").unwrap();
    let mut fetcher = FakeFetcher::new();
    for file in entry.files {
        fetcher = fetcher.with_body(
            entry.file_url(file),
            format!("payload-{file}").into_bytes(),
        );
    }

    let mut a = args("smollm2-135m");
    a.download = true;
    a.dir = Some(dest.clone());

    let code = run_quickstart_with(a, &fetcher);
    assert_eq!(code, 0);

    // The same files the `download` subcommand would have written
    // must be on disk — proves we reused the downloader instead of
    // duplicating its logic.
    for file in entry.files {
        let body = fs::read(dest.join(file)).expect(file);
        let expected = format!("payload-{file}");
        assert_eq!(body, expected.as_bytes(), "wrong body for {file}");
    }
}

#[test]
fn quickstart_download_unknown_alias_does_not_call_fetcher() {
    // Defence-in-depth: alias resolution happens before any HTTP
    // call, so even with `--download` an unknown model must fail
    // early without touching the fetcher.
    let mut a = args("nope");
    a.download = true;
    a.dir = Some(PathBuf::from("/tmp/should-not-be-used"));

    let fetcher = FakeFetcher::new();
    let code = run_quickstart_with(a, &fetcher);
    assert_eq!(code, 2);
    assert!(fetcher.calls().is_empty());
}

#[test]
fn quickstart_download_no_suggest_suppresses_next_footer() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");

    let entry = catalog::find("smollm2-135m").unwrap();
    let mut fetcher = FakeFetcher::new();
    for file in entry.files {
        fetcher = fetcher.with_body(entry.file_url(file), file.as_bytes().to_vec());
    }

    let mut a = args("smollm2-135m");
    a.download = true;
    a.dir = Some(dest);
    a.no_suggest = true;

    let code = run_quickstart_with(a, &fetcher);
    assert_eq!(code, 0);
    // No assertion on output — we mainly want to confirm exit 0
    // and that `--no-suggest` does not break the download path.
}

// --------------------------------------------------------------- helpers

fn tempdir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("atenia-qs-{pid}-{nanos}-{n}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}
