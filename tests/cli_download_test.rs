//! Integration tests for `atenia download`.
//!
//! The CLI binary is exercised via `Command::cargo_bin`-style
//! subprocess for the surfaces that only matter at the process
//! boundary (exit codes, stderr framing). For everything else we
//! drive `run_download_with` directly with a `FakeFetcher` so the
//! tests never touch the network and never depend on Hugging Face
//! being reachable from CI.

use std::fs;
use std::process::Command;

use atenia_engine::cli::download::{
    catalog, fetch::test_support::FakeFetcher, run_download_with, DownloadArgs,
};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_atenia"))
}

fn args(alias: &str) -> DownloadArgs {
    DownloadArgs {
        alias: alias.into(),
        dir: None,
        force: false,
        dry_run: false,
        no_suggest: true,
    }
}

#[test]
fn list_subcommand_prints_each_alias() {
    let out = bin().args(["download", "list"]).output().expect("run atenia");
    assert!(out.status.success(), "exit={:?}", out.status.code());
    let stderr = String::from_utf8_lossy(&out.stderr);
    for entry in catalog::CATALOG {
        assert!(
            stderr.contains(entry.alias),
            "alias `{}` missing from list output:\n{stderr}",
            entry.alias
        );
    }
    // The "Available models (N):" header must reflect the actual
    // catalog length so the help text stays in sync.
    assert!(stderr.contains(&format!("Available models ({})", catalog::CATALOG.len())));
}

#[test]
fn unknown_alias_exits_user_input() {
    let out = bin()
        .args(["download", "definitely-not-real"])
        .output()
        .expect("run atenia");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("E-DOWNLOAD-UNKNOWN-MODEL"));
    assert!(stderr.contains("atenia download list"));
}

#[test]
fn dry_run_does_not_create_destination() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");
    let mut a = args("smollm2-135m");
    a.dir = Some(dest.clone());
    a.dry_run = true;

    let fetcher = FakeFetcher::new();
    let code = run_download_with(a, &fetcher);
    assert_eq!(code, 0);
    assert!(!dest.exists(), "dry-run must not create {}", dest.display());
    assert!(fetcher.calls().is_empty(), "dry-run must not fetch");
}

#[test]
fn destination_exists_without_force_exits_user_input() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");
    fs::create_dir_all(&dest).unwrap();

    let mut a = args("smollm2-135m");
    a.dir = Some(dest.clone());

    let fetcher = FakeFetcher::new();
    let code = run_download_with(a, &fetcher);
    assert_eq!(code, 2, "destination-exists should map to UserInput (2)");
    assert!(fetcher.calls().is_empty(), "no fetch should happen");
}

#[test]
fn destination_exists_with_force_proceeds() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");
    fs::create_dir_all(&dest).unwrap();

    let entry = catalog::find("smollm2-135m").unwrap();
    let mut fetcher = FakeFetcher::new();
    for file in entry.files {
        fetcher = fetcher.with_body(entry.file_url(file), file.as_bytes().to_vec());
    }

    let mut a = args("smollm2-135m");
    a.dir = Some(dest.clone());
    a.force = true;

    let code = run_download_with(a, &fetcher);
    assert_eq!(code, 0);
    for file in entry.files {
        assert!(dest.join(file).exists(), "missing {file}");
    }
}

#[test]
fn happy_path_writes_every_catalog_file() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");

    let entry = catalog::find("smollm2-135m").unwrap();
    let mut fetcher = FakeFetcher::new();
    for file in entry.files {
        fetcher = fetcher.with_body(
            entry.file_url(file),
            format!("body-of-{file}").into_bytes(),
        );
    }

    let mut a = args("smollm2-135m");
    a.dir = Some(dest.clone());

    let code = run_download_with(a, &fetcher);
    assert_eq!(code, 0);

    for file in entry.files {
        let body = fs::read(dest.join(file)).unwrap();
        let expected = format!("body-of-{file}");
        assert_eq!(body, expected.as_bytes(), "wrong body for {file}");
        // No `.partial` files should be left behind.
        assert!(
            !dest.join(format!("{file}.partial")).exists(),
            "stale .partial for {file}"
        );
    }
}

#[test]
fn empty_response_triggers_incomplete_error() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");

    let entry = catalog::find("smollm2-135m").unwrap();
    let mut fetcher = FakeFetcher::new();
    // Serve a normal body for the first file, but zero bytes for
    // the second. The orchestrator should bail with E-DOWNLOAD-NETWORK
    // (zero-byte responses are surfaced as a network error before
    // the post-condition check even runs).
    fetcher = fetcher.with_body(entry.file_url(entry.files[0]), b"ok".to_vec());
    fetcher = fetcher.with_body(entry.file_url(entry.files[1]), Vec::new());

    let mut a = args("smollm2-135m");
    a.dir = Some(dest);

    let code = run_download_with(a, &fetcher);
    assert_eq!(code, 1, "empty body should map to System (1)");
}

#[test]
fn network_failure_propagates_as_system_exit() {
    let tmp = tempdir();
    let dest = tmp.join("smollm2-135m");

    let entry = catalog::find("smollm2-135m").unwrap();
    let mut fetcher = FakeFetcher::new();
    // First file fails outright.
    fetcher = fetcher.with_failure(entry.file_url(entry.files[0]), "simulated DNS failure");

    let mut a = args("smollm2-135m");
    a.dir = Some(dest);

    let code = run_download_with(a, &fetcher);
    assert_eq!(code, 1, "network error should map to System (1)");
}

#[test]
fn cli_dry_run_smoke() {
    // Ensures the binary actually parses the `--dry-run` flag and
    // exits 0 without touching the network. We pipe to a temp dir
    // via --dir so even if dry-run misbehaved we would not pollute
    // ./models.
    let tmp = tempdir();
    let out = bin()
        .args([
            "download",
            "smollm2-135m",
            "--dry-run",
            "--dir",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("run atenia");
    assert!(out.status.success(), "exit={:?}", out.status.code());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("dry run"));
    assert!(stderr.contains("No files were written"));
}

// --------------------------------------------------------------- helpers

/// Tiny stand-in for `tempfile::tempdir()` so the test suite does
/// not have to pull a new dev-dependency just for this file. The
/// directory is created under the OS temp dir with a unique-enough
/// suffix; it is intentionally **not** cleaned up automatically —
/// the OS reaps `tmp` on its own schedule and these tests write
/// only a few kilobytes per run.
fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("atenia-dl-{pid}-{nanos}-{n}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}
