//! **CLI-6** — `atenia download`: curated model downloader.
//!
//! Reduces "first model" friction. Users type
//! `atenia download <alias>` and get a working checkpoint under
//! `./models/<alias>/` without having to navigate Hugging Face,
//! install `huggingface-cli`, or learn the LFS path conventions.
//!
//! Boundaries (intentional, audited):
//!
//!   - reads a small hardcoded [`catalog`] of known-good public
//!     checkpoints — no arbitrary HF repo support,
//!   - downloads via [`fetch::HttpFetcher`] (production: ureq +
//!     rustls; tests: in-memory fake), one file at a time,
//!     `.partial` then atomic rename,
//!   - emits all progress on **stderr** so stdout stays free for
//!     a future `--json` mode and so piping does not pollute the
//!     downloaded content,
//!   - never touches the runtime core, the graph builders, the
//!     loaders, or the adapters. The post-download "Next:" footer
//!     suggests `atenia diagnose` / `atenia chat`; verifying the
//!     model is left to those commands.

pub mod catalog;
pub mod fetch;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cli::error::CliError;
use crate::cli::exit::CliExit;
use crate::cli::logging;

pub use catalog::{CatalogEntry, ModelFormat};
pub use fetch::{HttpFetcher, UreqFetcher};

/// Arguments collected by `clap` for the `download` subcommand.
///
/// `alias` doubles as the catalog key and as the literal token
/// `list`, which short-circuits to printing the catalog without
/// downloading anything. Keeping it a single positional argument
/// (rather than two subcommands) matches the existing CLI shape
/// — `atenia download foo` is uniform with `atenia chat`,
/// `atenia inspect`, etc.
pub struct DownloadArgs {
    pub alias: String,
    pub dir: Option<PathBuf>,
    pub force: bool,
    pub dry_run: bool,
    pub no_suggest: bool,
}

/// Entry point invoked by `src/bin/atenia.rs`. Returns the process
/// exit code so the bin file's dispatch table stays uniform with
/// the other `run_*` helpers.
pub fn run_download(args: DownloadArgs) -> i32 {
    let fetcher = UreqFetcher::new();
    run_download_with(args, &fetcher)
}

/// Same as [`run_download`], but with an injectable fetcher for
/// tests. Public so the integration tests in `tests/` can drive
/// the orchestration without touching the network.
pub fn run_download_with(args: DownloadArgs, fetcher: &dyn HttpFetcher) -> i32 {
    logging::info("command start: download");
    logging::debug(&format!("alias: {}", args.alias));

    if args.alias == "list" {
        print_list();
        return CliExit::Success.code();
    }

    let entry = match catalog::find(&args.alias) {
        Some(e) => e,
        None => {
            let err = CliError::download_unknown_model(&args.alias);
            eprintln!("{err}");
            return err.exit.code();
        }
    };

    if entry.gated {
        // Defence in depth: the v1 catalog excludes gated models
        // (asserted by `catalog::tests::v1_catalog_excludes_gated_models`),
        // but if a future entry slips through with `gated = true`
        // we refuse to download and point the user at the manual
        // flow instead of trying to fetch HTML 401 pages.
        let err = CliError::download_gated_model(entry);
        eprintln!("{err}");
        return err.exit.code();
    }

    let dest = resolve_destination(&args, entry);
    logging::debug(&format!("destination: {}", dest.display()));

    print_header(entry, &dest, args.dry_run);

    if args.dry_run {
        eprintln!();
        eprintln!(
            "Would download {} file(s) (~{} MB). No files were written.",
            entry.files.len(),
            entry.approx_size_mb
        );
        return CliExit::Success.code();
    }

    if dest.exists() && !args.force {
        let err = CliError::download_destination_exists(&dest);
        eprintln!("{err}");
        return err.exit.code();
    }

    if let Err(e) = fs::create_dir_all(&dest) {
        let err = CliError::from(e);
        eprintln!("{err}");
        return err.exit.code();
    }

    // Sweep any stale `.partial` files left over from a previous
    // interrupted run. We do not resume — `.partial` semantics are
    // "the previous attempt did not complete; start fresh".
    if let Err(e) = sweep_partials(&dest) {
        logging::warn(&format!("could not clean stale .partial files: {e}"));
    }

    eprintln!();
    eprintln!("Downloading:");
    let total = entry.files.len();
    let mut total_bytes: u64 = 0;
    for (idx, file) in entry.files.iter().enumerate() {
        match download_one(entry, file, &dest, fetcher) {
            Ok(bytes) => {
                total_bytes += bytes;
                eprintln!(
                    "  [{}/{}] {} ........ {}",
                    idx + 1,
                    total,
                    file,
                    format_size(bytes)
                );
            }
            Err(detail) => {
                let err = CliError::download_network(file, detail);
                eprintln!();
                eprintln!("{err}");
                return err.exit.code();
            }
        }
    }

    // Post-condition: every expected file landed and is non-empty.
    for file in entry.files {
        let p = dest.join(file);
        match fs::metadata(&p) {
            Ok(m) if m.len() > 0 => {}
            Ok(_) => {
                let err = CliError::download_incomplete(file, "file is empty");
                eprintln!();
                eprintln!("{err}");
                return err.exit.code();
            }
            Err(e) => {
                let err = CliError::download_incomplete(file, &e.to_string());
                eprintln!();
                eprintln!("{err}");
                return err.exit.code();
            }
        }
    }

    eprintln!();
    eprintln!(
        "Done. Files written to {} ({} total).",
        dest.display(),
        format_size(total_bytes)
    );

    if !args.no_suggest {
        eprintln!();
        eprintln!("Next:");
        eprintln!("  atenia diagnose --model {}", dest.display());
        eprintln!("  atenia chat     --model {}", dest.display());
    }

    CliExit::Success.code()
}

/// Build the destination directory: `--dir` if provided, otherwise
/// `./models/<default_subdir>`. The caller is responsible for
/// creating the directory; this only resolves the path.
fn resolve_destination(args: &DownloadArgs, entry: &CatalogEntry) -> PathBuf {
    match &args.dir {
        Some(d) => d.clone(),
        None => Path::new("./models").join(entry.default_subdir),
    }
}

/// Stream one file from HF into `<dest>/<file>.partial`, then
/// atomically rename to `<dest>/<file>`. Returns the number of
/// bytes written so the orchestrator can print a size summary.
fn download_one(
    entry: &CatalogEntry,
    file: &str,
    dest: &Path,
    fetcher: &dyn HttpFetcher,
) -> Result<u64, String> {
    let url = entry.file_url(file);
    let partial = dest.join(format!("{file}.partial"));
    let final_path = dest.join(file);

    // Drop any stale partial so we start from a fresh, empty file.
    if partial.exists() {
        fs::remove_file(&partial).map_err(|e| format!("remove partial: {e}"))?;
    }

    let mut out = fs::File::create(&partial)
        .map_err(|e| format!("create {}: {e}", partial.display()))?;
    let bytes = fetcher.fetch_to_writer(&url, &mut out)?;
    out.flush().map_err(|e| format!("flush: {e}"))?;
    drop(out);

    if bytes == 0 {
        // Don't promote an empty file; let the post-condition check
        // surface it as `E-DOWNLOAD-INCOMPLETE`.
        return Err(format!("server returned 0 bytes for {file}"));
    }

    fs::rename(&partial, &final_path).map_err(|e| format!("rename {file}: {e}"))?;
    Ok(bytes)
}

/// Remove every `*.partial` file at the top of `dest`. Best-effort;
/// any failure is logged but does not abort the download — the
/// downloader will overwrite-then-rename anyway.
fn sweep_partials(dest: &Path) -> std::io::Result<()> {
    if !dest.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dest)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s == "partial")
            .unwrap_or(false)
        {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

/// Pre-download header — eight aligned key/value lines to stderr.
fn print_header(entry: &CatalogEntry, dest: &Path, dry_run: bool) {
    let mode = if dry_run { " (dry run)" } else { "" };
    eprintln!();
    eprintln!("Atenia Engine — model download{mode}");
    eprintln!();
    eprintln!("  alias        {}", entry.alias);
    eprintln!("  model        {}", entry.display_name);
    eprintln!("  family       {}", entry.family);
    eprintln!("  source       {}", entry.source_url());
    eprintln!("  format       {}", entry.format.as_str());
    eprintln!("  size         ~{} MB", entry.approx_size_mb);
    eprintln!("  destination  {}", dest.display());
    eprintln!("  notes        {}", entry.notes);
}

/// `atenia download list` — print a fixed-width catalog table to
/// stderr. stdout is left empty so piping into a script gets no
/// noise.
fn print_list() {
    eprintln!();
    eprintln!("Available models ({}):", catalog::CATALOG.len());
    eprintln!();
    eprintln!(
        "  {:<18} {:<10} {:<8} {:<13} {}",
        "ALIAS", "FAMILY", "SIZE", "FORMAT", "GATED"
    );
    for e in catalog::CATALOG {
        let size = if e.approx_size_mb >= 1024 {
            format!("{:.1} GB", e.approx_size_mb as f32 / 1024.0)
        } else {
            format!("{} MB", e.approx_size_mb)
        };
        eprintln!(
            "  {:<18} {:<10} {:<8} {:<13} {}",
            e.alias,
            e.family,
            size,
            e.format.as_str(),
            if e.gated { "yes" } else { "no" }
        );
    }
    eprintln!();
    eprintln!("Use:  atenia download <alias>");
}

/// Pretty-print a byte count. The download surface only needs
/// human approximation; no need to round-trip through `humansize`.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_picks_human_units() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(2048), "2.0 KB");
        assert_eq!(format_size(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(format_size(3 * 1024 * 1024 * 1024), "3.00 GB");
    }

    #[test]
    fn resolve_destination_uses_default_subdir_when_dir_absent() {
        let entry = catalog::find("tinyllama").unwrap();
        let args = DownloadArgs {
            alias: "tinyllama".into(),
            dir: None,
            force: false,
            dry_run: false,
            no_suggest: false,
        };
        let p = resolve_destination(&args, entry);
        assert_eq!(p, Path::new("./models").join("tinyllama-1.1b-chat"));
    }

    #[test]
    fn resolve_destination_honours_explicit_dir() {
        let entry = catalog::find("tinyllama").unwrap();
        let args = DownloadArgs {
            alias: "tinyllama".into(),
            dir: Some(PathBuf::from("/tmp/elsewhere")),
            force: false,
            dry_run: false,
            no_suggest: false,
        };
        let p = resolve_destination(&args, entry);
        assert_eq!(p, PathBuf::from("/tmp/elsewhere"));
    }
}
