//! **CLI-7** — `atenia quickstart`: first-run UX.
//!
//! A pure-presentation command that turns the four-step onboarding
//! flow (`doctor` → `download` → `diagnose` → `chat`) into a single,
//! self-explanatory entry point so a new user can go from `cargo
//! install` (or downloaded binary) to a running chat without
//! having to read the README first.
//!
//! By default the command is **non-destructive**: it prints the
//! exact commands to run and exits. With `--download` it delegates
//! to [`super::download::run_download_with`] using the same curated
//! catalog (CLI-6), then prints the "Next:" footer for `diagnose` /
//! `chat`. There is no interactive prompt — staying scriptable was
//! a hard constraint of the design.
//!
//! What this module does **not** do:
//!   - run `atenia chat` / `generate` automatically,
//!   - benchmark, probe GPUs, or pull diagnostics beyond what
//!     `doctor` already does (it does not even invoke `doctor`;
//!     it just points at it),
//!   - download anything outside the CLI-6 curated catalog.
//!
//! All output goes to **stderr**, matching the rest of the CLI
//! frontier: stdout stays free for any future `--json` mode and
//! piping into a script captures nothing by accident.

use std::path::PathBuf;

use crate::cli::download::{
    self,
    fetch::{HttpFetcher, UreqFetcher},
    DownloadArgs,
};
use crate::cli::error::CliError;
use crate::cli::exit::CliExit;
use crate::cli::logging;

/// Default model the quickstart suggests / downloads when the user
/// does not pass `--model`. Smallest entry in the CLI-6 catalog,
/// chosen so the first download fits in a coffee break on any
/// connection.
pub const DEFAULT_MODEL: &str = "smollm2-135m";

/// Arguments collected by `clap` for the `quickstart` subcommand.
pub struct QuickstartArgs {
    /// Actually run `atenia download <model>` instead of just
    /// printing the suggested commands. Off by default to keep
    /// the command idempotent and side-effect-free.
    pub download: bool,
    /// Curated alias to suggest / download. Must exist in the
    /// CLI-6 catalog; unknown aliases fail with the same
    /// `E-DOWNLOAD-UNKNOWN-MODEL` the download command emits.
    pub model: String,
    /// Custom destination passed through to `download`. Ignored
    /// when `--download` is absent (the model path is not written
    /// in plan mode either way).
    pub dir: Option<PathBuf>,
    /// Skip the post-download "Next:" footer. Mirrors the same
    /// flag on `atenia download`.
    pub no_suggest: bool,
}

/// Entry point called from `src/bin/atenia.rs`.
pub fn run_quickstart(args: QuickstartArgs) -> i32 {
    let fetcher = UreqFetcher::new();
    run_quickstart_with(args, &fetcher)
}

/// Same as [`run_quickstart`] with an injectable fetcher for
/// integration tests. Public so the tests in
/// `tests/cli_quickstart_test.rs` can drive the orchestration
/// without touching the network.
pub fn run_quickstart_with(args: QuickstartArgs, fetcher: &dyn HttpFetcher) -> i32 {
    logging::info("command start: quickstart");
    logging::debug(&format!(
        "model={} download={} dir={:?}",
        args.model, args.download, args.dir
    ));

    // Resolve the alias up front so an unknown model fails the
    // same way whether or not `--download` was passed. Errors
    // borrow the existing CLI-6 catalog and the download error
    // surface — no new code paths.
    let entry = match download::catalog::find(&args.model) {
        Some(e) => e,
        None => {
            let err = CliError::download_unknown_model(&args.model);
            eprintln!("{err}");
            return err.exit.code();
        }
    };

    // Pre-compute the destination path that the suggested
    // commands will reference. Mirrors `download::resolve_destination`;
    // we don't go through the download module's private resolver to
    // keep this module decoupled — a single `Path::new(...).join(...)`
    // is not worth re-exporting an internal helper for.
    let dest = args
        .dir
        .clone()
        .unwrap_or_else(|| std::path::Path::new("./models").join(entry.default_subdir));

    print_intro(entry, &dest, args.download);

    if args.download {
        // Delegate to the same orchestration the `atenia download`
        // subcommand uses. The fetcher comes from this layer so
        // tests can substitute a fake; production goes through
        // `UreqFetcher::new()`.
        let dl_args = DownloadArgs {
            alias: args.model.clone(),
            dir: args.dir.clone(),
            force: false,
            dry_run: false,
            // We will print our own (slightly richer) "Next:"
            // footer ourselves; suppress the one from download.
            no_suggest: true,
        };
        let code = download::run_download_with(dl_args, fetcher);
        if code != CliExit::Success.code() {
            // `run_download_with` already printed the error to
            // stderr; just propagate the exit code.
            return code;
        }
        if !args.no_suggest {
            print_post_download_next(&dest);
        }
        return CliExit::Success.code();
    }

    // Plan mode (no --download): print the four-step recipe with
    // the user's chosen alias substituted in.
    print_plan(entry, &dest, args.no_suggest);
    CliExit::Success.code()
}

/// Header — what this command is about. Same shape regardless of
/// whether we will download or just print the plan.
fn print_intro(
    entry: &download::catalog::CatalogEntry,
    dest: &std::path::Path,
    will_download: bool,
) {
    eprintln!();
    eprintln!("Atenia Engine — quickstart");
    eprintln!();
    eprintln!("  recommended model  {} ({})", entry.alias, entry.display_name);
    eprintln!("  family             {}", entry.family);
    eprintln!("  size               ~{} MB", entry.approx_size_mb);
    eprintln!("  destination        {}", dest.display());
    if will_download {
        eprintln!();
        eprintln!("Downloading now (this may take a few minutes on first run).");
    }
}

/// Four-step plan printed when `--download` is absent. Each step
/// is a literal copy-pasteable command, with the user's selected
/// alias / destination substituted in.
fn print_plan(
    entry: &download::catalog::CatalogEntry,
    dest: &std::path::Path,
    no_suggest: bool,
) {
    let alias = entry.alias;
    eprintln!();
    eprintln!("Suggested first-run flow:");
    eprintln!();
    eprintln!("  Step 1 — Check your system:");
    eprintln!("    atenia doctor");
    eprintln!();
    eprintln!("  Step 2 — Download the recommended model:");
    eprintln!("    atenia download {alias}");
    eprintln!();
    eprintln!("  Step 3 — Verify the model:");
    eprintln!("    atenia diagnose --model {}", dest.display());
    eprintln!();
    eprintln!("  Step 4 — Start chatting:");
    eprintln!("    atenia chat --model {}", dest.display());

    if !no_suggest {
        eprintln!();
        eprintln!("Want the shortest path?");
        eprintln!("  atenia quickstart --download");
        eprintln!();
        eprintln!("Tip:");
        eprintln!("  Use `atenia download list` to see other curated models.");
    }
}

/// Post-download "Next:" footer printed when `--download` succeeded.
/// We print our own instead of borrowing the one from `download`
/// because the alias / destination context is already in scope here
/// and we want to control the wording.
fn print_post_download_next(dest: &std::path::Path) {
    eprintln!();
    eprintln!("Next:");
    eprintln!("  atenia diagnose --model {}", dest.display());
    eprintln!("  atenia chat     --model {}", dest.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::download::catalog;

    #[test]
    fn default_model_is_smallest_catalog_entry() {
        // `DEFAULT_MODEL` is the right one: it has to be in the
        // catalog (otherwise the default arg fails the alias
        // resolution at runtime) and it should be the smallest
        // by `approx_size_mb` to stay a defensible "first download"
        // recommendation.
        let entry = catalog::find(DEFAULT_MODEL).expect("default model must be in catalog");
        let smallest = catalog::CATALOG
            .iter()
            .min_by_key(|e| e.approx_size_mb)
            .unwrap();
        assert_eq!(
            entry.alias, smallest.alias,
            "DEFAULT_MODEL must remain the smallest catalog entry; \
             update it whenever the catalog changes"
        );
    }
}
