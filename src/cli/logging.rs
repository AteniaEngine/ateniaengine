//! **CLI logging (CLI-2).**
//!
//! A small, dependency-free logging layer for the `atenia` command
//! line. It is deliberately *not* `tracing`: the scope here is the
//! CLI frontier — a handful of structured lines per command — not
//! engine-wide instrumentation, and the project keeps its
//! dependency surface minimal. A hand-rolled logger covers every
//! CLI-2 requirement (levels, quiet/verbose/debug/trace, a log
//! file, a per-run trace id) in well under a page of code and is
//! fully testable through the binary's stderr / the log file.
//!
//! Contract:
//! - **stderr** receives level-filtered, human-shaped lines.
//! - an optional **log file** receives every emitted line with a
//!   timestamp, level, trace id and target.
//! - **stdout is never touched** — generated text / reports / JSON
//!   stay clean.
//!
//! This module touches no runtime core, no graph builder and no
//! loader. It is initialised once at the start of `main()`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use super::error::CliError;

/// Log severity. `Error` is the most severe (ordinal 0); `Trace`
/// the least (ordinal 4). A message at level `m` is emitted when
/// `m <= configured_level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Failures the user must see.
    Error = 0,
    /// Real problems that did not stop the command.
    Warn = 1,
    /// Useful progress steps (`--verbose`).
    Info = 2,
    /// Resolved configuration / internal decisions (`--debug`).
    Debug = 3,
    /// Fine-grained CLI-frontier detail (`--trace`).
    Trace = 4,
}

impl LogLevel {
    /// Parse an explicit `--log-level` value.
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Ok(LogLevel::Error),
            "warn" => Ok(LogLevel::Warn),
            "info" => Ok(LogLevel::Info),
            "debug" => Ok(LogLevel::Debug),
            "trace" => Ok(LogLevel::Trace),
            other => Err(CliError::invalid_args(
                format!("unknown --log-level `{other}`"),
                "Use one of: error, warn, info, debug, trace.",
            )),
        }
    }

    /// Short uppercase label used in rendered lines.
    fn label(self) -> &'static str {
        match self {
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
            LogLevel::Trace => "TRACE",
        }
    }
}

/// Resolve the effective log level from the global verbosity
/// flags, applying the documented precedence:
///
/// 1. an explicit `--log-level` wins over everything;
/// 2. otherwise `--quiet` wins over `--verbose`/`--debug`/`--trace`;
/// 3. otherwise `--trace` > `--debug` > `--verbose`;
/// 4. the default is `Warn` — a non-technical user sees real
///    warnings and errors, but no progress or diagnostic chatter.
pub fn resolve_level(
    quiet: bool,
    verbose: bool,
    debug: bool,
    trace: bool,
    explicit: Option<&str>,
) -> Result<LogLevel, CliError> {
    if let Some(e) = explicit {
        return LogLevel::parse(e);
    }
    if quiet {
        return Ok(LogLevel::Error);
    }
    if trace {
        return Ok(LogLevel::Trace);
    }
    if debug {
        return Ok(LogLevel::Debug);
    }
    if verbose {
        return Ok(LogLevel::Info);
    }
    Ok(LogLevel::Warn)
}

/// Per-run context: the trace id and, if any, the log-file path.
/// Shared so [`CliError`] rendering can append a `Trace:` footer.
#[derive(Clone, Debug)]
pub struct RunContext {
    /// Stable id for this invocation, e.g. `atenia-1747929012-a91f3c2d`.
    pub trace_id: String,
    /// The log file in use, when `--log-file` was given.
    pub log_file: Option<PathBuf>,
}

/// Configuration assembled from the global CLI flags.
pub struct CliLogConfig {
    /// Effective level (see [`resolve_level`]).
    pub level: LogLevel,
    /// Optional `--log-file` path.
    pub log_file: Option<PathBuf>,
    /// Optional user-supplied `--trace-id`.
    pub trace_id: Option<String>,
    /// `--no-color`. Accepted and stored; colored output is not
    /// implemented yet, so this is currently a documented no-op.
    pub no_color: bool,
}

struct CliLogger {
    level: LogLevel,
    trace_id: String,
    file: Option<Mutex<File>>,
}

static LOGGER: OnceLock<CliLogger> = OnceLock::new();
static RUN_CONTEXT: OnceLock<RunContext> = OnceLock::new();
/// Mirror of the configured level as a plain atomic, so
/// [`level_at_least`] is cheap and works even before init.
static LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Warn as u8);

/// Generate a per-run trace id: `atenia-<unix_seconds>-<8 hex>`.
/// The hex suffix comes from `uuid` (already a project dependency),
/// so two runs in the same second still differ. No date crate is
/// pulled in — a Unix timestamp is a sortable, sufficient stamp.
fn generate_trace_id() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let suffix: String = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    format!("atenia-{secs}-{suffix}")
}

/// Initialise CLI logging. Called once at the start of `main()`.
///
/// On success the global logger + [`RunContext`] are installed and
/// the context is returned. The only failure is a `--log-file`
/// that cannot be created — reported as a [`CliError`] so `main`
/// can render it like any other CLI error.
pub fn init_cli_logging(cfg: CliLogConfig) -> Result<RunContext, CliError> {
    let trace_id = cfg.trace_id.unwrap_or_else(generate_trace_id);

    let file = match &cfg.log_file {
        Some(path) => Some(open_log_file(path)?),
        None => None,
    };

    LEVEL.store(cfg.level as u8, Ordering::Relaxed);

    // `set` fails only if logging was already initialised (e.g. a
    // test process re-entering); ignoring that is harmless.
    let _ = LOGGER.set(CliLogger {
        level: cfg.level,
        trace_id: trace_id.clone(),
        file: file.map(Mutex::new),
    });

    let ctx = RunContext {
        trace_id,
        log_file: cfg.log_file,
    };
    let _ = RUN_CONTEXT.set(ctx.clone());
    let _ = cfg.no_color; // reserved; color output not implemented yet
    Ok(ctx)
}

/// Open (creating parent directories and the file) the log file,
/// translating any I/O failure into a typed [`CliError`].
fn open_log_file(path: &PathBuf) -> Result<File, CliError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| match e.kind() {
                std::io::ErrorKind::PermissionDenied => {
                    CliError::io_permission("the log file directory", parent)
                }
                _ => CliError::io_not_found("the log file directory", parent),
            })?;
        }
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::PermissionDenied => {
                CliError::io_permission("the log file", path)
            }
            _ => CliError::io_not_found("the log file", path),
        })
}

/// The active run context, if logging has been initialised.
pub fn current_run_context() -> Option<RunContext> {
    RUN_CONTEXT.get().cloned()
}

/// `true` when a message at `level` would currently be emitted.
/// Cheap; safe to call before init (defaults to `Warn`).
pub fn level_at_least(level: LogLevel) -> bool {
    (level as u8) <= LEVEL.load(Ordering::Relaxed)
}

/// Emit a log line. Filtered by the configured level for stderr;
/// the log file (if any) receives every emitted line with full
/// metadata. Never writes to stdout.
pub fn log(level: LogLevel, message: &str) {
    let Some(logger) = LOGGER.get() else {
        // Not initialised (e.g. a unit test): fall back to a bare
        // stderr write for warnings and errors only.
        if (level as u8) <= (LogLevel::Warn as u8) {
            eprintln!("[{}] {message}", level.label());
        }
        return;
    };
    if level > logger.level {
        return;
    }
    // Human-shaped line on stderr.
    eprintln!("[{}] {message}", level.label());
    // Full structured line in the log file.
    if let Some(file) = &logger.file {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(
                f,
                "{secs} {} [{}] atenia.cli: {message}",
                level.label(),
                logger.trace_id
            );
        }
    }
}

/// Convenience: log at `Info`.
pub fn info(message: &str) {
    log(LogLevel::Info, message);
}

/// Convenience: log at `Warn`.
pub fn warn(message: &str) {
    log(LogLevel::Warn, message);
}

/// Convenience: log at `Debug`.
pub fn debug(message: &str) {
    log(LogLevel::Debug, message);
}

/// Convenience: log at `Trace`.
pub fn trace(message: &str) {
    log(LogLevel::Trace, message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_parse_accepts_all_names() {
        assert_eq!(LogLevel::parse("error").unwrap(), LogLevel::Error);
        assert_eq!(LogLevel::parse("WARN").unwrap(), LogLevel::Warn);
        assert_eq!(LogLevel::parse("Info").unwrap(), LogLevel::Info);
        assert_eq!(LogLevel::parse("debug").unwrap(), LogLevel::Debug);
        assert_eq!(LogLevel::parse("trace").unwrap(), LogLevel::Trace);
        assert!(LogLevel::parse("loud").is_err());
    }

    #[test]
    fn level_ordering_is_error_low_trace_high() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn explicit_log_level_wins_over_every_flag() {
        let lvl = resolve_level(true, true, true, true, Some("warn")).unwrap();
        assert_eq!(lvl, LogLevel::Warn);
    }

    #[test]
    fn quiet_wins_over_verbose_debug_trace() {
        let lvl = resolve_level(true, true, true, true, None).unwrap();
        assert_eq!(lvl, LogLevel::Error);
    }

    #[test]
    fn trace_beats_debug_beats_verbose() {
        assert_eq!(
            resolve_level(false, true, true, true, None).unwrap(),
            LogLevel::Trace
        );
        assert_eq!(
            resolve_level(false, true, true, false, None).unwrap(),
            LogLevel::Debug
        );
        assert_eq!(
            resolve_level(false, true, false, false, None).unwrap(),
            LogLevel::Info
        );
    }

    #[test]
    fn default_level_is_warn() {
        assert_eq!(
            resolve_level(false, false, false, false, None).unwrap(),
            LogLevel::Warn
        );
    }

    #[test]
    fn trace_id_has_expected_shape() {
        let id = generate_trace_id();
        assert!(id.starts_with("atenia-"), "{id}");
        // atenia-<digits>-<8 hex>
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 3, "{id}");
        assert!(parts[1].chars().all(|c| c.is_ascii_digit()), "{id}");
        assert_eq!(parts[2].len(), 8, "{id}");
        assert!(parts[2].chars().all(|c| c.is_ascii_hexdigit()), "{id}");
    }
}
