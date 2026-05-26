//! Human, actionable CLI errors.
//!
//! [`CliError`] is the **boundary** error type for the `atenia`
//! CLI. The engine keeps returning its own technical, typed errors
//! (`LoaderError`, `ConfigError`, `ToolkitError`, …); the CLI layer
//! translates them — once, here — into a stable, human-readable
//! form: a stable error code, a one-line summary, a "what happened"
//! explanation, a "how to fix" instruction, an optional command to
//! run, and optional technical key/value details.
//!
//! This module never touches the runtime core, the graph builders,
//! or the loaders. It only *reads* error values the CLI already
//! has and re-shapes them for the user.

use std::path::Path;

use crate::adapter_toolkit::ToolkitError;

use super::exit::CliExit;

/// A fully-formed, user-facing CLI error.
///
/// Construct via the typed helpers (e.g. [`CliError::config_missing`])
/// or the `From` conversions, never field-by-field at call sites —
/// that keeps the catalogue of error codes in one place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError {
    /// Stable machine code, e.g. `E-TOKENIZER-MISSING`. Never
    /// changes once shipped — scripts and docs may rely on it.
    pub code: &'static str,
    /// One-line summary printed on the `error[CODE]:` line.
    pub summary: String,
    /// Plain-language explanation of what went wrong.
    pub what_happened: String,
    /// Plain-language instruction for fixing it.
    pub how_to_fix: String,
    /// An optional concrete command the user can run to check or
    /// recover.
    pub check_command: Option<String>,
    /// Optional technical key/value pairs (paths, raw engine
    /// messages). Shown unconditionally in this phase; a future
    /// `--debug` flag will gate them.
    pub technical: Vec<(String, String)>,
    /// The process exit code this error maps to.
    pub exit: CliExit,
}

impl CliError {
    /// Base constructor — internal; call sites use the typed
    /// helpers below.
    fn new(
        code: &'static str,
        summary: impl Into<String>,
        what_happened: impl Into<String>,
        how_to_fix: impl Into<String>,
        exit: CliExit,
    ) -> Self {
        Self {
            code,
            summary: summary.into(),
            what_happened: what_happened.into(),
            how_to_fix: how_to_fix.into(),
            check_command: None,
            technical: Vec::new(),
            exit,
        }
    }

    /// Attach a concrete "check it with" command.
    pub fn with_check_command(mut self, cmd: impl Into<String>) -> Self {
        self.check_command = Some(cmd.into());
        self
    }

    /// Attach one technical key/value detail.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.technical.push((key.into(), value.into()));
        self
    }

    /// Render the error to its multi-line human form. Used for
    /// `eprintln!("{err}")` and asserted by the tests.
    ///
    /// When CLI logging has been initialised (CLI-2), a `Trace:`
    /// footer is appended with the run id and, if `--log-file` was
    /// given, the log-file path — so a user reporting a failure
    /// has the run id to quote and knows where the logs landed.
    pub fn render(&self) -> String {
        let mut out = format!("error[{}]: {}\n", self.code, self.summary);
        out.push_str("\nWhat happened:\n");
        out.push_str(&indent(&self.what_happened));
        out.push_str("\nHow to fix:\n");
        out.push_str(&indent(&self.how_to_fix));
        if let Some(cmd) = &self.check_command {
            out.push_str("\nCheck it with:\n");
            out.push_str(&indent(cmd));
        }
        if !self.technical.is_empty() {
            out.push_str("\nTechnical details:\n");
            for (k, v) in &self.technical {
                out.push_str(&indent(&format!("{k}: {v}")));
            }
        }
        if let Some(ctx) = super::logging::current_run_context() {
            out.push_str("\nTrace:\n");
            out.push_str(&indent(&format!("run id: {}", ctx.trace_id)));
            if let Some(path) = &ctx.log_file {
                out.push_str(&indent(&format!("log file: {}", path.display())));
            }
        }
        out
    }

    // ---- typed constructors -------------------------------------

    /// `E-CLI-INVALID-ARGS` — a command-line argument is invalid.
    pub fn invalid_args(summary: impl Into<String>, how_to_fix: impl Into<String>) -> Self {
        Self::new(
            "E-CLI-INVALID-ARGS",
            summary,
            "An argument passed to the command is missing or invalid.",
            how_to_fix,
            CliExit::UserInput,
        )
    }

    /// `E-IO-NOT-FOUND` — a path the user supplied does not exist.
    /// Exit code 2: a wrong `--model` / file argument is user
    /// input, not a host fault.
    pub fn io_not_found(what: &str, path: &Path) -> Self {
        Self::new(
            "E-IO-NOT-FOUND",
            format!("{what} was not found"),
            format!("Atenia could not find {what} at the path you provided."),
            "Check the path for typos, or provide the correct location.",
            CliExit::UserInput,
        )
        .with_detail("path", path.display().to_string())
    }

    /// `E-IO-PERMISSION` — a path exists but cannot be accessed.
    /// Exit code 1: a permission fault is an environment problem.
    pub fn io_permission(what: &str, path: &Path) -> Self {
        Self::new(
            "E-IO-PERMISSION",
            format!("permission denied accessing {what}"),
            format!("The operating system denied access to {what}."),
            "Check the file permissions and the user running `atenia`.",
            CliExit::System,
        )
        .with_detail("path", path.display().to_string())
    }

    /// `E-CONFIG-MISSING` — a model directory has no `config.json`.
    pub fn config_missing(model_dir: &Path) -> Self {
        Self::new(
            "E-CONFIG-MISSING",
            "config.json was not found",
            "Atenia could not find config.json in the model directory, \
             so it cannot tell which architecture this model uses.",
            "Make sure the directory is a complete HuggingFace checkpoint \
             (config.json + weights + tokenizer), or point --model at a \
             single .gguf file's directory instead.",
            CliExit::UserInput,
        )
        .with_check_command(format!("atenia inspect {}", model_dir.display()))
        .with_detail(
            "expected",
            model_dir.join("config.json").display().to_string(),
        )
    }

    /// `E-TOKENIZER-MISSING` — a HF checkpoint has no `tokenizer.json`.
    pub fn tokenizer_missing(model_dir: &Path) -> Self {
        Self::new(
            "E-TOKENIZER-MISSING",
            "tokenizer.json was not found",
            "Atenia could not load the tokenizer for this model.",
            "Re-download the full model files, or copy tokenizer.json \
             into the model directory.",
            CliExit::UserInput,
        )
        .with_check_command(format!("atenia inspect {}", model_dir.display()))
        .with_detail(
            "expected",
            model_dir.join("tokenizer.json").display().to_string(),
        )
    }

    /// `E-ADAPTER-UNSUPPORTED-ARCHITECTURE` — the model is not one
    /// of the families Atenia supports.
    pub fn adapter_unsupported(detail: impl Into<String>) -> Self {
        Self::new(
            "E-ADAPTER-UNSUPPORTED-ARCHITECTURE",
            "model architecture is not supported",
            "This model's architecture is not one Atenia can run. \
             Atenia supports the Llama, Qwen, Gemma, Phi, Mistral (dense) \
             and Falcon3 families.",
            "Use a model from a supported family. Classic Falcon, \
             mixture-of-experts and multimodal models are out of scope.",
            CliExit::UserInput,
        )
        .with_detail("reason", detail.into())
    }

    /// `E-ADAPTER-INVALID-SPEC` — an adapter DSL file is malformed
    /// or fails validation.
    pub fn adapter_invalid_spec(summary: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(
            "E-ADAPTER-INVALID-SPEC",
            summary,
            "The adapter spec file could not be parsed or did not pass \
             validation.",
            "Fix the YAML/JSON spec. Run `atenia debug <file>` for a \
             detailed report, or `atenia inspect <model_dir>` to \
             generate a valid spec automatically.",
            CliExit::UserInput,
        )
        .with_detail("reason", detail.into())
    }

    /// `E-ADAPTER-INSPECT-FAILED` — auto-detection of a model
    /// directory failed.
    pub fn adapter_inspect_failed(detail: impl Into<String>) -> Self {
        Self::new(
            "E-ADAPTER-INSPECT-FAILED",
            "could not inspect the model directory",
            "Atenia could not auto-detect an adapter spec for this \
             directory.",
            "Make sure the directory contains a config.json or a single \
             .gguf file from a supported family.",
            CliExit::UserInput,
        )
        .with_detail("reason", detail.into())
    }

    /// `E-GENERATION-FAILED` — model load or generation failed.
    /// Catch-all for the `generate` command when the underlying
    /// error arrives as an unclassifiable string; the raw message
    /// is preserved in the technical details.
    pub fn generation_failed(summary: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(
            "E-GENERATION-FAILED",
            summary,
            "Atenia failed while loading or running the model.",
            "Verify the model directory contains a valid config.json plus \
             weight files, or exactly one .gguf file. Run \
             `atenia inspect <model_dir>` to check it.",
            CliExit::Runtime,
        )
        .with_detail("engine_error", detail.into())
    }

    /// `E-DOWNLOAD-UNKNOWN-MODEL` — the alias supplied to
    /// `atenia download` is not in the curated catalog.
    pub fn download_unknown_model(alias: &str) -> Self {
        let aliases: Vec<&'static str> =
            crate::cli::download::catalog::aliases().collect();
        Self::new(
            "E-DOWNLOAD-UNKNOWN-MODEL",
            format!("unknown model alias `{alias}`"),
            "Atenia's download command only accepts aliases from a curated \
             catalog of known-good public checkpoints.",
            "Run `atenia download list` to see the available aliases, then \
             retry with one of them.",
            CliExit::UserInput,
        )
        .with_check_command("atenia download list")
        .with_detail("known_aliases", aliases.join(", "))
    }

    /// `E-DOWNLOAD-DESTINATION-EXISTS` — the destination directory
    /// already exists and `--force` was not supplied.
    pub fn download_destination_exists(dest: &Path) -> Self {
        Self::new(
            "E-DOWNLOAD-DESTINATION-EXISTS",
            "destination directory already exists",
            "Atenia refuses to overwrite an existing directory by default, \
             so a half-downloaded model cannot silently clobber a working \
             one.",
            "Pass `--force` to overwrite the directory, or remove it manually \
             and retry.",
            CliExit::UserInput,
        )
        .with_detail("destination", dest.display().to_string())
    }

    /// `E-DOWNLOAD-NETWORK` — a per-file HTTP/TLS/DNS fault. The
    /// raw underlying error is preserved verbatim in the technical
    /// details for the user to share when reporting.
    pub fn download_network(file: &str, detail: impl Into<String>) -> Self {
        Self::new(
            "E-DOWNLOAD-NETWORK",
            format!("network error while fetching `{file}`"),
            "Atenia could not retrieve a file from Hugging Face. This is \
             typically a transient network, DNS or TLS issue, or a Hugging \
             Face outage.",
            "Check your network connectivity and Hugging Face's status page, \
             then retry the command. The download starts over from the first \
             file each time; partial files from this attempt are discarded.",
            CliExit::System,
        )
        .with_detail("file", file.to_string())
        .with_detail("network_error", detail.into())
    }

    /// `E-DOWNLOAD-INCOMPLETE` — every file looked like it was
    /// fetched but a post-condition check failed (missing file or
    /// zero-sized file). Usually means the HF endpoint served an
    /// HTML error page instead of the expected payload.
    pub fn download_incomplete(file: &str, detail: impl Into<String>) -> Self {
        Self::new(
            "E-DOWNLOAD-INCOMPLETE",
            format!("file `{file}` did not download cleanly"),
            "The download reported success for this file but the result was \
             empty or missing on disk. This usually means the server returned \
             an error page instead of the expected payload.",
            "Re-run the command with `--force` to redownload from scratch.",
            CliExit::System,
        )
        .with_detail("file", file.to_string())
        .with_detail("reason", detail.into())
    }

    /// `E-DOWNLOAD-GATED-MODEL` — defence-in-depth check. v1 of the
    /// catalog excludes gated models entirely, but if one ever
    /// slips in with `gated = true` we refuse politely.
    pub fn download_gated_model(entry: &crate::cli::download::catalog::CatalogEntry) -> Self {
        Self::new(
            "E-DOWNLOAD-GATED-MODEL",
            format!("`{}` is a gated model", entry.alias),
            "This model requires accepting a licence on Hugging Face before \
             it can be downloaded. Atenia's download command does not \
             implement OAuth or token-based authentication in v1.",
            "Open the model page in a browser, accept the licence, then use \
             `huggingface-cli download` with a logged-in account.",
            CliExit::UserInput,
        )
        .with_check_command(entry.source_url())
        .with_detail("hf_repo", entry.hf_repo.to_string())
    }

    /// `E-INTERNAL-PANIC` — an unexpected panic was caught at the
    /// CLI boundary.
    pub fn internal_panic(detail: impl Into<String>) -> Self {
        Self::new(
            "E-INTERNAL-PANIC",
            "Atenia hit an unexpected internal error",
            "An internal panic occurred while running the command. \
             This is a bug in Atenia, not in your input.",
            "Re-run the command; if it persists, file an issue with the \
             exact command and model used.",
            CliExit::InternalPanic,
        )
        .with_detail("panic", detail.into())
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

impl std::error::Error for CliError {}

/// Indent every line of `text` by two spaces, ensuring a trailing
/// newline. Used for the body of each rendered section.
fn indent(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    if out.is_empty() {
        out.push('\n');
    }
    out
}

/// `std::io::Error` → `CliError`. `NotFound` and `PermissionDenied`
/// get their specific codes; anything else is a generic system
/// error. Callers that know *what* the path is should prefer the
/// typed constructors ([`CliError::io_not_found`], …) so the
/// summary names the artifact.
impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::NotFound => CliError::new(
                "E-IO-NOT-FOUND",
                "a required file or directory was not found",
                "Atenia could not find a path it needed.",
                "Check the path for typos, or provide the correct location.",
                CliExit::UserInput,
            )
            .with_detail("io_error", e.to_string()),
            std::io::ErrorKind::PermissionDenied => CliError::new(
                "E-IO-PERMISSION",
                "permission denied",
                "The operating system denied access to a path Atenia needed.",
                "Check the file permissions and the user running `atenia`.",
                CliExit::System,
            )
            .with_detail("io_error", e.to_string()),
            _ => CliError::new(
                "E-IO-NOT-FOUND",
                "a filesystem operation failed",
                "Atenia could not complete a filesystem operation.",
                "Check the path, the disk, and the file permissions.",
                CliExit::System,
            )
            .with_detail("io_error", e.to_string()),
        }
    }
}

/// `ToolkitError` (Adapter Toolkit v2) → `CliError`. This is the
/// only place ATKv2 errors are re-shaped for the CLI; the toolkit
/// itself is unchanged.
impl From<ToolkitError> for CliError {
    fn from(e: ToolkitError) -> Self {
        match e {
            ToolkitError::Parse(m) => {
                CliError::adapter_invalid_spec("adapter spec could not be parsed", m)
            }
            ToolkitError::Validation(m) => {
                CliError::adapter_invalid_spec("adapter spec failed validation", m)
            }
            ToolkitError::Resolution(m) => CliError::adapter_unsupported(m),
            ToolkitError::UnsupportedExtension(ext) => CliError::invalid_args(
                format!("unsupported adapter spec extension `{ext}`"),
                "Use an adapter spec file ending in .yaml, .yml or .json.",
            ),
            ToolkitError::Io(m) => CliError::new(
                "E-IO-NOT-FOUND",
                "the adapter spec file could not be read",
                "Atenia could not open the adapter spec file you provided.",
                "Check the path for typos, or provide the correct location.",
                CliExit::UserInput,
            )
            .with_detail("io_error", m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn render_includes_all_mandatory_sections() {
        let err = CliError::tokenizer_missing(&PathBuf::from("models/my-model"));
        let r = err.render();
        assert!(r.starts_with("error[E-TOKENIZER-MISSING]: "));
        assert!(r.contains("What happened:"));
        assert!(r.contains("How to fix:"));
        assert!(r.contains("Check it with:"));
        assert!(r.contains("Technical details:"));
        assert!(r.contains("atenia inspect models/my-model"));
    }

    #[test]
    fn render_omits_optional_sections_when_absent() {
        let err = CliError::invalid_args("bad flag", "pass a valid flag");
        let r = err.render();
        assert!(!r.contains("Check it with:"));
        assert!(!r.contains("Technical details:"));
        assert!(r.contains("What happened:"));
        assert!(r.contains("How to fix:"));
    }

    #[test]
    fn exit_codes_match_fault_category() {
        assert_eq!(
            CliError::io_not_found("the model directory", &PathBuf::from("x")).exit,
            CliExit::UserInput
        );
        assert_eq!(
            CliError::io_permission("the cache directory", &PathBuf::from("x")).exit,
            CliExit::System
        );
        assert_eq!(
            CliError::config_missing(&PathBuf::from("x")).exit,
            CliExit::UserInput
        );
        assert_eq!(
            CliError::generation_failed("load failed", "boom").exit,
            CliExit::Runtime
        );
        assert_eq!(
            CliError::internal_panic("boom").exit,
            CliExit::InternalPanic
        );
    }

    #[test]
    fn toolkit_parse_error_maps_to_invalid_spec() {
        let err: CliError = ToolkitError::Parse("bad yaml".into()).into();
        assert_eq!(err.code, "E-ADAPTER-INVALID-SPEC");
        assert_eq!(err.exit, CliExit::UserInput);
    }

    #[test]
    fn toolkit_validation_error_maps_to_invalid_spec() {
        let err: CliError = ToolkitError::Validation("gqa needs kv_heads".into()).into();
        assert_eq!(err.code, "E-ADAPTER-INVALID-SPEC");
        assert_eq!(err.exit, CliExit::UserInput);
    }

    #[test]
    fn toolkit_resolution_error_maps_to_unsupported_architecture() {
        let err: CliError = ToolkitError::Resolution("unknown family `falcon`".into()).into();
        assert_eq!(err.code, "E-ADAPTER-UNSUPPORTED-ARCHITECTURE");
        assert_eq!(err.exit, CliExit::UserInput);
        // The raw reason is preserved for the user.
        assert!(err.render().contains("falcon"));
    }

    #[test]
    fn toolkit_io_error_maps_to_io_not_found() {
        let err: CliError = ToolkitError::Io("no such file".into()).into();
        assert_eq!(err.code, "E-IO-NOT-FOUND");
        assert_eq!(err.exit, CliExit::UserInput);
    }

    #[test]
    fn std_io_error_kinds_map_to_distinct_codes() {
        let nf: CliError = std::io::Error::from(std::io::ErrorKind::NotFound).into();
        assert_eq!(nf.code, "E-IO-NOT-FOUND");
        let pd: CliError = std::io::Error::from(std::io::ErrorKind::PermissionDenied).into();
        assert_eq!(pd.code, "E-IO-PERMISSION");
        assert_eq!(pd.exit, CliExit::System);
    }

    #[test]
    fn generation_failed_preserves_raw_engine_message() {
        let err = CliError::generation_failed("model load failed", "cuda OOM at layer 12");
        assert!(err.render().contains("cuda OOM at layer 12"));
    }
}
