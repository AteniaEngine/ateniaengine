//! **CLI boundary layer.**
//!
//! This module is the translation boundary between the engine's
//! technical, typed errors and the human-facing `atenia` command
//! line. The engine stays technical; the CLI speaks to people.
//!
//! - [`error::CliError`] — a stable error code + a human
//!   explanation + a how-to-fix + optional technical details.
//! - [`exit::CliExit`] — the consistent process exit-code scheme.
//! - [`logging`] — CLI-2 logging: levels, quiet/verbose/debug/
//!   trace, an optional log file, and a per-run trace id.
//!
//! The diagnostic subcommands (`doctor` / `diagnose` /
//! `capabilities`) are deliberately **not** here yet — they are a
//! separate, later phase. This module touches no runtime core, no
//! graph builder, and no loader.

pub mod chat;
pub mod diagnostics;
pub mod error;
pub mod exit;
pub mod logging;

pub use error::CliError;
pub use exit::CliExit;
pub use logging::{CliLogConfig, LogLevel, RunContext};
