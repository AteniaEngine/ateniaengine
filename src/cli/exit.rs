//! CLI exit-code contract.
//!
//! A single, consistent exit-code scheme shared by every `atenia`
//! subcommand that routes its failures through [`super::CliError`].
//! Codes are stable: scripts and CI can branch on them.

/// Process exit code for an `atenia` invocation.
///
/// The variant *categorises the fault*; it is set by each
/// [`super::CliError`] constructor, not derived from the error
/// code string, so a constructor can pick the right category
/// explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliExit {
    /// `0` — the command completed successfully.
    Success,
    /// `1` — system / environment fault: a filesystem permission
    /// error, an unreadable file, a missing backend. Not something
    /// the user typed wrong — something the host could not provide.
    System,
    /// `2` — user-input fault: a bad argument, a missing or
    /// malformed config / adapter spec, a path the user supplied
    /// that does not exist, a validation failure.
    UserInput,
    /// `3` — runtime-execution fault: model load, generation, GPU
    /// or memory failure that happened while actually running the
    /// model.
    Runtime,
    /// `101` — an unexpected internal panic was caught at the CLI
    /// boundary. Mirrors Rust's own default panic exit code.
    InternalPanic,
}

impl CliExit {
    /// The numeric process exit code.
    pub fn code(self) -> i32 {
        match self {
            CliExit::Success => 0,
            CliExit::System => 1,
            CliExit::UserInput => 2,
            CliExit::Runtime => 3,
            CliExit::InternalPanic => 101,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(CliExit::Success.code(), 0);
        assert_eq!(CliExit::System.code(), 1);
        assert_eq!(CliExit::UserInput.code(), 2);
        assert_eq!(CliExit::Runtime.code(), 3);
        assert_eq!(CliExit::InternalPanic.code(), 101);
    }
}
