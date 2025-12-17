//! CUDA backend implementations and detection helpers for the HAL.

use std::{env, process::Command};

/// Returns `true` when CUDA appears to be available on the host.
///
/// It first checks the `ATENIA_FORCE_CUDA` environment variable, which allows
/// forcing detection to succeed in controlled environments (e.g., CI). If the
/// variable is not set, it attempts to execute `nvidia-smi` and reports success
/// when the command exits successfully.
pub fn is_available() -> bool {
    if env::var_os("ATENIA_FORCE_CUDA").is_some() {
        return true;
    }

    Command::new("nvidia-smi")
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
