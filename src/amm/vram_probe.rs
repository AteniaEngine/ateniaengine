//! Real VRAM free readings from NVIDIA GPUs via `nvidia-smi`.
//!
//! This module is intentionally narrow: it only reads free VRAM from a single
//! NVIDIA GPU through the `nvidia-smi` CLI. It does not integrate with the
//! AMM forecaster, guards, or policies. Multi-GPU, AMD/Intel/Apple, and
//! NVML-based paths are out of scope.

use std::io;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VramProbeError {
    /// `nvidia-smi` binary was not found on PATH.
    NvidiaSmiNotFound,
    /// `nvidia-smi` executed but returned a non-zero status or unparseable output.
    CommandFailed(String),
    /// Output could not be parsed as a VRAM value.
    ParseError(String),
    /// More than one GPU was detected. This module supports only single-GPU
    /// reads; multi-GPU handling will be introduced in a future iteration
    /// with an explicit per-index API.
    MultipleGpusUnsupported,
}

const MIB: u64 = 1024 * 1024;

/// Reads free VRAM (in bytes) from the single NVIDIA GPU present on the system.
///
/// Fails loudly if more than one GPU is present.
pub fn read_nvidia_vram_free_bytes() -> Result<u64, VramProbeError> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => VramProbeError::NvidiaSmiNotFound,
            _ => VramProbeError::CommandFailed(e.to_string()),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(VramProbeError::CommandFailed(stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_csv_output(&stdout)
}

/// Parses the CSV-style output of `nvidia-smi --query-gpu=memory.free
/// --format=csv,noheader,nounits` and returns the free VRAM in bytes.
///
/// Expects exactly one non-empty line containing a MiB integer. Multiple
/// non-empty lines are rejected as `MultipleGpusUnsupported`.
fn parse_csv_output(s: &str) -> Result<u64, VramProbeError> {
    let lines: Vec<&str> = s
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return Err(VramProbeError::ParseError(
            "nvidia-smi returned no data lines".to_string(),
        ));
    }

    if lines.len() > 1 {
        return Err(VramProbeError::MultipleGpusUnsupported);
    }

    let mib: u64 = lines[0]
        .parse()
        .map_err(|e| VramProbeError::ParseError(format!("'{}': {}", lines[0], e)))?;

    Ok(mib * MIB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_gpu_line() {
        let s = "7123\n";
        let bytes = parse_csv_output(s).unwrap();
        assert_eq!(bytes, 7123 * MIB);
    }

    #[test]
    fn parses_with_trailing_whitespace() {
        let s = "  7123  \n";
        assert_eq!(parse_csv_output(s).unwrap(), 7123 * MIB);
    }

    #[test]
    fn rejects_empty_output() {
        let err = parse_csv_output("").unwrap_err();
        assert!(matches!(err, VramProbeError::ParseError(_)));
    }

    #[test]
    fn rejects_multiple_gpus() {
        let s = "7123\n5432\n";
        let err = parse_csv_output(s).unwrap_err();
        assert_eq!(err, VramProbeError::MultipleGpusUnsupported);
    }

    #[test]
    fn rejects_non_numeric() {
        let s = "not a number\n";
        let err = parse_csv_output(s).unwrap_err();
        assert!(matches!(err, VramProbeError::ParseError(_)));
    }
}
