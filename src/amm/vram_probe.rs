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

/// Snapshot of NVIDIA VRAM state at a single point in time.
///
/// All fields are in bytes. `used_bytes + free_bytes` should equal
/// `total_bytes` modulo reporting rounding from `nvidia-smi`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramSnapshot {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub used_bytes: u64,
}

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

/// Reads a full VRAM snapshot (total, free, used) from the single NVIDIA GPU
/// present on the system.
///
/// Fails loudly if more than one GPU is present.
pub fn read_nvidia_vram_snapshot() -> Result<VramSnapshot, VramProbeError> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free,memory.used",
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
    parse_snapshot_csv_output(&stdout)
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

/// Parses the CSV output of the snapshot query (`total, free, used` per line).
/// Rejects empty or multi-GPU outputs, same contract as [`parse_csv_output`].
fn parse_snapshot_csv_output(s: &str) -> Result<VramSnapshot, VramProbeError> {
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

    let parts: Vec<&str> = lines[0].split(',').map(|p| p.trim()).collect();
    if parts.len() != 3 {
        return Err(VramProbeError::ParseError(format!(
            "expected 3 CSV fields, got {}: '{}'",
            parts.len(),
            lines[0]
        )));
    }

    let parse_mib = |s: &str| -> Result<u64, VramProbeError> {
        s.parse::<u64>()
            .map(|mib| mib * MIB)
            .map_err(|e| VramProbeError::ParseError(format!("'{}': {}", s, e)))
    };

    Ok(VramSnapshot {
        total_bytes: parse_mib(parts[0])?,
        free_bytes: parse_mib(parts[1])?,
        used_bytes: parse_mib(parts[2])?,
    })
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

    #[test]
    fn parses_snapshot_line() {
        let s = "8192, 7123, 1069\n";
        let snap = parse_snapshot_csv_output(s).unwrap();
        assert_eq!(snap.total_bytes, 8192 * MIB);
        assert_eq!(snap.free_bytes, 7123 * MIB);
        assert_eq!(snap.used_bytes, 1069 * MIB);
    }

    #[test]
    fn snapshot_rejects_wrong_arity() {
        let s = "8192, 7123\n";
        let err = parse_snapshot_csv_output(s).unwrap_err();
        assert!(matches!(err, VramProbeError::ParseError(_)));
    }

    #[test]
    fn snapshot_rejects_multiple_gpus() {
        let s = "8192, 7123, 1069\n8192, 6000, 2192\n";
        let err = parse_snapshot_csv_output(s).unwrap_err();
        assert_eq!(err, VramProbeError::MultipleGpusUnsupported);
    }
}
