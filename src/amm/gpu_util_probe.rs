//! GPU compute-utilization probe with **process attribution** (M3-e.7).
//!
//! Parallel to [`vram_probe`](crate::amm::vram_probe) — also an
//! `nvidia-smi` subprocess — but targets **compute**, not memory, and
//! breaks the reading down per-process so the reaction loop can tell
//! "Atenia is using the GPU" from "some other process is using the
//! GPU".
//!
//! ## Subcommand
//!
//! Uses `nvidia-smi pmon -c 1 -s u`:
//!
//! - `pmon` is the Process Monitor subcommand. Unlike
//!   `--query-compute-apps` (which only exposes **memory** per process),
//!   `pmon` reports per-process `sm` (Streaming Multiprocessor
//!   utilization, a.k.a. compute %) which is what we need.
//! - `-c 1` takes exactly one sample and exits.
//! - `-s u` pins the column selection to the "utilization" set
//!   (10 columns: `gpu, pid, type, sm, mem, enc, dec, jpg, ofa,
//!   command`). Without this flag some driver versions add `fb` and
//!   `ccpm` columns (12-column output), which would break a
//!   position-based parser.
//!
//! ## Why not NVML
//!
//! NVML via `nvml-wrapper` would give a typed API with
//! `process_utilization_stats`, but it requires a new crate
//! dependency and creates a parallel path to `vram_probe`. The
//! subprocess approach stays consistent with what the rest of the
//! `amm` module already does. Latency measured at ~75 ms steady-
//! state on Windows with NVIDIA driver 551 — well under the
//! [`SIGNAL_BUS_CACHE_TTL`][crate::amm::signal_bus::SIGNAL_BUS_CACHE_TTL]
//! of 100 ms, so no separate cache TTL is needed.
//!
//! ## Why observability-only
//!
//! This probe feeds `GuardConditions` but **does not add a new
//! `GuardAction` variant or veto logic**. When the GPU is saturated
//! by an external process while VRAM is fine, there is no effective
//! action the current reaction loop can take — `Degrade` exists to
//! relieve VRAM pressure, not compute contention. See the M3-e.7
//! research report in the handoff for the full rationale. The signal
//! is still valuable: it shows up in the enriched `[AMG Guard]` logs,
//! it lets future milestones reason about GPU compute state without
//! re-implementing the probe, and it populates a dormant field in
//! `GuardConditions` the way `latency_spike` does for a consumer
//! that does not exist yet (M3-e.10).
//!
//! ## Fail-open and platform degradation
//!
//! Inherits the fail-open contract of the other probes: any error
//! (nvidia-smi missing, pmon failing, output unparseable, multi-GPU
//! detected) results in `Err(GpuUtilProbeError)` that `SignalBus`
//! turns into `None` on the corresponding `GuardConditions` fields.
//! On non-NVIDIA platforms the probe degrades to "no signal" rather
//! than blocking execution.

use std::io;
use std::process::Command;
use std::sync::Mutex;

/// Snapshot of GPU compute utilization at a single point in time.
///
/// Both fields normalized to `[0.0, 1.0]`:
/// - `total_fraction`: sum of `sm` values over all rows in the
///   `pmon` output, divided by 100.0. A process that saturates
///   the whole SM pool reads ~100; if two such processes coexist
///   the sum may exceed 100 slightly (samples are not synchronized
///   per-core). Clamped to `[0.0, 1.0]`.
/// - `self_fraction`: the `sm` value from the row matching this
///   process's PID, divided by 100.0. If Atenia issued no GPU work
///   during the sample interval it does not appear in the `pmon`
///   output and this field is `0.0` — not `None`. The distinction
///   matters: "Atenia is using 0% compute right now" is different
///   from "we don't know Atenia's compute usage", and the `Option`
///   wrapping lives one layer up in `GuardConditions`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuUtilSnapshot {
    pub total_fraction: f32,
    pub self_fraction: f32,
}

/// Reasons a GPU utilization probe can fail. All variants map to
/// `None` on `GuardConditions` downstream — fail-open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuUtilProbeError {
    /// `nvidia-smi` binary was not found on PATH. Expected on non-
    /// NVIDIA platforms (Mac Apple Silicon, AMD-only, Intel-only).
    NvidiaSmiNotFound,
    /// `nvidia-smi pmon` executed but returned a non-zero status.
    /// Possible causes: driver too old to support pmon, WDDM-mode
    /// restrictions on Windows for certain fields, or transient
    /// errors.
    PmonFailed(String),
    /// Output could not be parsed as a pmon result.
    ParseError(String),
    /// Multi-GPU detected in the pmon output. Single-GPU only is
    /// the same constraint as `vram_probe::MultipleGpusUnsupported`;
    /// inherited for consistency.
    MultipleGpusUnsupported,
}

/// Abstract interface over a GPU utilization probe. Production code
/// instantiates [`GpuUtilProbe`]; tests inject a fake that returns
/// canned snapshots. `Send + Sync` so `SignalBus` can carry it
/// behind `Arc<dyn GpuUtilProbeApi>` across threads.
pub trait GpuUtilProbeApi: Send + Sync {
    fn snapshot(&self) -> Result<GpuUtilSnapshot, GpuUtilProbeError>;
}

/// Production GPU utilization probe. Resolves the current PID once
/// at construction and uses it to attribute the `pmon` rows; the
/// subprocess itself is stateless, so no warmup is required (unlike
/// the CPU probe from M3-e.6).
pub struct GpuUtilProbe {
    own_pid: u32,
    /// Serializes concurrent `snapshot` calls so multiple threads do
    /// not race subprocess spawns. Not strictly necessary for
    /// correctness (each subprocess is independent), but reduces
    /// driver-side contention when the reaction loop is exercised
    /// from multiple threads in tests.
    lock: Mutex<()>,
}

impl GpuUtilProbe {
    /// Construct a probe. Cheap (no warmup, no subprocess). The
    /// current PID is resolved at this point so the attribution
    /// logic in `snapshot` has a stable key.
    pub fn new() -> Self {
        Self {
            own_pid: std::process::id(),
            lock: Mutex::new(()),
        }
    }

}

impl Default for GpuUtilProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuUtilProbeApi for GpuUtilProbe {
    fn snapshot(&self) -> Result<GpuUtilSnapshot, GpuUtilProbeError> {
        // Best-effort serialization — poisoned lock is recovered
        // rather than propagated (same policy as CpuProbe).
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());

        let output = Command::new("nvidia-smi")
            .args(["pmon", "-c", "1", "-s", "u"])
            .output()
            .map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => GpuUtilProbeError::NvidiaSmiNotFound,
                _ => GpuUtilProbeError::PmonFailed(e.to_string()),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(GpuUtilProbeError::PmonFailed(stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_pmon_output(&stdout, self.own_pid)
    }
}

/// Parses the stdout of `nvidia-smi pmon -c 1 -s u` into a
/// normalized `GpuUtilSnapshot`. Attribution is driven by `own_pid`:
/// rows matching it contribute to `self_fraction`, all rows
/// contribute to `total_fraction`.
///
/// Grammar (empirical, driver 551.x):
///
/// ```text
/// # gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
/// # Idx           #    C/G      %      %      %      %      %      %    name
///     0          -     -      -      -      -      -      -      -    -
/// ```
///
/// or, with active processes:
///
/// ```text
///     0       12345     C     42     13      0      0      0      0    python.exe
///     0        6789     C     15      8      0      0      0      0    other.exe
/// ```
///
/// Rules:
/// - Lines starting with `#` are comments / headers — skip.
/// - Empty lines — skip.
/// - Each data line has at least 10 whitespace-separated tokens; the
///   first is the GPU index, the second is the PID, the fourth is
///   the SM percent.
/// - A `-` in the `sm` column means the driver did not report a
///   value for this row; it contributes `0` to `total` and `0` to
///   `self` regardless of PID.
/// - A row with `pid == "-"` is an empty-GPU placeholder (no active
///   compute/graphics processes) — skip.
/// - Distinct GPU index values across rows → `MultipleGpusUnsupported`.
fn parse_pmon_output(
    stdout: &str,
    own_pid: u32,
) -> Result<GpuUtilSnapshot, GpuUtilProbeError> {
    let mut total_sm: f32 = 0.0;
    let mut self_sm: f32 = 0.0;
    let mut gpu_index_seen: Option<&str> = None;

    for raw_line in stdout.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 10 {
            return Err(GpuUtilProbeError::ParseError(format!(
                "expected at least 10 tokens in pmon data line, got {}: '{}'",
                tokens.len(),
                line
            )));
        }

        // Multi-GPU detection. All rows must share the same gpu
        // index; if not, we do not try to arbitrate — same policy
        // as vram_probe.
        let gpu_idx = tokens[0];
        match gpu_index_seen {
            None => gpu_index_seen = Some(gpu_idx),
            Some(first) if first == gpu_idx => {}
            Some(_) => return Err(GpuUtilProbeError::MultipleGpusUnsupported),
        }

        let pid_token = tokens[1];
        if pid_token == "-" {
            // No process on this row — an "empty GPU" placeholder.
            continue;
        }

        // Parse SM column. `-` means the driver did not report a
        // value; treat as 0 contribution.
        let sm_token = tokens[3];
        let sm_value: f32 = if sm_token == "-" {
            0.0
        } else {
            sm_token.parse::<f32>().map_err(|e| {
                GpuUtilProbeError::ParseError(format!(
                    "sm column '{}': {}",
                    sm_token, e
                ))
            })?
        };

        total_sm += sm_value;

        // Attribute to self only if the PID matches. A parse error
        // on the PID token is fatal — it means the line layout is
        // not what we expect.
        let pid_value: u32 = pid_token.parse::<u32>().map_err(|e| {
            GpuUtilProbeError::ParseError(format!(
                "pid column '{}': {}",
                pid_token, e
            ))
        })?;

        if pid_value == own_pid {
            self_sm += sm_value;
        }
    }

    Ok(GpuUtilSnapshot {
        total_fraction: (total_sm / 100.0).clamp(0.0, 1.0),
        self_fraction: (self_sm / 100.0).clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parsing edge cases -----------------------------------

    #[test]
    fn parse_empty_gpu_placeholder() {
        // Real observed output when no compute/graphics processes
        // are active: a single data line with dashes.
        let s = "\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command\n\
# Idx           #    C/G      %      %      %      %      %      %    name\n\
    0          -     -      -      -      -      -      -      -    -\n";
        let snap = parse_pmon_output(s, 12345).expect("must parse");
        assert_eq!(snap.total_fraction, 0.0);
        assert_eq!(snap.self_fraction, 0.0);
    }

    #[test]
    fn parse_single_process_is_atenia() {
        // One compute process, and it is us. Both total and self
        // should reflect the SM value.
        let s = "\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command\n\
# Idx           #    C/G      %      %      %      %      %      %    name\n\
    0       12345     C     42     13      0      0      0      0    atenia.exe\n";
        let snap = parse_pmon_output(s, 12345).expect("must parse");
        assert!((snap.total_fraction - 0.42).abs() < 1e-5);
        assert!((snap.self_fraction - 0.42).abs() < 1e-5);
    }

    #[test]
    fn parse_single_process_is_other() {
        // One compute process, and it is NOT us. Total reflects it,
        // self is zero (we did not run any GPU work during the
        // sample).
        let s = "\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command\n\
    0        9999     C     77     20      0      0      0      0    other.exe\n";
        let snap = parse_pmon_output(s, 12345).expect("must parse");
        assert!((snap.total_fraction - 0.77).abs() < 1e-5);
        assert_eq!(snap.self_fraction, 0.0);
    }

    #[test]
    fn parse_multiple_processes_attribution() {
        // Two compute processes: Atenia (PID 1111, 30% SM) and an
        // external one (PID 2222, 55% SM). Total = 0.85, self = 0.30.
        let s = "\
    0        1111     C     30     10      0      0      0      0    atenia.exe\n\
    0        2222     C     55     15      0      0      0      0    external.exe\n";
        let snap = parse_pmon_output(s, 1111).expect("must parse");
        assert!((snap.total_fraction - 0.85).abs() < 1e-5);
        assert!((snap.self_fraction - 0.30).abs() < 1e-5);
    }

    #[test]
    fn parse_graphics_plus_compute() {
        // A mix of compute (C) and graphics (G) processes. `type`
        // column is not used for attribution — we sum all SM values
        // regardless of type, because a gaming process still
        // competes with Atenia for the same GPU time.
        let s = "\
    0        1111     C     20      8      0      0      0      0    atenia.exe\n\
    0        3333     G     60     25      0      0      0      0    game.exe\n";
        let snap = parse_pmon_output(s, 1111).expect("must parse");
        assert!((snap.total_fraction - 0.80).abs() < 1e-5);
        assert!((snap.self_fraction - 0.20).abs() < 1e-5);
    }

    #[test]
    fn parse_dash_sm_contributes_zero() {
        // Driver reported `-` for SM (happens in WDDM for some
        // metrics). Row should count as zero contribution, not
        // cause a parse error.
        let s = "\
    0       12345     C      -     13      0      0      0      0    someapp.exe\n";
        let snap = parse_pmon_output(s, 12345).expect("must parse");
        assert_eq!(snap.total_fraction, 0.0);
        assert_eq!(snap.self_fraction, 0.0);
    }

    #[test]
    fn parse_rejects_multi_gpu() {
        let s = "\
    0        1111     C     20     10      0      0      0      0    atenia.exe\n\
    1        2222     C     30     15      0      0      0      0    elsewhere.exe\n";
        let err = parse_pmon_output(s, 1111).unwrap_err();
        assert_eq!(err, GpuUtilProbeError::MultipleGpusUnsupported);
    }

    #[test]
    fn parse_rejects_too_few_columns() {
        // A data line with 9 or fewer tokens would indicate the
        // layout changed; bail out loudly rather than mis-parse.
        let s = "    0       1111     C     20     10      0      0      0      0\n";
        // ^ only 9 tokens after splitting
        let err = parse_pmon_output(s, 1111).unwrap_err();
        assert!(matches!(err, GpuUtilProbeError::ParseError(_)));
    }

    #[test]
    fn parse_rejects_non_numeric_pid() {
        let s = "    0        NOPE     C     20     10      0      0      0      0    app.exe\n";
        let err = parse_pmon_output(s, 1111).unwrap_err();
        assert!(matches!(err, GpuUtilProbeError::ParseError(_)));
    }

    #[test]
    fn parse_ignores_leading_trailing_whitespace() {
        let s = "\n  \n   # header\n    0        1111     C     42     13      0      0      0      0    a.exe\n  \n";
        let snap = parse_pmon_output(s, 1111).expect("must parse");
        assert!((snap.total_fraction - 0.42).abs() < 1e-5);
    }

    #[test]
    fn parse_sum_clamped_to_one() {
        // SM values can sometimes sum above 100 due to sampling
        // artifacts across cores; the fraction must clamp to 1.0
        // so downstream logic stays in a well-defined range.
        let s = "\
    0        1111     C     70     10      0      0      0      0    a.exe\n\
    0        2222     C     80     20      0      0      0      0    b.exe\n";
        let snap = parse_pmon_output(s, 1111).expect("must parse");
        assert_eq!(snap.total_fraction, 1.0); // 0.70 + 0.80 = 1.50 -> clamp
        assert!((snap.self_fraction - 0.70).abs() < 1e-5);
    }

    // --- probe construction / trait-object ---------------------

    #[test]
    fn probe_constructs_without_warmup() {
        // GpuUtilProbe is stateless. Construction is O(1) and must
        // not panic even on machines without nvidia-smi — the
        // subprocess is only spawned in snapshot().
        let _probe = GpuUtilProbe::new();
    }

    #[test]
    fn probe_is_trait_object() {
        let probe = GpuUtilProbe::new();
        let dyn_probe: std::sync::Arc<dyn GpuUtilProbeApi> = std::sync::Arc::new(probe);
        let _ = dyn_probe.snapshot();
    }

    #[test]
    fn probe_returns_values_in_range_on_nvidia_host() {
        // On a host with nvidia-smi the probe should return a well-
        // formed snapshot. On hosts without it the test gracefully
        // skips with a println.
        let probe = GpuUtilProbe::new();
        match probe.snapshot() {
            Ok(snap) => {
                assert!(
                    snap.total_fraction >= 0.0 && snap.total_fraction <= 1.0,
                    "total_fraction out of range: {}",
                    snap.total_fraction
                );
                assert!(
                    snap.self_fraction >= 0.0 && snap.self_fraction <= 1.0,
                    "self_fraction out of range: {}",
                    snap.self_fraction
                );
                assert!(
                    snap.self_fraction <= snap.total_fraction + 1e-5,
                    "self_fraction ({}) exceeds total_fraction ({}) — \
                     attribution bug",
                    snap.self_fraction,
                    snap.total_fraction
                );
            }
            Err(GpuUtilProbeError::NvidiaSmiNotFound) => {
                println!(
                    "[TEST:probe_returns_values_in_range_on_nvidia_host] \
                     nvidia-smi not found -> graceful skip"
                );
            }
            Err(e) => {
                // PmonFailed / ParseError / MultipleGpusUnsupported: any
                // of these on a machine that DOES have nvidia-smi should
                // be investigated. Print the error for diagnostics but
                // do not fail the test on CI hosts that are already
                // skipping via NvidiaSmiNotFound — the above arm handles
                // those. If we got here, nvidia-smi exists but pmon
                // misbehaved; that IS a regression.
                panic!("unexpected probe failure: {:?}", e);
            }
        }
    }
}
