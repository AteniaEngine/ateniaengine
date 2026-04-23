#![allow(dead_code)]

/// Read-only snapshot of conditions that guards can inspect during execution.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardConditions {
    /// Normalized memory pressure in [0.0, 1.0]. Higher means more pressure.
    pub memory_pressure: f32,
    /// Number of recent failures observed in this execution context.
    pub recent_failures: u32,
    /// Whether latency has exceeded an expected range.
    pub latency_spike: bool,
    /// Whether a pre-OOM style signal has been observed.
    pub pre_oom_signal: bool,
    /// M3-e.6: system-wide CPU utilization averaged across all cores,
    /// normalized to `[0.0, 1.0]`. `None` when the CPU probe failed
    /// or the signal bus was built without a probe. Consumers must
    /// treat absence as "unknown — do not gate decisions on CPU"
    /// (fail-open).
    pub cpu_pressure_total: Option<f32>,
    /// M3-e.6: this process's CPU utilization, normalized to
    /// `[0.0, 1.0]` so it is directly comparable with
    /// `cpu_pressure_total` (i.e. `self / total` gives Atenia's
    /// share of the observed CPU load). `None` under the same
    /// conditions as `cpu_pressure_total`.
    pub cpu_pressure_self: Option<f32>,
    /// M3-e.7: total GPU compute utilization (SM %) across all
    /// processes with a compute or graphics context on the device,
    /// normalized to `[0.0, 1.0]`. `None` when the GPU probe was
    /// unavailable (no NVIDIA driver / pmon failed / parsing error).
    /// **Observability-only** in M3-e.7: no current guard or
    /// reaction site gates decisions on this field; it shows up in
    /// enriched `[AMG Guard]` logs and is available for future
    /// milestones (M3-e.11 cascade migration, future GPU-aware
    /// throttling).
    pub gpu_util_total: Option<f32>,
    /// M3-e.7: this process's GPU compute utilization (its own SM %)
    /// normalized to `[0.0, 1.0]`, directly comparable with
    /// `gpu_util_total`. `0.0` (not `None`) when Atenia issued no
    /// GPU work during the sample interval — the process does not
    /// appear in the `pmon` output, which is semantically distinct
    /// from "we do not know Atenia's usage". `None` tracks
    /// `gpu_util_total` — if the probe failed entirely, both fields
    /// are `None`.
    pub gpu_util_self: Option<f32>,
}

impl GuardConditions {
    /// Primary constructor — preserved with the original 4-field
    /// signature so existing callers (tests, integrations) keep
    /// working verbatim. CPU fields default to `None`; callers that
    /// have them populate them via [`with_cpu_pressure`](Self::with_cpu_pressure).
    pub fn new(
        memory_pressure: f32,
        recent_failures: u32,
        latency_spike: bool,
        pre_oom_signal: bool,
    ) -> Self {
        Self {
            memory_pressure,
            recent_failures,
            latency_spike,
            pre_oom_signal,
            cpu_pressure_total: None,
            cpu_pressure_self: None,
            gpu_util_total: None,
            gpu_util_self: None,
        }
    }

    /// Builder-style setter for the CPU-pressure fields. Returns the
    /// modified struct so `SignalBus` can produce conditions in a
    /// single chained expression:
    ///
    /// ```ignore
    /// GuardConditions::new(mp, rf, ls, pre)
    ///     .with_cpu_pressure(total, self_)
    /// ```
    pub fn with_cpu_pressure(mut self, total: f32, self_: f32) -> Self {
        self.cpu_pressure_total = Some(total);
        self.cpu_pressure_self = Some(self_);
        self
    }

    /// Builder-style setter for the GPU-utilization fields (M3-e.7).
    /// Chained after `with_cpu_pressure` / `new` exactly like its
    /// CPU twin.
    pub fn with_gpu_util(mut self, total: f32, self_: f32) -> Self {
        self.gpu_util_total = Some(total);
        self.gpu_util_self = Some(self_);
        self
    }
}
