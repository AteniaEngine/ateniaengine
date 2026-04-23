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
    /// M3-e.8: foreground-application indicator. `Some(true)` when
    /// the OS foreground window belongs to this process;
    /// `Some(false)` when it belongs to another process; `None`
    /// when the probe cannot determine (non-supported platform,
    /// screen locked, no foreground). **Observability-only in
    /// M3-e.8**: no reaction site gates decisions on this field
    /// today. Future milestones (M3-e.12 behavior modes) will
    /// consume it to distinguish `UserActive` from `SoloMachine`.
    pub foreground_is_atenia: Option<bool>,
    /// M3-e.9: `Some(true)` when the system is running on battery
    /// (unplugged); `Some(false)` when plugged into AC; `None`
    /// when no battery is present (desktop) or the probe cannot
    /// determine AC state. Independent of `battery_level` — some
    /// drivers expose one but not the other.
    pub on_battery: Option<bool>,
    /// M3-e.9: battery charge level as a fraction in `[0.0, 1.0]`.
    /// `None` when no battery is present or the probe cannot
    /// determine the level. Can be `Some(_)` even when
    /// `on_battery` is `Some(false)` — "plugged in at 15%"
    /// remains useful policy input. **Observability-only in
    /// M3-e.9**: the signal feeds `GuardConditions` and logs;
    /// M3-e.12 `Conservation` mode will be the first consumer
    /// that gates decisions on it.
    pub battery_level: Option<f32>,
    /// M3-e.10: P50 baseline of recent node execution latencies,
    /// in milliseconds. `None` when the `LatencyMonitor` has fewer
    /// than `min_samples` measurements in window (cold baseline).
    pub latency_baseline_ms: Option<f32>,
    /// M3-e.10: EWMA of recent node execution latencies, in
    /// milliseconds. `None` with the same cold-baseline criterion
    /// as `latency_baseline_ms`.
    pub latency_current_ms: Option<f32>,
    /// M3-e.10: derived ratio `latency_current_ms / latency_baseline_ms`.
    /// Values above 1.0 indicate Atenia is running slower than its
    /// own recent baseline — the most honest signal that *something*
    /// is applying pressure, regardless of which probe would
    /// otherwise report it. `None` when either input is `None`.
    /// **Observability-only in M3-e.10**: no reaction site gates
    /// decisions on this field; future milestones (M3-e.12 behavior
    /// modes) can consume it as a primary trigger.
    pub latency_ratio: Option<f32>,
    /// M3-e.11.3: VRAM pressure as a fraction in `[0.0, 1.0]` on the
    /// single NVIDIA GPU detected. `None` when the probe failed
    /// (nvidia-smi missing, driver error, parse failure) or was
    /// disabled at bus construction time. The legacy aggregate
    /// `memory_pressure` field is preserved as `max(vram, ram)` for
    /// backwards compatibility; consumers that need the
    /// discrimination (M3-e.11.5 dual-pressure promotion logic) read
    /// this field directly.
    pub vram_pressure: Option<f32>,
    /// M3-e.11.3: system RAM pressure as a fraction in `[0.0, 1.0]`.
    /// Same `None` semantics as `vram_pressure`: probe failure or
    /// disabled. Independent of `vram_pressure` — a partial reading
    /// where only one tier's probe works is representable and
    /// propagates to downstream consumers.
    pub ram_pressure: Option<f32>,
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
            foreground_is_atenia: None,
            on_battery: None,
            battery_level: None,
            latency_baseline_ms: None,
            latency_current_ms: None,
            latency_ratio: None,
            vram_pressure: None,
            ram_pressure: None,
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

    /// Builder-style setter for the foreground indicator (M3-e.8).
    /// Takes a concrete `bool`; callers that want to leave the
    /// field `None` simply do not call this method. This matches
    /// the semantics of `with_cpu_pressure` / `with_gpu_util`,
    /// where absence is the default.
    pub fn with_foreground(mut self, is_atenia: bool) -> Self {
        self.foreground_is_atenia = Some(is_atenia);
        self
    }

    /// Builder-style setter for the battery on/off-AC indicator
    /// (M3-e.9). Independent of `with_battery_level` — the two
    /// fields are granular by design so `SignalBus` can populate
    /// whichever it observed without forcing a fabricated value
    /// for the other.
    pub fn with_on_battery(mut self, on_battery: bool) -> Self {
        self.on_battery = Some(on_battery);
        self
    }

    /// Builder-style setter for the battery level (M3-e.9). Input
    /// clamped defensively to `[0.0, 1.0]`.
    pub fn with_battery_level(mut self, level: f32) -> Self {
        self.battery_level = Some(level.clamp(0.0, 1.0));
        self
    }

    /// Builder-style setter for the three latency-context fields
    /// (M3-e.10). The three fields come from the same
    /// `LatencyMonitor` snapshot and always travel together — a
    /// single setter prevents half-populated states where baseline
    /// is known but current is not, or vice versa.
    ///
    /// The `ratio` parameter is computed by the caller so the
    /// formula (`current / baseline` vs alternatives such as
    /// `(current - baseline) / baseline`) stays centralized at the
    /// call site that knows the semantics.
    pub fn with_latency_context(
        mut self,
        baseline_ms: f32,
        current_ms: f32,
        ratio: f32,
    ) -> Self {
        self.latency_baseline_ms = Some(baseline_ms);
        self.latency_current_ms = Some(current_ms);
        self.latency_ratio = Some(ratio);
        self
    }

    /// Builder-style setter for the VRAM pressure (M3-e.11.3).
    /// Independent from [`Self::with_ram_pressure`] — the two tiers
    /// can be populated in any combination. Input clamped to
    /// `[0.0, 1.0]`.
    pub fn with_vram_pressure(mut self, pressure: f32) -> Self {
        self.vram_pressure = Some(pressure.clamp(0.0, 1.0));
        self
    }

    /// Builder-style setter for the system-RAM pressure (M3-e.11.3).
    /// Independent from [`Self::with_vram_pressure`]. Input clamped
    /// to `[0.0, 1.0]`.
    pub fn with_ram_pressure(mut self, pressure: f32) -> Self {
        self.ram_pressure = Some(pressure.clamp(0.0, 1.0));
        self
    }
}
