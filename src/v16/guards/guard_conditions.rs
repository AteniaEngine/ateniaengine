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
}

impl GuardConditions {
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
        }
    }
}
