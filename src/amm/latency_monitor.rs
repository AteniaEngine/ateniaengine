//! Time-series latency monitor with P50 baseline and spike detection.
//!
//! Scope: records only events that callers explicitly submit via
//! [`LatencyMonitor::record_latency`]. It does **not** capture execution
//! timings automatically. Per-instance (not global). Thread-safe via
//! internal `Mutex`es; poisoning is handled conservatively — writes
//! silently drop events, reads return safe defaults.
//!
//! A measurement is classified as a spike when it exceeds the P50 of
//! prior in-window measurements by a multiplicative factor. Spike
//! detection is skipped when prior history has fewer than `min_samples`
//! measurements (not enough data for a reliable baseline).

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_BASELINE_WINDOW_SECS: u64 = 60;
const DEFAULT_SPIKE_RECENCY_SECS: u64 = 30;
const DEFAULT_SPIKE_MULTIPLIER: f64 = 2.0;
const DEFAULT_MIN_SAMPLES: u32 = 10;

#[derive(Clone, Copy)]
struct LatencyEvent {
    timestamp: Instant,
    duration: Duration,
}

pub struct LatencyMonitor {
    baseline_window: Duration,
    spike_recency: Duration,
    spike_multiplier: f64,
    min_samples: u32,
    events: Mutex<VecDeque<LatencyEvent>>,
    spike_events: Mutex<VecDeque<Instant>>,
}

impl LatencyMonitor {
    /// Creates a monitor with the default configuration
    /// (baseline 60s, spike-recency 30s, multiplier 2.0, min_samples 10).
    pub fn new() -> Self {
        Self::with_config(
            Duration::from_secs(DEFAULT_BASELINE_WINDOW_SECS),
            Duration::from_secs(DEFAULT_SPIKE_RECENCY_SECS),
            DEFAULT_SPIKE_MULTIPLIER,
            DEFAULT_MIN_SAMPLES,
        )
    }

    /// Creates a monitor with caller-specified configuration.
    pub fn with_config(
        baseline_window: Duration,
        spike_recency: Duration,
        spike_multiplier: f64,
        min_samples: u32,
    ) -> Self {
        Self {
            baseline_window,
            spike_recency,
            spike_multiplier,
            min_samples,
            events: Mutex::new(VecDeque::new()),
            spike_events: Mutex::new(VecDeque::new()),
        }
    }

    /// Records a latency measurement. Compares against the P50 of prior
    /// in-window measurements; if `duration` exceeds baseline by the
    /// configured multiplier, a spike is recorded. No spike is recorded
    /// when prior history has fewer than `min_samples` entries.
    ///
    /// Silently drops the event if the events mutex is poisoned.
    pub fn record_latency(&self, duration: Duration) {
        let now = Instant::now();

        // Lock events, compute baseline from *prior* history, then push.
        let baseline = {
            let Ok(mut events) = self.events.lock() else {
                return;
            };
            purge_events(&mut events, self.baseline_window, now);
            let baseline_opt = if (events.len() as u32) >= self.min_samples {
                Some(median_duration(&events))
            } else {
                None
            };
            events.push_back(LatencyEvent {
                timestamp: now,
                duration,
            });
            baseline_opt
        };

        if let Some(b) = baseline {
            if duration > b.mul_f64(self.spike_multiplier) {
                if let Ok(mut spikes) = self.spike_events.lock() {
                    purge_instants(&mut spikes, self.spike_recency, now);
                    spikes.push_back(now);
                }
            }
        }
    }

    /// Returns `true` if any spike has been recorded within the
    /// `spike_recency` window. Purges expired spikes lazily.
    pub fn has_recent_spike(&self) -> bool {
        let Ok(mut spikes) = self.spike_events.lock() else {
            return false;
        };
        let now = Instant::now();
        purge_instants(&mut spikes, self.spike_recency, now);
        !spikes.is_empty()
    }

    /// Returns `true` when both:
    /// - There is no spike within the `spike_recency` window.
    /// - At least `min_samples` measurements sit in the baseline window.
    pub fn has_stable_latency(&self) -> bool {
        if self.has_recent_spike() {
            return false;
        }
        self.sample_count() >= self.min_samples
    }

    /// Returns the P50 (median) of in-window measurements, or `None`
    /// when there are fewer than `min_samples`.
    pub fn baseline_p50(&self) -> Option<Duration> {
        let Ok(mut events) = self.events.lock() else {
            return None;
        };
        let now = Instant::now();
        purge_events(&mut events, self.baseline_window, now);
        if (events.len() as u32) < self.min_samples {
            return None;
        }
        Some(median_duration(&events))
    }

    /// Number of measurements in the baseline window after purging.
    /// Returns `0` on a poisoned mutex.
    pub fn sample_count(&self) -> u32 {
        let Ok(mut events) = self.events.lock() else {
            return 0;
        };
        let now = Instant::now();
        purge_events(&mut events, self.baseline_window, now);
        events.len() as u32
    }
}

impl Default for LatencyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

fn purge_events(events: &mut VecDeque<LatencyEvent>, window: Duration, now: Instant) {
    while let Some(front) = events.front() {
        if now.duration_since(front.timestamp) > window {
            events.pop_front();
        } else {
            break;
        }
    }
}

fn purge_instants(instants: &mut VecDeque<Instant>, window: Duration, now: Instant) {
    while let Some(&front) = instants.front() {
        if now.duration_since(front) > window {
            instants.pop_front();
        } else {
            break;
        }
    }
}

/// Simple median by sorting durations. For windows of 10–100 samples
/// the O(n log n) cost is negligible. Returns the upper middle for
/// even counts.
fn median_duration(events: &VecDeque<LatencyEvent>) -> Duration {
    let mut durs: Vec<Duration> = events.iter().map(|e| e.duration).collect();
    durs.sort();
    durs[durs.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new_has_no_spike() {
        let m = LatencyMonitor::new();
        assert!(!m.has_recent_spike());
    }

    #[test]
    fn test_new_has_no_stable_latency() {
        let m = LatencyMonitor::new();
        assert!(!m.has_stable_latency());
    }

    #[test]
    fn test_sample_count_increases() {
        let m = LatencyMonitor::new();
        for _ in 0..5 {
            m.record_latency(Duration::from_millis(50));
        }
        assert_eq!(m.sample_count(), 5);
    }

    #[test]
    fn test_baseline_none_below_min_samples() {
        let m = LatencyMonitor::new();
        for _ in 0..5 {
            m.record_latency(Duration::from_millis(50));
        }
        assert!(m.baseline_p50().is_none());
    }

    #[test]
    fn test_baseline_some_above_min_samples() {
        let m = LatencyMonitor::new();
        for _ in 0..15 {
            m.record_latency(Duration::from_millis(50));
        }
        assert!(m.baseline_p50().is_some());
    }

    #[test]
    fn test_normal_latency_no_spike() {
        let m = LatencyMonitor::new();
        for _ in 0..15 {
            m.record_latency(Duration::from_millis(50));
        }
        assert!(!m.has_recent_spike());
    }

    #[test]
    fn test_obvious_spike_detected() {
        let m = LatencyMonitor::new();
        for _ in 0..15 {
            m.record_latency(Duration::from_millis(50));
        }
        m.record_latency(Duration::from_millis(500));
        assert!(m.has_recent_spike());
    }

    #[test]
    fn test_stable_when_no_spikes_and_enough_samples() {
        let m = LatencyMonitor::new();
        for _ in 0..15 {
            m.record_latency(Duration::from_millis(50));
        }
        assert!(m.has_stable_latency());
    }

    #[test]
    fn test_not_stable_with_recent_spike() {
        let m = LatencyMonitor::new();
        for _ in 0..15 {
            m.record_latency(Duration::from_millis(50));
        }
        m.record_latency(Duration::from_millis(500));
        assert!(!m.has_stable_latency());
    }

    #[test]
    fn test_spike_expires() {
        // Long baseline (keeps samples alive), short spike recency.
        let m = LatencyMonitor::with_config(
            Duration::from_secs(5),
            Duration::from_millis(50),
            2.0,
            10,
        );
        for _ in 0..15 {
            m.record_latency(Duration::from_millis(50));
        }
        m.record_latency(Duration::from_millis(500));
        assert!(m.has_recent_spike(), "spike must be present immediately");
        thread::sleep(Duration::from_millis(120));
        assert!(!m.has_recent_spike(), "spike must expire after window");
    }

    #[test]
    fn test_spike_ignored_below_min_samples() {
        let m = LatencyMonitor::with_config(
            Duration::from_secs(5),
            Duration::from_secs(5),
            2.0,
            10,
        );
        // Only 5 prior samples — below min_samples of 10.
        for _ in 0..5 {
            m.record_latency(Duration::from_millis(50));
        }
        m.record_latency(Duration::from_millis(500));
        assert!(
            !m.has_recent_spike(),
            "spike must be ignored when prior history < min_samples"
        );
    }
}
