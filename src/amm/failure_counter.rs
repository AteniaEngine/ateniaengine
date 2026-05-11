//! Time-windowed counter of manually-recorded failure events.
//!
//! Scope: counts only events that callers explicitly record via
//! [`FailureCounter::record_failure`]. It does **not** capture panics,
//! `Result::Err`, or any implicit signal. The counter is per-instance
//! (not global), thread-safe via an internal `Mutex`, and purges
//! expired events lazily on every read.
//!
//! A "poisoned" `Mutex` here implies a panic happened inside a critical
//! section upstream. Rather than propagate that panic further, the
//! read methods return conservative defaults (`0` / `None`) and the
//! write method silently drops the event. The root cause of the
//! poisoning is a bug that must be fixed at its source; this module
//! chooses availability over loudness for the telemetry path.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_WINDOW_SECS: u64 = 60;

pub struct FailureCounter {
    window_duration: Duration,
    events: Mutex<VecDeque<Instant>>,
}

impl FailureCounter {
    /// Creates a counter with the default 60-second window.
    pub fn new() -> Self {
        Self::with_window(Duration::from_secs(DEFAULT_WINDOW_SECS))
    }

    /// Creates a counter with a caller-specified window.
    pub fn with_window(window_duration: Duration) -> Self {
        Self {
            window_duration,
            events: Mutex::new(VecDeque::new()),
        }
    }

    /// Records a failure at the current instant. Silently drops the
    /// event if the internal mutex is poisoned (see module docs).
    pub fn record_failure(&self) {
        if let Ok(mut events) = self.events.lock() {
            let now = Instant::now();
            events.push_back(now);
            purge(&mut events, self.window_duration, now);
        }
    }

    /// Number of failures within the configured window. Lazily purges
    /// expired events. Returns `0` on a poisoned mutex.
    pub fn recent_count(&self) -> u32 {
        let Ok(mut events) = self.events.lock() else {
            return 0;
        };
        let now = Instant::now();
        purge(&mut events, self.window_duration, now);
        events.len() as u32
    }

    /// Time elapsed since the most recent in-window failure. Returns
    /// `None` when there are no failures, when the last failure has
    /// aged out of the window, or when the mutex is poisoned.
    pub fn time_since_last_failure(&self) -> Option<Duration> {
        let Ok(mut events) = self.events.lock() else {
            return None;
        };
        let now = Instant::now();
        purge(&mut events, self.window_duration, now);
        events.back().map(|last| now.duration_since(*last))
    }
}

impl Default for FailureCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Removes entries older than `window` relative to `now` from the front
/// of the deque. Standalone to keep the borrow of `events` contained.
fn purge(events: &mut VecDeque<Instant>, window: Duration, now: Instant) {
    while let Some(&front) = events.front() {
        if now.duration_since(front) > window {
            events.pop_front();
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new_has_zero_count() {
        let c = FailureCounter::new();
        assert_eq!(c.recent_count(), 0);
    }

    #[test]
    fn test_record_increases_count() {
        let c = FailureCounter::new();
        c.record_failure();
        assert_eq!(c.recent_count(), 1);
    }

    #[test]
    fn test_multiple_records() {
        let c = FailureCounter::new();
        for _ in 0..5 {
            c.record_failure();
        }
        assert_eq!(c.recent_count(), 5);
    }

    #[test]
    fn test_old_events_expire() {
        let c = FailureCounter::with_window(Duration::from_millis(50));
        c.record_failure();
        c.record_failure();
        thread::sleep(Duration::from_millis(120));
        assert_eq!(c.recent_count(), 0);
    }

    #[test]
    fn test_time_since_last_none_when_empty() {
        let c = FailureCounter::new();
        assert!(c.time_since_last_failure().is_none());
    }

    #[test]
    fn test_time_since_last_some_after_record() {
        let c = FailureCounter::new();
        c.record_failure();
        let elapsed = c
            .time_since_last_failure()
            .expect("must be Some after record");
        // Elapsed should be very small on a non-hung machine.
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn test_time_since_last_none_after_expiry() {
        let c = FailureCounter::with_window(Duration::from_millis(50));
        c.record_failure();
        thread::sleep(Duration::from_millis(120));
        assert!(c.time_since_last_failure().is_none());
    }
}
