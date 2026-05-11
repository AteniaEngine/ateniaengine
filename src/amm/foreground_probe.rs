//! Foreground application detection (M3-e.8).
//!
//! Reports whether the OS foreground (the window the user is
//! actively interacting with) belongs to this process or to another.
//! The signal is **observability-only** in M3-e.8: it populates
//! `GuardConditions::foreground_is_atenia` and shows up in
//! `[AMG Guard]` log lines, but no current guard or reaction site
//! gates decisions on it. Future milestones — especially M3-e.12
//! behavior modes — will consume this signal together with CPU / GPU
//! pressure / battery state to drive discrete-mode transitions.
//!
//! ## Platform coverage — first pass
//!
//! | Platform | Status | Rationale |
//! |----------|--------|-----------|
//! | **Windows** | implemented via raw FFI to `user32.dll` (`GetForegroundWindow` + `GetWindowThreadProcessId`) | zero new crate dependencies; Win32 API is stable and rock-solid since Windows 95 |
//! | Linux X11 | `None` (deferred) | would require either a new crate (`x11rb`) or ~150 LOC of `libloading`-based X protocol code with atom interning and `_NET_ACTIVE_WINDOW` parsing; candidate for a future follow-up milestone |
//! | Linux Wayland | `None` (structural) | Wayland has no portable foreground protocol — each compositor (GNOME, KDE, wlroots, Hyprland, ...) exposes its own D-Bus or protocol-extension API. Cross-compositor support is a milestone in its own right, not a sub-bullet of e.8 |
//! | macOS | `None` (deferred) | requires `objc` + `cocoa` or `core-foundation` bindings (3+ new crates); Mac is not a primary dev platform for Atenia today |
//!
//! Non-supported platforms return `Ok(ForegroundSnapshot { foreground_is_atenia: None })`
//! — **not** an `Err`. The probe is not broken on those platforms;
//! it just does not have a signal to provide. The fail-open policy
//! in `SignalBus::collect_probes` then nulls out the corresponding
//! field on `GuardConditions` and downstream code treats absence as
//! "unknown — do not gate decisions on foreground state".
//!
//! ## Semantics
//!
//! - `Some(true)`  → foreground window belongs to this process.
//! - `Some(false)` → foreground window belongs to a different process.
//! - `None`        → cannot determine. Reasons include: non-Windows
//!   platforms in this first pass, Windows screen lock / UAC in
//!   transition / no window is foreground, FFI call returned 0.
//!
//! The distinction between `Some(false)` and `None` is preserved
//! deliberately. "Another app is foreground" and "we don't know who
//! is foreground" are different states for a policy that has to
//! decide whether to throttle.

#[cfg(windows)]
use std::sync::Mutex;

/// Snapshot of the foreground-application state at a single point
/// in time. Wrapper struct around a single `Option<bool>` so the
/// API mirrors the other probes (`CpuSnapshot`, `GpuUtilSnapshot`)
/// and future fields (e.g. foreground-app name, active-window class)
/// can be added without breaking the trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForegroundSnapshot {
    /// `Some(true)` when the OS foreground matches this process's
    /// PID; `Some(false)` when it matches another process; `None`
    /// when the probe cannot decide (unsupported platform, screen
    /// locked, FFI error, no foreground).
    pub foreground_is_atenia: Option<bool>,
}

/// Reasons the probe can fail. A "failure" is reserved for cases
/// where the probe is structurally unable to run — not for the
/// case where it ran but produced no signal (that is a successful
/// `Ok(ForegroundSnapshot { foreground_is_atenia: None })`). In
/// the Windows first pass there is no failure mode that maps here,
/// so the variant is defined defensively for future platforms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForegroundProbeError {
    /// Retained for future platform implementations. Currently
    /// unused — Windows FFI errors are downgraded to `None` rather
    /// than `Err`, because "no foreground" is a valid answer.
    QueryFailed(String),
}

/// Abstract interface over a foreground probe. Production code
/// instantiates [`ForegroundProbe`]; tests inject a mock that
/// returns canned snapshots (see `tests/m3_e_8_*`).
pub trait ForegroundProbeApi: Send + Sync {
    fn snapshot(&self) -> Result<ForegroundSnapshot, ForegroundProbeError>;
}

/// Production foreground probe. Stateless from the caller's
/// perspective: `new()` only caches `std::process::id()`, so
/// construction is O(1) and cannot fail. On non-Windows platforms
/// the struct is still present; its `snapshot` simply returns
/// `Ok(ForegroundSnapshot { foreground_is_atenia: None })`.
pub struct ForegroundProbe {
    own_pid: u32,
    /// Serializes FFI calls across threads. Win32 `GetForegroundWindow`
    /// is itself thread-safe, but serializing keeps parallel tests
    /// from exercising driver / user32 edge cases that are not our
    /// concern to debug.
    #[cfg(windows)]
    lock: Mutex<()>,
}

impl ForegroundProbe {
    /// Construct a probe. Cheap, never fails.
    pub fn new() -> Self {
        Self {
            own_pid: std::process::id(),
            #[cfg(windows)]
            lock: Mutex::new(()),
        }
    }
}

impl Default for ForegroundProbe {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Windows implementation
// =========================================================================

#[cfg(windows)]
mod win {
    use std::os::raw::c_void;
    pub type HWND = *mut c_void;
    pub type DWORD = u32;

    // Link to user32.dll and expose the two Win32 functions we need.
    // Both have been part of the stable Win32 surface since NT 3.1
    // / Windows 95 respectively; `extern "system"` is the correct
    // calling convention on both x86 (stdcall) and x64 (sysv-ish).
    #[link(name = "user32")]
    unsafe extern "system" {
        pub fn GetForegroundWindow() -> HWND;
        pub fn GetWindowThreadProcessId(hwnd: HWND, lpdw_process_id: *mut DWORD) -> DWORD;
    }
}

#[cfg(windows)]
impl ForegroundProbeApi for ForegroundProbe {
    fn snapshot(&self) -> Result<ForegroundSnapshot, ForegroundProbeError> {
        // Best-effort serialization — poisoned lock is recovered
        // rather than propagated (same policy as the other probes).
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());

        // GetForegroundWindow returns NULL in several expected
        // situations (screen lock, UAC prompt in transition, the
        // taskbar being "focused"). Treat NULL as "no foreground"
        // rather than as an error — the caller's downstream logic
        // is already prepared to interpret `None` as "unknown".
        let hwnd = unsafe { win::GetForegroundWindow() };
        if hwnd.is_null() {
            return Ok(ForegroundSnapshot {
                foreground_is_atenia: None,
            });
        }

        let mut foreground_pid: win::DWORD = 0;
        // Return value is the thread id (irrelevant for us); the
        // PID comes out via the out-param. A return of 0 indicates
        // failure — map to `None`.
        let tid = unsafe { win::GetWindowThreadProcessId(hwnd, &mut foreground_pid as *mut _) };
        if tid == 0 {
            return Ok(ForegroundSnapshot {
                foreground_is_atenia: None,
            });
        }

        Ok(ForegroundSnapshot {
            foreground_is_atenia: Some(foreground_pid == self.own_pid),
        })
    }
}

// =========================================================================
// Non-Windows stub — always returns `Ok(None)`.
// =========================================================================

#[cfg(not(windows))]
impl ForegroundProbeApi for ForegroundProbe {
    fn snapshot(&self) -> Result<ForegroundSnapshot, ForegroundProbeError> {
        // Unused field suppression — on non-Windows platforms we do
        // not need the PID but keep it in the struct for API parity.
        let _ = self.own_pid;
        Ok(ForegroundSnapshot {
            foreground_is_atenia: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_constructs_without_panic() {
        let _probe = ForegroundProbe::new();
    }

    #[test]
    fn probe_is_usable_as_trait_object() {
        let probe = ForegroundProbe::new();
        let dyn_probe: std::sync::Arc<dyn ForegroundProbeApi> = std::sync::Arc::new(probe);
        let _ = dyn_probe.snapshot();
    }

    #[test]
    fn snapshot_returns_ok_regardless_of_platform() {
        // On Windows, snapshot returns Ok(Some(true/false)) or
        // Ok(None) depending on what Win32 reports. On non-Windows,
        // it unconditionally returns Ok(None). Either way, no Err
        // and no panic — the fail-open contract.
        let probe = ForegroundProbe::new();
        let result = probe.snapshot();
        assert!(
            result.is_ok(),
            "snapshot must never return Err in M3-e.8: {:?}",
            result.err()
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn non_windows_snapshot_is_none() {
        // Platform stub contract: non-Windows always returns None.
        let probe = ForegroundProbe::new();
        let snap = probe.snapshot().expect("stub never errors");
        assert_eq!(snap.foreground_is_atenia, None);
    }

    #[cfg(windows)]
    #[test]
    fn windows_snapshot_returns_bool_or_none() {
        // On Windows the snapshot's Option can be either variant
        // depending on what is in foreground at test time. We only
        // assert that the value is well-typed (trivially true via
        // the type system) and that two consecutive calls are
        // consistent over a short window (they should see the same
        // foreground unless the user tabs away during the test).
        let probe = ForegroundProbe::new();
        let a = probe.snapshot().expect("snapshot must succeed");
        let b = probe.snapshot().expect("snapshot must succeed");
        assert_eq!(
            a.foreground_is_atenia, b.foreground_is_atenia,
            "consecutive snapshots on Windows should agree unless foreground \
             changes between them (a={:?}, b={:?})",
            a, b
        );
    }
}
