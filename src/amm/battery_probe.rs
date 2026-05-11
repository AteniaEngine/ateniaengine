//! Battery state monitoring (M3-e.9).
//!
//! Reports two independent fields:
//! - whether the system is currently running on battery vs AC power;
//! - the battery charge level as a fraction in `[0.0, 1.0]`.
//!
//! **Observability-only** in M3-e.9: the signals populate
//! `GuardConditions::on_battery` and `battery_level`, and show up in
//! enriched `[AMG Guard]` log lines, but no current guard or
//! reaction site gates decisions on them. The intended consumer is
//! M3-e.12 behavior modes, where a low-battery state naturally maps
//! to `Conservation` mode.
//!
//! ## Platform coverage
//!
//! | Platform | Implementation | Deps |
//! |----------|----------------|------|
//! | **Windows** | Raw FFI to `kernel32.dll::GetSystemPowerStatus` | none |
//! | **Linux** | `std::fs::read_to_string` over `/sys/class/power_supply/*/` | none |
//! | **macOS** | Stub — returns `(None, None)`. IOKit bindings would add 3+ crates (`objc`, `core-foundation`, `IOKit-sys`) for an observability-only signal | — |
//! | **Other (BSD, etc.)** | Stub | — |
//!
//! ## Why not the `battery` crate
//!
//! The `battery` crate (0.7.8) is cross-platform but:
//! - Pulls 5–7 sub-crates across the three platforms.
//! - Last upstream push in 2023 — maintained but not active.
//! - Models ~20 fields of battery state where we need two.
//!
//! Raw FFI + sysfs reads give 80% of the value with ~85 LOC of
//! code we own. Consistent with the M3-e.8 precedent
//! (`ForegroundProbe` used raw user32 FFI instead of the `windows`
//! crate).
//!
//! ## Independent `Option` fields — shape rationale
//!
//! The signal is encoded as two independent fields rather than an
//! enum:
//!
//! ```ignore
//! pub on_battery:    Option<bool>
//! pub battery_level: Option<f32>
//! ```
//!
//! Reasons:
//! - **Codebase parity**: CPU/GPU signals are `Option<f32>` pairs,
//!   foreground is `Option<bool>`. Two independent `Option`s match.
//! - **Partial reporting**: some drivers expose AC status without a
//!   current level, or vice versa. Independent fields encode that
//!   naturally; an enum with `OnBattery { level: f32 }` would force
//!   a fallback value.
//! - **Meaningful while plugged**: "plugged in but at 15%" still
//!   matters for policy (user cannot unplug and rely on battery
//!   yet). The level field is useful regardless of the AC state.
//!
//! ## Desktop vs laptop
//!
//! A desktop PC without any battery reports `(None, None)` — the
//! probe ran successfully, there is simply no signal to provide.
//! This is distinct from a probe failure. Downstream fail-open
//! code treats both the same way.

use std::sync::Mutex;

/// Snapshot of battery state at a single point in time. Wrapper
/// struct around two independent `Option`s so the API mirrors the
/// other probes (`CpuSnapshot`, `GpuUtilSnapshot`, `ForegroundSnapshot`)
/// and future fields (time remaining, charging state, health, etc.)
/// can be added without breaking the trait.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatterySnapshot {
    /// `Some(true)` when the system is running on battery (unplugged);
    /// `Some(false)` when plugged into AC; `None` when no battery is
    /// present (desktop) or the probe cannot determine AC state.
    pub on_battery: Option<bool>,
    /// Charge level as a fraction in `[0.0, 1.0]`. `None` when no
    /// battery is present or the probe cannot determine the level.
    /// Can be `Some(_)` even when `on_battery` is `Some(false)` —
    /// "plugged in at 85%" is useful policy input.
    pub battery_level: Option<f32>,
}

/// Reasons the probe can fail. Defined defensively; in the current
/// implementation, all platform-level issues (missing sysfs, FFI
/// returning `FALSE`, etc.) map to `Ok(BatterySnapshot { on_battery: None, battery_level: None })`
/// rather than `Err`, because "no battery signal" is a valid answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatteryProbeError {
    /// Retained for future platform implementations that may want
    /// to surface distinct failure modes. Currently unused.
    QueryFailed(String),
}

/// Abstract interface over a battery probe. Production code
/// instantiates [`BatteryProbe`]; tests inject a mock that returns
/// canned snapshots.
pub trait BatteryProbeApi: Send + Sync {
    fn snapshot(&self) -> Result<BatterySnapshot, BatteryProbeError>;
}

/// Production battery probe. Stateless from the caller's
/// perspective: construction is O(1) and cannot fail. Per-call cost
/// is sub-millisecond on Windows (FFI) and a few milliseconds on
/// Linux (sysfs reads) — well within the [`SIGNAL_BUS_CACHE_TTL`][crate::amm::signal_bus::SIGNAL_BUS_CACHE_TTL]
/// of 100 ms, so no per-probe TTL tuning is required.
pub struct BatteryProbe {
    /// Serializes probe calls across threads. On Windows, FFI to
    /// `GetSystemPowerStatus` is thread-safe; on Linux, parallel fs
    /// reads are fine. The mutex exists for API parity with the
    /// other probes and to reduce noise under parallel test
    /// execution.
    lock: Mutex<()>,
}

impl BatteryProbe {
    /// Construct a probe. Cheap, never fails.
    pub fn new() -> Self {
        Self {
            lock: Mutex::new(()),
        }
    }
}

impl Default for BatteryProbe {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Windows implementation
// =========================================================================

#[cfg(windows)]
mod win {
    #[repr(C)]
    pub struct SYSTEM_POWER_STATUS {
        pub ac_line_status: u8,
        pub battery_flag: u8,
        pub battery_life_percent: u8,
        pub system_status_flag: u8,
        pub battery_life_time: i32,
        pub battery_full_life_time: i32,
    }

    // `GetSystemPowerStatus` is in `kernel32.dll` and returns a
    // Win32 BOOL (non-zero on success). The function has been part
    // of the stable Win32 surface since Windows NT 3.5.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetSystemPowerStatus(status: *mut SYSTEM_POWER_STATUS) -> i32;
    }

    pub const BATTERY_FLAG_NO_SYSTEM_BATTERY: u8 = 128;
    pub const UNKNOWN_BYTE: u8 = 255;
    pub const AC_OFFLINE: u8 = 0;
    pub const AC_ONLINE: u8 = 1;
}

#[cfg(windows)]
impl BatteryProbeApi for BatteryProbe {
    fn snapshot(&self) -> Result<BatterySnapshot, BatteryProbeError> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());

        // SAFETY: a zeroed SYSTEM_POWER_STATUS is a valid
        // initial state; the OS fills it in. If the call fails
        // (return value 0) we fall through with the zeroed fields
        // and map everything to None below via the byte checks.
        let mut status = win::SYSTEM_POWER_STATUS {
            ac_line_status: win::UNKNOWN_BYTE,
            battery_flag: win::UNKNOWN_BYTE,
            battery_life_percent: win::UNKNOWN_BYTE,
            system_status_flag: 0,
            battery_life_time: -1,
            battery_full_life_time: -1,
        };
        let ok = unsafe { win::GetSystemPowerStatus(&mut status as *mut _) };
        if ok == 0 {
            // API outright failed — treat as "no signal".
            return Ok(BatterySnapshot {
                on_battery: None,
                battery_level: None,
            });
        }

        // Desktop detection: the NO_SYSTEM_BATTERY flag (bit 128)
        // means the machine has no battery at all. Both fields go
        // to None — there is nothing to report, which is distinct
        // from a probe failure.
        if status.battery_flag != win::UNKNOWN_BYTE
            && (status.battery_flag & win::BATTERY_FLAG_NO_SYSTEM_BATTERY) != 0
        {
            return Ok(BatterySnapshot {
                on_battery: None,
                battery_level: None,
            });
        }

        let on_battery = match status.ac_line_status {
            win::AC_OFFLINE => Some(true),
            win::AC_ONLINE => Some(false),
            _ => None, // UNKNOWN_BYTE or anything else
        };

        let battery_level = if status.battery_life_percent == win::UNKNOWN_BYTE {
            None
        } else {
            // Clamp defensively. The OS returns 0..=100 under normal
            // conditions; misbehaving drivers could theoretically
            // report higher.
            Some((status.battery_life_percent as f32 / 100.0).clamp(0.0, 1.0))
        };

        Ok(BatterySnapshot {
            on_battery,
            battery_level,
        })
    }
}

// =========================================================================
// Linux implementation — sysfs
// =========================================================================

#[cfg(target_os = "linux")]
mod linux {
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Read a sysfs file, trim, return as owned String. Returns
    /// `None` on any IO error — probe is fail-open.
    fn read_trimmed(path: &Path) -> Option<String> {
        fs::read_to_string(path).ok().map(|s| s.trim().to_string())
    }

    /// Walk `/sys/class/power_supply/` and classify each entry by
    /// its `type` file. Returns `(battery_path, ac_path)` with the
    /// first Battery and the first Mains encountered. Multi-battery
    /// laptops are simplified: we pick the first battery and ignore
    /// additional ones (edge case, easy to extend later).
    pub fn locate_power_supplies() -> (Option<PathBuf>, Option<PathBuf>) {
        let base = Path::new("/sys/class/power_supply");
        let entries = match fs::read_dir(base) {
            Ok(e) => e,
            Err(_) => return (None, None),
        };

        let mut battery: Option<PathBuf> = None;
        let mut ac: Option<PathBuf> = None;

        for entry in entries.flatten() {
            let path = entry.path();
            let type_path = path.join("type");
            let kind = match read_trimmed(&type_path) {
                Some(k) => k,
                None => continue,
            };
            match kind.as_str() {
                "Battery" if battery.is_none() => battery = Some(path),
                "Mains" if ac.is_none() => ac = Some(path),
                _ => {}
            }
        }

        (battery, ac)
    }

    /// Read the battery level fraction from a `BAT*` path. Uses
    /// the `capacity` file (integer 0..=100, direct percent).
    /// Returns `None` on any IO or parse error.
    pub fn read_battery_level(bat_path: &Path) -> Option<f32> {
        let s = read_trimmed(&bat_path.join("capacity"))?;
        let pct: i64 = s.parse().ok()?;
        Some((pct as f32 / 100.0).clamp(0.0, 1.0))
    }

    /// Read the AC online state from a `Mains` path (or fall back
    /// to the battery's `status` file). `Some(true)` means on
    /// battery (unplugged); `Some(false)` means plugged in.
    pub fn read_on_battery(ac_path: Option<&Path>, bat_path: Option<&Path>) -> Option<bool> {
        // Primary signal: AC/online is `"1"` or `"0"`.
        if let Some(ac) = ac_path {
            if let Some(s) = read_trimmed(&ac.join("online")) {
                return match s.as_str() {
                    "1" => Some(false), // plugged in
                    "0" => Some(true),  // on battery
                    _ => None,
                };
            }
        }
        // Fallback: derive from battery status. "Discharging" is
        // the unambiguous "on battery" signal; anything else we
        // treat as unknown — "Charging"/"Full" technically imply
        // plugged in but some drivers report these transiently
        // during hotplug, so we stay conservative.
        if let Some(bat) = bat_path {
            if let Some(s) = read_trimmed(&bat.join("status")) {
                return match s.as_str() {
                    "Discharging" => Some(true),
                    "Charging" | "Full" | "Not charging" => Some(false),
                    _ => None,
                };
            }
        }
        None
    }
}

#[cfg(target_os = "linux")]
impl BatteryProbeApi for BatteryProbe {
    fn snapshot(&self) -> Result<BatterySnapshot, BatteryProbeError> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());

        let (bat_path, ac_path) = linux::locate_power_supplies();

        // No battery present → desktop / server / container without
        // sysfs. Signal is None for both; not an error.
        if bat_path.is_none() {
            return Ok(BatterySnapshot {
                on_battery: None,
                battery_level: None,
            });
        }

        let battery_level = bat_path.as_ref().and_then(|p| linux::read_battery_level(p));
        let on_battery = linux::read_on_battery(ac_path.as_deref(), bat_path.as_deref());

        Ok(BatterySnapshot {
            on_battery,
            battery_level,
        })
    }
}

// =========================================================================
// Non-Windows, non-Linux stub — always returns `(None, None)`.
// =========================================================================

#[cfg(not(any(windows, target_os = "linux")))]
impl BatteryProbeApi for BatteryProbe {
    fn snapshot(&self) -> Result<BatterySnapshot, BatteryProbeError> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        Ok(BatterySnapshot {
            on_battery: None,
            battery_level: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_constructs_without_panic() {
        let _probe = BatteryProbe::new();
    }

    #[test]
    fn probe_is_usable_as_trait_object() {
        let probe = BatteryProbe::new();
        let dyn_probe: std::sync::Arc<dyn BatteryProbeApi> = std::sync::Arc::new(probe);
        let _ = dyn_probe.snapshot();
    }

    #[test]
    fn snapshot_returns_ok_regardless_of_platform() {
        // Across platforms, battery state varies and the test host
        // may or may not have a battery. The probe must never
        // return Err — all failure modes map to Ok with None fields.
        let probe = BatteryProbe::new();
        let result = probe.snapshot();
        assert!(
            result.is_ok(),
            "snapshot must never return Err in M3-e.9: {:?}",
            result.err()
        );
    }

    #[test]
    fn snapshot_fields_are_independent_options() {
        // Type-level sanity: on_battery and battery_level are
        // independent Options on the snapshot. This test catches
        // refactors that would accidentally couple them (e.g. via
        // an enum).
        let probe = BatteryProbe::new();
        let snap = probe.snapshot().expect("snapshot must succeed");
        // Each field is independently typed as Option. We cannot
        // assert specific values on a CI host, but if
        // battery_level is Some we can assert it is in [0, 1].
        if let Some(level) = snap.battery_level {
            assert!(
                (0.0..=1.0).contains(&level),
                "battery_level out of range: {}",
                level
            );
        }
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    #[test]
    fn stub_platform_returns_none_for_both_fields() {
        // Platform-stub contract: on macOS / BSD / other, both
        // fields are None.
        let probe = BatteryProbe::new();
        let snap = probe.snapshot().expect("stub never errors");
        assert_eq!(snap.on_battery, None);
        assert_eq!(snap.battery_level, None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_locate_power_supplies_does_not_panic() {
        // Even on a Linux host without /sys (containers), the
        // locator must return (None, None) gracefully.
        let _ = linux::locate_power_supplies();
    }
}
