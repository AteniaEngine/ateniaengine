//! Output data model for the hardware probe.
//!
//! Every field is serde-serializable so the same struct drives both the
//! human-readable text output and the machine-readable JSON output. A
//! hand-rolled UTC formatter generates ISO 8601 timestamps without
//! pulling `chrono` or `time` as a dependency.

use serde::Serialize;

/// Top-level report. Produced by [`crate::hw_probe::probe`] and
/// rendered in either text or JSON form by the `hardware_probe`
/// binary.
#[derive(Debug, Serialize)]
pub struct ProbeReport {
    /// Semver of the atenia-engine crate that produced this report.
    /// Surfaces in output so users reporting hardware via an issue
    /// tracker can be matched against a known code version.
    pub probe_version: String,
    /// Unix epoch seconds when the probe ran. JSON consumers can
    /// format as desired; the text renderer formats this as
    /// `probed_at_iso8601` for humans (see
    /// [`format_iso8601_utc`]).
    pub probed_at_unix_secs: u64,
    /// Human-readable ISO 8601 UTC rendering of `probed_at_unix_secs`.
    /// Pre-computed so JSON consumers have the formatted string
    /// without replicating the civil-date algorithm.
    pub probed_at_iso8601: String,
    pub system: SystemInfo,
    pub gpus: Vec<GpuInfo>,
    /// Non-fatal notes emitted during the probe: missing subsystems,
    /// suspicious reported values, platform-specific caveats. Intended
    /// to be read by humans.
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SystemInfo {
    /// Lowercased OS family: `"linux"`, `"windows"`, `"macos"`, or
    /// `"unknown"` if `os_info` could not classify.
    pub os: String,
    /// Detailed human-facing version string, e.g.
    /// `"Windows 11 (24H2, build 26100)"` or `"Ubuntu 24.04"`.
    pub os_version: String,
    /// Target triple's arch component: `"x86_64"`, `"aarch64"`, etc.
    pub arch: String,
    /// Host name as reported by `sysinfo`. May be an empty string if
    /// the platform does not expose one.
    pub hostname: String,
    /// Total physical RAM, megabytes (base 10).
    pub ram_total_mb: u64,
}

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    /// Index within the `gpus` array. Not a vendor-specific device
    /// index — that lives in `vendor_id` / `device_id` (PCI) or in
    /// the NVML ordinal (not exposed here).
    pub index: usize,
    /// Vendor family derived from `vendor_id`, e.g. `"NVIDIA"`. For
    /// unknown vendor IDs the string is `"Unknown"`.
    pub vendor: String,
    /// PCI vendor ID as reported by wgpu's `AdapterInfo`. Examples:
    /// 0x10DE (NVIDIA), 0x1002 (AMD), 0x8086 (Intel), 0x106B (Apple).
    pub vendor_id: u32,
    /// PCI device ID as reported by wgpu's `AdapterInfo`.
    pub device_id: u32,
    /// Human-facing product name, e.g. `"NVIDIA GeForce RTX 4070 Laptop GPU"`.
    pub name: String,
    /// One of `"discrete"`, `"integrated"`, `"virtual"`, `"cpu"`, or
    /// `"other"`.
    pub device_type: String,
    /// Which graphics backend enumerated this adapter: `"vulkan"`,
    /// `"dx12"`, `"metal"`, `"gl"`, `"browser-webgpu"`, or `"unknown"`.
    pub backend: String,
    /// wgpu's `driver` field — short description of the driver
    /// (e.g. `"NVIDIA"`). May be empty on some backends.
    pub driver: Option<String>,
    /// wgpu's `driver_info` field — detailed version string
    /// (e.g. `"560.70"`). May be empty on some backends.
    pub driver_version: Option<String>,
    /// Total VRAM in megabytes. Populated by NVML augmentation on
    /// NVIDIA GPUs; `None` for other vendors until a vendor-specific
    /// augmentation path is added.
    pub vram_mb_total: Option<u64>,
    /// Free VRAM in megabytes at probe time. Populated by NVML on
    /// NVIDIA; `None` otherwise.
    pub vram_mb_free: Option<u64>,
    /// Compute runtimes detected on this GPU. Only populated by
    /// vendor-specific augmentation. Example entry: `"CUDA 13.2"`.
    pub runtime_detected: Vec<String>,
    /// NVIDIA compute capability as `"major.minor"` (e.g. `"8.9"` for
    /// Ada Lovelace). Populated by NVML; `None` for non-NVIDIA GPUs.
    pub compute_capability: Option<String>,
    /// Placeholder for future classification via
    /// `hardware_compatibility.toml`. Always `None` in this version
    /// of the probe.
    pub atenia_support_tier: Option<u8>,
}

/// Map a PCI vendor ID to a human-readable family. Unknown IDs fall
/// through to `"Unknown"`.
pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD", // 0x1022 is AMD proper; 0x1002 is ATI-era ID still used
        0x8086 => "Intel",
        0x106B => "Apple",
        0x5143 => "Qualcomm",
        0x13B5 => "ARM",
        0x1414 => "Microsoft", // Basic Render Driver on Windows (WARP / software)
        0 => "Software",       // wgpu returns vendor=0 for the fallback software adapter
        _ => "Unknown",
    }
}

/// Convert a unix epoch seconds value to an ISO 8601 UTC timestamp
/// (`YYYY-MM-DDTHH:MM:SSZ`). Hand-rolled to avoid adding `chrono` or
/// `time` as a dependency. Uses Howard Hinnant's "days from civil"
/// algorithm, which is exact for the proleptic Gregorian calendar
/// across the full u64 epoch range.
pub fn format_iso8601_utc(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs_in_day = (unix_secs % 86_400) as u32;

    // Howard Hinnant: shift civil epoch to 0000-03-01 to simplify
    // leap-year handling.
    let z = days + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = y + if m <= 2 { 1 } else { 0 };

    let hour = secs_in_day / 3_600;
    let minute = (secs_in_day / 60) % 60;
    let second = secs_in_day % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hour, minute, second
    )
}

impl std::fmt::Display for ProbeReport {
    /// Human-readable rendering. Keep the format stable-ish: tools
    /// that grep for labels like `"GPU 0:"` should keep working
    /// across minor refactors. JSON callers should not parse this
    /// format — use `--output json`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Atenia hardware probe v{}", self.probe_version)?;
        writeln!(f, "Probed at: {} UTC", self.probed_at_iso8601)?;
        writeln!(f)?;

        writeln!(f, "System:")?;
        writeln!(
            f,
            "  OS:       {} ({})",
            self.system.os, self.system.os_version
        )?;
        writeln!(f, "  Arch:     {}", self.system.arch)?;
        writeln!(
            f,
            "  Hostname: {}",
            if self.system.hostname.is_empty() {
                "(not exposed)"
            } else {
                &self.system.hostname
            }
        )?;
        writeln!(f, "  RAM:      {} MB", self.system.ram_total_mb)?;
        writeln!(f)?;

        if self.gpus.is_empty() {
            writeln!(f, "GPUs: none detected")?;
            writeln!(
                f,
                "  (wgpu enumeration returned 0 adapters — either the host truly \
                 has no GPU, or the graphics drivers for Vulkan/DX12/Metal/GL are \
                 missing. On headless Linux servers with CUDA-only drivers this is \
                 expected; install Vulkan tools to get wgpu enumeration.)"
            )?;
        } else {
            writeln!(f, "GPUs detected: {}", self.gpus.len())?;
            for g in &self.gpus {
                writeln!(f)?;
                writeln!(f, "GPU {}:", g.index)?;
                writeln!(f, "  Vendor:   {} (0x{:04X})", g.vendor, g.vendor_id)?;
                writeln!(f, "  Device:   0x{:04X}", g.device_id)?;
                writeln!(f, "  Name:     {}", g.name)?;
                writeln!(f, "  Type:     {}", g.device_type)?;
                writeln!(f, "  Backend:  {}", g.backend)?;
                if let Some(d) = &g.driver {
                    if !d.is_empty() {
                        writeln!(f, "  Driver:   {}", d)?;
                    }
                }
                if let Some(v) = &g.driver_version {
                    if !v.is_empty() {
                        writeln!(f, "  Version:  {}", v)?;
                    }
                }
                if let Some(total) = g.vram_mb_total {
                    let free_str = match g.vram_mb_free {
                        Some(f) => format!(" ({} MB free)", f),
                        None => String::new(),
                    };
                    writeln!(f, "  VRAM:     {} MB{}", total, free_str)?;
                }
                if let Some(cc) = &g.compute_capability {
                    writeln!(f, "  Compute:  {} (NVIDIA CC)", cc)?;
                }
                if !g.runtime_detected.is_empty() {
                    writeln!(f, "  Runtime:  {}", g.runtime_detected.join(", "))?;
                }
                if let Some(tier) = g.atenia_support_tier {
                    writeln!(f, "  Atenia:   tier {}", tier)?;
                } else {
                    writeln!(
                        f,
                        "  Atenia:   tier classification not available \
                         (hardware_compatibility.toml pending)"
                    )?;
                }
            }
        }

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_at_epoch() {
        assert_eq!(format_iso8601_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn iso8601_known_point() {
        // 2023-01-01T00:00:00Z -> 1672531200
        assert_eq!(format_iso8601_utc(1_672_531_200), "2023-01-01T00:00:00Z");
    }

    #[test]
    fn iso8601_leap_day() {
        // 2024-02-29T12:34:56Z -> 1709210096
        assert_eq!(format_iso8601_utc(1_709_210_096), "2024-02-29T12:34:56Z");
    }

    #[test]
    fn vendor_lookup_known() {
        assert_eq!(vendor_name(0x10DE), "NVIDIA");
        assert_eq!(vendor_name(0x1002), "AMD");
        assert_eq!(vendor_name(0x8086), "Intel");
        assert_eq!(vendor_name(0x106B), "Apple");
    }

    #[test]
    fn vendor_lookup_unknown() {
        assert_eq!(vendor_name(0xDEADBEEF), "Unknown");
    }
}
