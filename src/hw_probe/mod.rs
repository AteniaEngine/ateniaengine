//! Hardware-probe module.
//!
//! Entry point: [`probe`] — returns a [`ProbeReport`] with system
//! info, every GPU enumerated via wgpu, and NVIDIA-specific details
//! augmented via NVML when available.
//!
//! The module is feature-gated behind `hw-probe` in `Cargo.toml`;
//! normal library builds do not pull wgpu or NVML. See
//! `docs/HARDWARE_PROBE.md` for usage and the investigation note in
//! `docs/RESEARCH_INTEL_APIS.md` for background on multi-vendor API
//! choices.

pub mod nvml_augment;
pub mod report;
pub mod wgpu_probe;

pub use report::{GpuInfo, ProbeReport, SystemInfo};

use std::time::{SystemTime, UNIX_EPOCH};
use sysinfo::System;

/// Run the probe. Never panics; all fallible subsystems produce
/// warnings attached to the report rather than bubbling errors.
pub fn probe() -> ProbeReport {
    let mut warnings: Vec<String> = Vec::new();

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_else(|_| {
            warnings.push(
                "system clock reports a time before the unix epoch; \
                 probed_at_unix_secs set to 0."
                    .to_string(),
            );
            0
        });

    let system = collect_system_info();
    let mut gpus = wgpu_probe::enumerate(&mut warnings);
    nvml_augment::augment(&mut gpus, &mut warnings);

    ProbeReport {
        probe_version: env!("CARGO_PKG_VERSION").to_string(),
        probed_at_unix_secs: now_unix,
        probed_at_iso8601: report::format_iso8601_utc(now_unix),
        system,
        gpus,
        warnings,
    }
}

fn collect_system_info() -> SystemInfo {
    // `sysinfo 0.30` returns memory in bytes from `total_memory`.
    let mut sys = System::new();
    sys.refresh_memory();
    let ram_total_mb = sys.total_memory() / 1_000_000;

    let info = os_info::get();

    let os = match info.os_type() {
        os_info::Type::Linux
        | os_info::Type::Alpine
        | os_info::Type::Amazon
        | os_info::Type::Arch
        | os_info::Type::CentOS
        | os_info::Type::Debian
        | os_info::Type::EndeavourOS
        | os_info::Type::Fedora
        | os_info::Type::Gentoo
        | os_info::Type::Manjaro
        | os_info::Type::Mariner
        | os_info::Type::NixOS
        | os_info::Type::OpenCloudOS
        | os_info::Type::openEuler
        | os_info::Type::openSUSE
        | os_info::Type::OracleLinux
        | os_info::Type::Pop
        | os_info::Type::Raspbian
        | os_info::Type::RedHatEnterprise
        | os_info::Type::Redhat
        | os_info::Type::Solus
        | os_info::Type::SUSE
        | os_info::Type::Ubuntu
        | os_info::Type::Mint
        | os_info::Type::Garuda
        | os_info::Type::Kali
        | os_info::Type::AlmaLinux
        | os_info::Type::Android => "linux",
        os_info::Type::Windows => "windows",
        os_info::Type::Macos => "macos",
        _ => "unknown",
    }
    .to_string();

    let os_version = format!("{} {}", info.os_type(), info.version());

    // `std::env::consts::ARCH` is the compile-time target arch. For a
    // probe that has to be built on the host anyway, this is accurate
    // and does not require a runtime CPUID query.
    let arch = std::env::consts::ARCH.to_string();

    let hostname = System::host_name().unwrap_or_default();

    SystemInfo {
        os,
        os_version,
        arch,
        hostname,
        ram_total_mb,
    }
}
