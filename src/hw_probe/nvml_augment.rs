//! NVIDIA-specific augmentation via NVML.
//!
//! wgpu gives us name / vendor / device IDs for every adapter the
//! graphics stack exposes, but it does not expose physical VRAM,
//! compute capability, or the CUDA driver version. NVML (NVIDIA
//! Management Library) fills those gaps for NVIDIA GPUs.
//!
//! Matching NVML devices to wgpu adapters: NVML's `pci_info` exposes
//! `pci_device_id: u32`, which packs `(device_id << 16) | vendor_id`
//! — the same two u16 fields that wgpu's `AdapterInfo` exposes
//! separately. We match by (vendor_id, device_id) pair; this works
//! even on multi-GPU hosts where NVML ordinal and wgpu ordinal
//! disagree.
//!
//! Failure mode: if NVML cannot initialize (driver missing, CUDA
//! runtime absent, non-NVIDIA host), we push one warning and return
//! without touching `gpus`. Every NVIDIA entry keeps its `None`
//! augmentation fields; the probe still completes successfully.

use super::report::GpuInfo;

const NVIDIA_PCI_VENDOR_ID: u32 = 0x10DE;

pub fn augment(gpus: &mut [GpuInfo], warnings: &mut Vec<String>) {
    // Skip early if no NVIDIA GPU was enumerated by wgpu — no reason
    // to spin up NVML just to iterate over nothing.
    let any_nvidia = gpus
        .iter()
        .any(|g| g.vendor_id == NVIDIA_PCI_VENDOR_ID);
    if !any_nvidia {
        return;
    }

    let nvml = match nvml_wrapper::Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            warnings.push(format!(
                "NVML unavailable, NVIDIA augmentation skipped: {}. \
                 NVIDIA GPUs will appear in the report without \
                 compute_capability, VRAM, or driver version.",
                e
            ));
            return;
        }
    };

    // CUDA driver version (shared across all NVIDIA devices on the
    // host) — query once up front so we can attach it to each entry.
    let cuda_runtime_label = nvml
        .sys_cuda_driver_version()
        .ok()
        .map(|v| {
            // nvml returns an integer like 13020 for CUDA 13.2.
            // Formula documented by NVIDIA: major = v / 1000,
            // minor = (v % 1000) / 10.
            let major = v / 1000;
            let minor = (v % 1000) / 10;
            format!("CUDA {}.{}", major, minor)
        });

    let device_count = match nvml.device_count() {
        Ok(n) => n,
        Err(e) => {
            warnings.push(format!(
                "NVML device_count failed: {}. NVIDIA augmentation skipped.",
                e
            ));
            return;
        }
    };

    // Build a (vendor, device) -> NVML ordinal map. NVML devices are
    // identified by u32 ordinal; we look up the PCI pair once per
    // device so the match loop stays cheap.
    //
    // Matching strategy: the SAME physical GPU often appears in the
    // wgpu enumeration multiple times (e.g. Vulkan + DX12 + GL on
    // Windows — three entries, one card). All three wgpu entries
    // share the same (vendor_id, device_id) pair and must match the
    // same NVML device; we do NOT consume the match. This also
    // handles multi-GPU hosts with identical cards in the common
    // case: both cards have the same PCI IDs and all wgpu entries
    // for either card get the same augmentation (CC, driver version,
    // CUDA runtime). Per-device free-memory drift between two
    // identical cards is accepted as a known limitation — exact
    // per-card attribution would require PCI bus/device/function
    // matching, which wgpu does not expose.
    let mut nvml_by_pci: Vec<(u16, u16, u32)> = Vec::new();
    for ordinal in 0..device_count {
        let Ok(dev) = nvml.device_by_index(ordinal) else { continue };
        let Ok(pci) = dev.pci_info() else { continue };
        let combined = pci.pci_device_id;
        let vendor = (combined & 0xFFFF) as u16;
        let device = ((combined >> 16) & 0xFFFF) as u16;
        nvml_by_pci.push((vendor, device, ordinal));
    }

    for gpu in gpus.iter_mut().filter(|g| g.vendor_id == NVIDIA_PCI_VENDOR_ID) {
        let match_idx = nvml_by_pci.iter().position(|(v, d, _)| {
            *v as u32 == gpu.vendor_id && *d as u32 == gpu.device_id
        });

        let Some(i) = match_idx else {
            warnings.push(format!(
                "NVIDIA GPU {} ({}) enumerated by wgpu but no matching NVML device \
                 found (vendor=0x{:04X}, device=0x{:04X}); augmentation skipped for \
                 this entry.",
                gpu.index, gpu.name, gpu.vendor_id, gpu.device_id
            ));
            continue;
        };
        let ordinal = nvml_by_pci[i].2;

        let Ok(dev) = nvml.device_by_index(ordinal) else { continue };

        if let Ok(cc) = dev.cuda_compute_capability() {
            gpu.compute_capability = Some(format!("{}.{}", cc.major, cc.minor));
        }

        if let Ok(mem) = dev.memory_info() {
            // nvml memory fields are in bytes.
            gpu.vram_mb_total = Some(mem.total / 1_000_000);
            gpu.vram_mb_free = Some(mem.free / 1_000_000);
        }

        // Driver version from NVML overrides the wgpu `driver_info`
        // string (which can be empty or a short build descriptor).
        // Keep the wgpu value if NVML's call fails.
        if let Ok(drv) = nvml.sys_driver_version() {
            gpu.driver_version = Some(drv);
        }

        if let Some(label) = &cuda_runtime_label {
            gpu.runtime_detected.push(label.clone());
        }
    }
}
