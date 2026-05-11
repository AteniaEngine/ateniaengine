//! wgpu-backed adapter enumeration.
//!
//! Cross-vendor, cross-OS: on each supported platform wgpu falls
//! through Vulkan / DX12 / Metal / OpenGL depending on what the host
//! exposes. Every GPU visible to the graphics stack is returned —
//! including integrated GPUs, which matters for notebooks where
//! Atenia's "good citizen" design must not treat the iGPU as free
//! memory.
//!
//! What wgpu does NOT give us:
//! - VRAM total / free. `AdapterInfo` exposes no memory fields;
//!   wgpu's `Limits::max_buffer_size` is a logical cap, not physical
//!   VRAM. Physical VRAM is populated by vendor-specific augmentation
//!   (NVML for NVIDIA; future: DXGI `QueryVideoMemoryInfo` on Windows,
//!   Metal `recommendedMaxWorkingSetSize` on macOS).
//! - Compute runtime presence (CUDA / ROCm / oneAPI). Also handled by
//!   augmentation.
//!
//! The functions below do not create a wgpu device, only an instance
//! and a round of `enumerate_adapters`. This keeps the probe cheap —
//! no GPU memory is allocated, no shader modules are compiled.

use super::report::{GpuInfo, vendor_name};

/// Enumerate every graphics adapter reachable from this host via
/// wgpu. Returns an empty Vec on hosts where no supported backend is
/// present (some headless servers, or containers without GPU
/// passthrough). Never panics — failures inside wgpu's enumerate
/// path surface as 0 adapters plus a warning pushed into the outer
/// report.
pub fn enumerate(warnings: &mut Vec<String>) -> Vec<GpuInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    if adapters.is_empty() {
        warnings.push(
            "wgpu enumerate_adapters returned 0 adapters; no Vulkan/DX12/\
             Metal/GL backend is available to this process. If a GPU is \
             physically present, check graphics driver installation."
                .to_string(),
        );
        return Vec::new();
    }

    adapters
        .into_iter()
        .enumerate()
        .map(|(index, adapter)| {
            let info = adapter.get_info();

            let driver = (!info.driver.is_empty()).then(|| info.driver.clone());
            let driver_version = (!info.driver_info.is_empty()).then(|| info.driver_info.clone());

            GpuInfo {
                index,
                vendor: vendor_name(info.vendor).to_string(),
                vendor_id: info.vendor,
                device_id: info.device,
                name: info.name,
                device_type: device_type_string(info.device_type),
                backend: backend_string(info.backend),
                driver,
                driver_version,
                vram_mb_total: None, // filled by augmentation
                vram_mb_free: None,
                runtime_detected: Vec::new(),
                compute_capability: None,
                atenia_support_tier: None,
            }
        })
        .collect()
}

fn device_type_string(t: wgpu::DeviceType) -> String {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
    .to_string()
}

fn backend_string(b: wgpu::Backend) -> String {
    match b {
        wgpu::Backend::Empty => "empty",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "browser-webgpu",
    }
    .to_string()
}
