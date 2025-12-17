// APX 8.16 — Multi-Arch Kernel Routing v0
// Ruta simbólica de kernels a arquitecturas (CPU / CUDA / HIP / Metal / Vulkan).
// No compila ni ejecuta nada real.

use crate::apx8::kernel_generator::KernelIR;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetArch {
    CPU,
    CUDA,
    HIP,
    METAL,
    VULKAN,
}

pub fn route_kernel(ir: &KernelIR) -> TargetArch {
    let sig = ir.signature();

    if sig.contains("cuda") {
        return TargetArch::CUDA;
    }
    if sig.contains("hip") {
        return TargetArch::HIP;
    }
    if sig.contains("metal") {
        return TargetArch::METAL;
    }
    if sig.contains("vk") || sig.contains("vulkan") {
        return TargetArch::VULKAN;
    }

    TargetArch::CPU
}
