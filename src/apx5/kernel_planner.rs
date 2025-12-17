// APX 5: Dynamic Kernel Planner (estructura inicial)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTarget {
    Cpu,
    Gpu,
    HybridCpuGpu,
    CpuFastAvx2, // APX 6.2: ruta AVX2 opcional para MatMul
}

#[derive(Debug, Clone, Copy)]
pub struct KernelPlan {
    pub target: KernelTarget,
    pub reason: &'static str,
}

use crate::apx5_4::DeviceTarget;

pub struct KernelPlanner;

impl KernelPlanner {
    pub fn new() -> Self {
        KernelPlanner
    }

    /// Heurística inicial (APX 5.1).
    /// Por ahora, usamos simplemente el producto de las dimensiones como
    /// proxy del tamaño total de la operación.
    pub fn select_kernel(
        &self,
        op_name: &str,
        dims: &[usize],
        adaptive_pref: Option<DeviceTarget>,
    ) -> KernelPlan {
        // Si APX 5.4 ha aprendido una preferencia clara, la priorizamos.
        if let Some(pref) = adaptive_pref {
            return match pref {
                DeviceTarget::CPU => KernelPlan {
                    target: KernelTarget::Cpu,
                    reason: "APX 5.4: preferencia adaptativa CPU",
                },
                DeviceTarget::GPU => KernelPlan {
                    target: KernelTarget::Gpu,
                    reason: "APX 5.4: preferencia adaptativa GPU",
                },
            };
        }

        // APX 6.2: para MatMul en modos >= 6.2, sugerimos explícitamente la ruta
        // CpuFastAvx2. La integración final se hace en execute_single, con
        // fallback seguro al dispatcher APX 3.8 cuando AVX2 no está disponible.
        let apx_mode = crate::apx_mode();
        let is_62_or_higher = apx_mode.starts_with("6.2") || apx_mode > "6.2".to_string();
        if is_62_or_higher && op_name == "MatMul" {
            return KernelPlan {
                target: KernelTarget::CpuFastAvx2,
                reason: "APX 6.2: preferencia AVX2 para MatMul en CPU",
            };
        }

        // Heurística original de APX 5.2 basada en tamaño.
        let size: usize = if dims.is_empty() { 0 } else { dims.iter().product() };

        if size < 4096 {
            KernelPlan {
                target: KernelTarget::Cpu,
                reason: "OP pequeño   CPU es más eficiente",
            }
        } else {
            KernelPlan {
                target: KernelTarget::Gpu,
                reason: "OP grande   GPU recomendado",
            }
        }
    }
}
