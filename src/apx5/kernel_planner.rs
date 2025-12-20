// APX 5: Dynamic Kernel Planner (initial structure)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTarget {
    Cpu,
    Gpu,
    HybridCpuGpu,
    CpuFastAvx2, // APX 6.2: optional AVX2 path for MatMul
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

    /// Initial heuristic (APX 5.1).
    /// For now, we simply use the product of dimensions as a proxy for the
    /// operation's total size.
    pub fn select_kernel(
        &self,
        op_name: &str,
        dims: &[usize],
        adaptive_pref: Option<DeviceTarget>,
    ) -> KernelPlan {
        // If APX 5.4 has learned a clear preference, we prioritize it.
        if let Some(pref) = adaptive_pref {
            return match pref {
                DeviceTarget::CPU => KernelPlan {
                    target: KernelTarget::Cpu,
                    reason: "APX 5.4: adaptive CPU preference",
                },
                DeviceTarget::GPU => KernelPlan {
                    target: KernelTarget::Gpu,
                    reason: "APX 5.4: adaptive GPU preference",
                },
            };
        }

        // APX 6.2: for MatMul in modes >= 6.2, explicitly suggest the
        // CpuFastAvx2 path. Final integration is done in execute_single, with
        // safe fallback to the APX 3.8 dispatcher when AVX2 is not available.
        let apx_mode = crate::apx_mode();
        let is_62_or_higher = apx_mode.starts_with("6.2") || apx_mode > "6.2".to_string();
        if is_62_or_higher && op_name == "MatMul" {
            return KernelPlan {
                target: KernelTarget::CpuFastAvx2,
                reason: "APX 6.2: AVX2 preference for MatMul on CPU",
            };
        }

        // Original APX 5.2 size-based heuristic.
        let size: usize = if dims.is_empty() { 0 } else { dims.iter().product() };

        if size < 4096 {
            KernelPlan {
                target: KernelTarget::Cpu,
                reason: "Small op — CPU is more efficient",
            }
        } else {
            KernelPlan {
                target: KernelTarget::Gpu,
                reason: "Large op — GPU recommended",
            }
        }
    }
}
