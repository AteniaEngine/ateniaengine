// APX 8.20 — Hybrid Execution Orchestrator (HXO)
// Coordinador simbólico que unifica planner de dispositivo, partición,
// router multi-arch, cache de precompilación y codegen mock.
// No ejecuta GPU real ni modifica ningún cálculo numérico.

use crate::apx8::device_planner::plan_for_ir;
use crate::apx8::gpu_partition::suggest_partition;
use crate::apx8::kernel_generator::KernelIR;
use crate::apx8::multiarch_router::{route_kernel, TargetArch};
use crate::apx8::precompile_cache::PrecompileCache;
use crate::apx8::codegen::gpu_codegen_v1::GPUCodegenV1;

#[derive(Debug, Clone)]
pub struct HybridOpPlan {
    pub device: String,
    pub partition: String,
    pub backend: String,
    pub codegen: String,
    pub precompiled: bool,
}

/// Construye un plan híbrido simbólico para un IR y una forma lógica de tensor.
/// No ejecuta nada real; sólo combina heurísticas existentes.
pub fn build_hxo_plan(ir: &KernelIR, shape: &[usize]) -> HybridOpPlan {
    // 1. Device planner (8.18)
    let dev_plan = plan_for_ir(&ir.name);
    let device = dev_plan
        .target_gpu
        .as_ref()
        .map(|g| g.name.clone())
        .unwrap_or_else(|| "CPU".to_string());

    // 2. Partition planner (8.19)
    let part = suggest_partition(shape);
    let partition = format!("{:?}", part.policy);

    // 3. Backend router (8.16)
    let arch = route_kernel(ir);
    let backend = match arch {
        TargetArch::CPU => "CPU".to_string(),
        TargetArch::CUDA => "CUDA".to_string(),
        TargetArch::HIP => "HIP".to_string(),
        TargetArch::METAL => "METAL".to_string(),
        TargetArch::VULKAN => "VULKAN".to_string(),
    };

    // 4. Pre-compilation cache (8.15) — simulación local y determinista
    let mut cache = PrecompileCache::new();
    let was_precompiled = cache.contains(ir);
    let _compiled = cache.compile_if_missing(ir);

    // 5. Codegen mock + finalizer (8.13 + 8.17)
    let codegen = GPUCodegenV1::codegen_with_finalizer(ir);

    HybridOpPlan {
        device,
        partition,
        backend,
        codegen,
        precompiled: was_precompiled,
    }
}

/// Resultado simbólico de un dispatch híbrido controlado por HXO.
#[derive(Debug, Clone)]
pub enum HybridDispatchResult {
    Pseudo {
        device: String,
        backend: String,
        partition: String,
        codegen: String,
    },
}

/// Versión puramente simbólica de un "hybrid dispatch" basado en HXO.
/// No toca Graph ni Tensor; sólo devuelve metadatos.
pub fn hybrid_dispatch(ir: &KernelIR, shape: &[usize]) -> HybridDispatchResult {
    let plan = build_hxo_plan(ir, shape);
    HybridDispatchResult::Pseudo {
        device: plan.device,
        backend: plan.backend,
        partition: plan.partition,
        codegen: plan.codegen,
    }
}
