// APX 8.13 — GPU Codegen v1
// Generación de kernels GPU sintéticos a partir de KernelIR. No ejecuta nada.

use crate::apx8::kernel_generator::KernelIR;
use crate::apx8::gpu_autoselector::GPUAutoSelector;
use crate::apx8::precompile_cache::PrecompileCache;
use crate::apx8::multiarch_router::route_kernel;
use crate::apx8::gpu_finalizer::gpu_finalize;
use crate::apx8::device_planner::plan_for_ir;

#[derive(Debug, Clone)]
pub struct GPUCodegenV1 {
    pub target: String, // "cuda" | "hip" | "metal"
}

impl GPUCodegenV1 {
    pub fn generate_kernel(&self, ir: &KernelIR) -> String {
        match self.target.as_str() {
            "cuda" => self.codegen_cuda(ir),
            "hip" => self.codegen_hip(ir),
            "metal" => self.codegen_metal(ir),
            _ => panic!("Unknown GPU target"),
        }
    }

    fn codegen_cuda(&self, ir: &KernelIR) -> String {
        format!(
            "__global__ void {}({}) {{ /* synthetic cuda */ }}",
            ir.name,
            ir.params.join(", "),
        )
    }

    fn codegen_hip(&self, ir: &KernelIR) -> String {
        format!(
            "__global__ void {}({}) {{ /* synthetic hip */ }}",
            ir.name,
            ir.params.join(", "),
        )
    }

    fn codegen_metal(&self, ir: &KernelIR) -> String {
        format!(
            "kernel void {}({}) {{ /* synthetic metal */ }}",
            ir.name,
            ir.params.join(", "),
        )
    }

    /// Crea un GPUCodegenV1 eligiendo automáticamente el backend
    /// en función del KernelIR usando el GPUAutoSelector v0.
    pub fn with_autoselect(ir: &KernelIR) -> Self {
        let sel = GPUAutoSelector::detect();
        let raw = sel.choose_backend(ir);
        // Normalizar backend simbólico del auto-selector a los targets soportados
        // por GPUCodegenV1. Esto mantiene la simulación pero evita panics.
        let target = match raw.as_str() {
            "cuda" | "hip" | "metal" => raw,
            "nvidia" => "cuda".into(),
            "amd" => "hip".into(),
            "intel" => "metal".into(),
            _ => "cuda".into(), // fallback determinístico
        };
        GPUCodegenV1 { target }
    }

    /// Versión que simula el uso de un cache de pre-compilación.
    /// No altera la ejecución real; sólo devuelve strings sintéticos.
    pub fn codegen_with_cache(ir: &KernelIR, cache: &mut PrecompileCache) -> String {
        let compiled = cache.compile_if_missing(ir);
        format!("target={} | {}", ir.name, compiled)
    }

    /// Versión multi-arch que selecciona simbólicamente la arquitectura
    /// pero sólo devuelve texto; no ejecuta ni compila nada.
    pub fn codegen_multiarch(ir: &KernelIR) -> String {
        let arch = route_kernel(ir);
        format!("arch={:?} | kernel_ir={}", arch, ir.name)
    }

    /// Pipeline simbólico: IR -> codegen sintético -> finalizer stub.
    pub fn codegen_with_finalizer(ir: &KernelIR) -> String {
        // Usamos GPUCodegenV1 con autoselector como etapa intermedia.
        let cg = GPUCodegenV1::with_autoselect(ir);
        let intermediate = cg.generate_kernel(ir);
        let final_stage = gpu_finalize(ir);

        format!("{}\n{}", intermediate, final_stage)
    }

    /// Versión que integra el planner de dispositivo simulado.
    pub fn codegen_with_planner(ir: &KernelIR) -> String {
        let plan = plan_for_ir(&ir.name);
        format!("MOCK CODEGEN for {}\nPLAN={:?}", ir.name, plan)
    }
}
