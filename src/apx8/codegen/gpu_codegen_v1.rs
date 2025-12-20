// APX 8.13 — GPU Codegen v1
// Synthetic GPU kernel generation from KernelIR. Does not execute anything.

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

    /// Create a GPUCodegenV1 by automatically choosing the backend
    /// based on KernelIR using GPUAutoSelector v0.
    pub fn with_autoselect(ir: &KernelIR) -> Self {
        let sel = GPUAutoSelector::detect();
        let raw = sel.choose_backend(ir);
        // Normalize the auto-selector symbolic backend to targets supported
        // by GPUCodegenV1. This keeps the simulation but avoids panics.
        let target = match raw.as_str() {
            "cuda" | "hip" | "metal" => raw,
            "nvidia" => "cuda".into(),
            "amd" => "hip".into(),
            "intel" => "metal".into(),
            _ => "cuda".into(), // deterministic fallback
        };
        GPUCodegenV1 { target }
    }

    /// Version that simulates using a pre-compilation cache.
    /// Does not affect real execution; only returns synthetic strings.
    pub fn codegen_with_cache(ir: &KernelIR, cache: &mut PrecompileCache) -> String {
        let compiled = cache.compile_if_missing(ir);
        format!("target={} | {}", ir.name, compiled)
    }

    /// Multi-arch version that symbolically selects the architecture
    /// but only returns text; it does not execute nor compile anything.
    pub fn codegen_multiarch(ir: &KernelIR) -> String {
        let arch = route_kernel(ir);
        format!("arch={:?} | kernel_ir={}", arch, ir.name)
    }

    /// Symbolic pipeline: IR -> synthetic codegen -> finalizer stub.
    pub fn codegen_with_finalizer(ir: &KernelIR) -> String {
        // Use GPUCodegenV1 with autoselector as an intermediate stage.
        let cg = GPUCodegenV1::with_autoselect(ir);
        let intermediate = cg.generate_kernel(ir);
        let final_stage = gpu_finalize(ir);

        format!("{}\n{}", intermediate, final_stage)
    }

    /// Version that integrates the simulated device planner.
    pub fn codegen_with_planner(ir: &KernelIR) -> String {
        let plan = plan_for_ir(&ir.name);
        format!("MOCK CODEGEN for {}\nPLAN={:?}", ir.name, plan)
    }
}
