// APX 8.10 — GPU Codegen Mock
// Generación de código GPU textual (no compilable, no ejecutable).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Hip,
    Vulkan,
}

pub struct KernelTemplate {
    pub op_name: &'static str,
    pub arity: usize,
    pub body_ir: String,
}

pub trait GpuCodegen {
    fn generate(&self, tpl: &KernelTemplate) -> String;
}

pub struct CudaCodegen;

impl GpuCodegen for CudaCodegen {
    fn generate(&self, tpl: &KernelTemplate) -> String {
        format!(
r#"// CUDA mock kernel
extern "C" __global__
void {name}(float* a, float* b, float* out, int N) {{
    // mock IR:
    // {ir}
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {{
        out[i] = a[i] + b[i]; // fake op
    }}
}}"#,
            name = tpl.op_name,
            ir = tpl.body_ir.replace("\n", "\n    // "),
        )
    }
}

pub struct MetalCodegen;

impl GpuCodegen for MetalCodegen {
    fn generate(&self, tpl: &KernelTemplate) -> String {
        format!(
r#"// Metal mock kernel
#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {{
    // mock IR:
    // {ir}
    out[id] = a[id] + b[id];
}}"#,
            name = tpl.op_name,
            ir = tpl.body_ir.replace("\n", "\n    // "),
        )
    }
}

pub struct HipCodegen;

impl GpuCodegen for HipCodegen {
    fn generate(&self, tpl: &KernelTemplate) -> String {
        format!(
r#"// HIP mock kernel
extern "C" __global__
void {name}(float* a, float* b, float* out, int N) {{
    // mock IR:
    // {ir}
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {{
        out[i] = a[i] + b[i]; // fake op
    }}
}}"#,
            name = tpl.op_name,
            ir = tpl.body_ir.replace("\n", "\n    // "),
        )
    }
}

pub struct VulkanCodegen;

impl GpuCodegen for VulkanCodegen {
    fn generate(&self, tpl: &KernelTemplate) -> String {
        format!(
r#"// Vulkan mock compute shader
// entry: {name}
// mock IR:
// {ir}

// layout(local_size_x = 64) in;
// layout(set = 0, binding = 0) readonly buffer A {{ float a[]; }};
// layout(set = 0, binding = 1) readonly buffer B {{ float b[]; }};
// layout(set = 0, binding = 2) writeonly buffer Out {{ float out[]; }};

// void main() {{
//     uint id = gl_GlobalInvocationID.x;
//     out[id] = a[id] + b[id];
// }}"#,
            name = tpl.op_name,
            ir = tpl.body_ir.replace("\n", "\n// "),
        )
    }
}
