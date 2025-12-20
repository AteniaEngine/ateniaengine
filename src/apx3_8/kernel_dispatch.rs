use crate::apx3_8::device_context::DeviceContext;
use crate::apx3_8::kernel_registry::KernelRegistry;
use crate::apx6_5::matmul_tiled_6_5;

pub fn dispatch_matmul(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    ctx: &DeviceContext,
) {
    let registry = KernelRegistry::global();

    // APX 6.5: experimental path only for CPU forward when the mode is
    // exactly "6.5" and AVX2 is available. If anything fails, the rest of the
    // dispatcher preserves APX 3.8 behavior.
    if !ctx.is_gpu()
        && crate::apx_mode() == "6.5"
        && std::is_x86_feature_detected!("avx2")
    {
        matmul_tiled_6_5(a, b, out, m, k, n);
        return;
    }

    // GPU (future work with APX 4.0)
    if ctx.is_gpu() {
        if let Some(f) = registry.get("gpu_matmul") {
            return f(a, b, out, m, k, n);
        }
    }

    // AVX512
    if let Some(f) = registry.get("avx512_matmul") {
        return f(a, b, out, m, k, n);
    }

    // AVX2
    if let Some(f) = registry.get("avx2_matmul") {
        return f(a, b, out, m, k, n);
    }

    // Fallback
    if let Some(f) = registry.get("scalar_matmul") {
        return f(a, b, out, m, k, n);
    }

    panic!("APX 3.8 error: no kernel registered for matmul");
}
