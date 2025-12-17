pub mod hal;
pub mod tensor;
pub mod amg;
pub mod age;
pub mod amm;
pub mod pps;
pub mod data;
pub mod training;
pub mod api;
pub mod nn;
pub mod autograd;
pub mod optim;
pub mod ops;
pub mod profiler;
pub mod cpu_features;
pub mod simd_kernels;
pub mod matmul_dispatcher;
pub mod simd_fused_kernels;
pub mod config;
pub mod gpu;
pub mod gpu_autodiff;
pub mod engine;
pub mod validator;
pub mod apx3;
pub mod apx3_5;
pub mod apx3_8;
pub mod apx3_9;
pub mod apx4;
pub mod cuda;
pub mod apx4_3;
pub mod apx4_5;
pub mod apx4_7;
pub mod apx4_8;
pub mod apx4_9;
pub mod apx4_11;
pub mod apx4_12;
pub mod apx4_13;
pub mod apx5;
pub mod apx5_4;
pub mod kernels;
pub mod matmul;
pub mod apx6_2;
pub mod apx6_3 {
    pub mod tiled_avx2;
}
pub mod apx6;
pub mod apx6_4;
pub mod apx6_5;
pub mod apx6_6_auto_tiling;
pub mod apx6_7 {
    pub mod runtime_profile;
    pub mod auto_bench;
}
pub mod apx6_8;
pub mod apx6_9;
pub mod apx6_10;
pub mod apx6_11;
pub mod apx6_12;
pub mod apx6_13;
pub mod apx6_14;
pub mod apx6_15;
pub mod apx7;
pub mod apx7_2;
pub mod apx8;
pub mod apx9;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use crate::apx5_4::AdaptiveSelector;
use crate::apx6_7::runtime_profile::RuntimeProfile;
use crate::apx6_8::BlockSizePredictor;

pub use crate::apx6_12::adaptive_scheduler::*;
pub use crate::apx6_13::tempered_selector::*;
pub use crate::apx6_14::temperature_schedule::*;
pub use crate::apx6_14::temperature_manager::*;
pub use crate::apx6_15::stabilizer::*;
pub use crate::apx7_2::pgl::*;
pub use crate::apx8::dualgraph::*;
pub use crate::apx8::hybrid_dispatcher::*;
pub use crate::apx8::gpu_transfer_estimator::*;
pub use crate::apx8::mirror::*;
pub use crate::apx8::persistent::*;
pub use crate::apx8::gpu_kernels::*;
pub use crate::apx8::kernel_registry::*;
pub use crate::apx8::gpu_kernel_signature::*;
pub use crate::apx8::kernel_generator::*;
pub use crate::apx8::codegen_mock::*;
pub use crate::apx8::gpu_compiler_stub::*;
pub use crate::apx8::gpu_metalayer::*;
pub use crate::apx8::codegen::gpu_codegen_v1::*;
pub use crate::apx8::gpu_autoselector::*;
pub use crate::apx8::precompile_cache::*;
pub use crate::apx8::multiarch_router::*;
pub use crate::apx8::gpu_finalizer::*;
pub use crate::apx8::device_planner::*;
pub use crate::apx8::gpu_partition::*;
pub use crate::apx8::hxo::*;
// APX 9.1: exponer sólo el IR de alto nivel; GpuKernelIR/GpuOp se usan vía apx9::gpu_ir.
pub use crate::apx9::gpu_ir::{GpuIrKernel, GpuIrParam, GpuIrType, GpuIrStmt};
pub use crate::apx9::ptx_emitter::*;
pub use crate::apx9::ptx_validator::*;
pub use crate::apx9::sass_translator::*;
pub use crate::apx9::sass_optimizer::*;
pub use crate::apx9::memory_planner::*;
pub use crate::apx9::gpu_execution_planner::*;
pub use crate::apx9::gpu_executor_mock::*;
pub use crate::apx9::gpu_autotuner::*;
pub use crate::apx9::gpu_codegen_real::*;
pub use crate::apx9::cpu_to_ptx::*;
pub use crate::apx9::vgpu_executor::*;
pub use crate::apx9::vgpu_memory::*;
pub use crate::apx9::vgpu_runner::*;
pub use crate::apx9::vgpu_block_launcher::*;
pub use crate::apx9::vgpu_sync::*;
pub use crate::apx9::vgpu_warp::*;
pub use crate::apx9::vgpu_warp_scheduler::*;
pub use crate::apx9::vgpu_divergence::*;
pub use crate::apx9::vgpu_scoreboard::*;
pub use crate::apx9::vgpu_tensor_core::*;
pub use crate::apx9::vgpu_sm::*;

pub static APX_SILENT_MODE: AtomicBool = AtomicBool::new(false);

pub fn apx_set_silent_mode(v: bool) {
    APX_SILENT_MODE.store(v, Ordering::Relaxed);
}

pub fn apx_is_silent() -> bool {
    APX_SILENT_MODE.load(Ordering::Relaxed)
}

static ADAPTIVE_SELECTOR: OnceLock<Mutex<AdaptiveSelector>> = OnceLock::new();
static RUNTIME_PROFILE: OnceLock<Mutex<RuntimeProfile>> = OnceLock::new();
static BLOCK_PREDICTOR: OnceLock<Mutex<BlockSizePredictor>> = OnceLock::new();

pub fn global_adaptive_selector() -> &'static Mutex<AdaptiveSelector> {
    ADAPTIVE_SELECTOR.get_or_init(|| Mutex::new(AdaptiveSelector::new()))
}

pub fn global_runtime_profile() -> &'static Mutex<RuntimeProfile> {
    RUNTIME_PROFILE.get_or_init(|| Mutex::new(RuntimeProfile::new()))
}

pub fn global_block_predictor() -> &'static Mutex<BlockSizePredictor> {
    BLOCK_PREDICTOR.get_or_init(|| Mutex::new(BlockSizePredictor::new()))
}

/// Global debug flag controlled by ATENIA_DEBUG o APX_DEBUG.
/// Cuando es true, se habilitan trazas APX verbosas y logs de planner.
pub fn apx_debug_enabled() -> bool {
    // Primero intentamos ATENIA_DEBUG (nombre "oficial"), y como
    // alternativa aceptamos también APX_DEBUG para conveniencia.
    let raw = std::env::var("ATENIA_DEBUG")
        .or_else(|_| std::env::var("APX_DEBUG"))
        .unwrap_or_else(|_| "0".to_string());

    raw == "1" || raw.to_lowercase() == "true"
}

/// Initialize global APX-related facilities (e.g. GPU memory pool).
pub fn init_apx() {
    // APX 4.12: pool de 64MB con 8 bloques.
    crate::apx4_12::init_pool(64 * 1024 * 1024, 8);
}

/// Returns the current APX mode as configured by the environment.
/// Defaults to "4.19" when ATENIA_APX_MODE is not set.
pub fn apx_mode() -> String {
    std::env::var("ATENIA_APX_MODE").unwrap_or_else(|_| "4.19".to_string())
}

pub fn apx_mode_at_least(target: &str) -> bool {
    let mode = apx_mode();
    mode == target || mode > target.to_string()
}

#[ctor::ctor]
fn init_parallel_runtime_entrypoint() {
    crate::cpu_features::init_parallel_runtime();
    let feats = crate::cpu_features::cpu_features();
    if !crate::apx_is_silent() {
        println!(
            "[APX] Parallel runtime initialized: {} threads | AVX2={} | AVX512={} | FMA={}",
            feats.threads, feats.avx2, feats.avx512f, feats.fma
        );
    }

    // Loguear el modo APX actual cuando el debug está activo.
    if crate::apx_debug_enabled() {
        eprintln!("[APX] Using mode {}", crate::apx_mode());
    }

    // APX 8.6: habilitar mini-kernels GPU v0 (VecAdd) cuando el modo lo permite.
    if crate::apx_mode_at_least("8.6") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_kernels = true;
    }

    // APX 9.2/9.3: habilitar mini-kernels GPU v0 cuando se utiliza la toolchain PTX simulada.
    if crate::apx_mode_at_least("9.2") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_kernels = true;

        // Ejemplo mínimo de generación de PTX para depuración simbólica.
        if crate::apx_debug_enabled() {
            use crate::apx9::gpu_ir::{GpuKernelIR, GpuOp};
            use crate::apx9::ptx_emitter::PtxEmitter;
            use crate::apx9::ptx_validator::PtxValidator;
            use crate::apx9::sass_translator::SassTranslator;
            use crate::apx9::sass_optimizer::SassOptimizer;

            let ir = GpuKernelIR {
                name: "vecadd".to_string(),
                threads: 256,
                ops: vec![
                    GpuOp::Load { dst: "%f1".into(), src: "%A+tid*4".into() },
                    GpuOp::Load { dst: "%f2".into(), src: "%B+tid*4".into() },
                    GpuOp::Add  { dst: "%f3".into(), a: "%f1".into(), b: "%f2".into() },
                    GpuOp::Store { dst: "%Out+tid*4".into(), src: "%f3".into() },
                ],
            };

            let ptx = PtxEmitter::emit(&ir);
            if crate::apx_mode_at_least("9.3") {
                let validation = PtxValidator::validate(&ptx);
                if !validation.ok {
                    eprintln!("[APX 9.3] PTX VALIDATION ERRORS:");
                    for err in validation.errors {
                        eprintln!("  - {}", err);
                    }
                }
            }

            if crate::apx_mode_at_least("9.4") {
                let sass = SassTranslator::translate(&ptx).sass;
                eprintln!("[APX 9.4] SASS mock:\n{}", sass);

                if crate::apx_mode_at_least("9.5") {
                    let optimized = SassOptimizer::optimize(&sass);
                    eprintln!("[APX 9.5] SASS optimized:\n{}", optimized);
                }
            }

            eprintln!("[APX 9.2] PTX generated:\n{}", ptx);
        }
    }

    // APX 9.6: habilitar el planificador de memoria GPU simulado.
    if crate::apx_mode_at_least("9.6") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_memory_planner = true;
    }

    // APX 9.7: habilitar el planificador de ejecución GPU simulado.
    if crate::apx_mode_at_least("9.7") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_execution_planner = true;
    }

    // APX 9.8: habilitar el ejecutor GPU simulado.
    if crate::apx_mode_at_least("9.8") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_executor_mock = true;
    }

    // Initialize kernel registry for APX 3.8 dispatch layer.
    init_kernels();
}

use std::sync::Arc;
pub fn init_kernels() {
    use crate::apx3_8::kernel_registry::KernelRegistry;

    let reg = KernelRegistry::global();

    // Scalar matmul fallback.
    reg.register(
        "scalar_matmul",
        Arc::new(|a, b, out, m, k, n| {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += a[i * k + p] * b[p * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
        }),
    );

    // AVX2 matmul if available.
    #[cfg(target_feature = "avx2")]
    reg.register(
        "avx2_matmul",
        Arc::new(|a, b, out, m, k, n| unsafe {
            crate::simd_kernels::avx2::matmul_avx2(a, b, out, m, k, n)
        }),
    );

    // APX 8.7: registrar mini-kernels GPU v0 en el registry de kernels GPU.
    {
        use crate::apx8::gpu_kernels::gpu_vec_add;
        use crate::apx8::kernel_registry::{KERNEL_REGISTRY as APX8_KERNEL_REGISTRY, KernelKey as APX8KernelKey};

        APX8_KERNEL_REGISTRY.register(APX8KernelKey::VecAdd, gpu_vec_add);
    }

    // APX 8.8: registrar firmas de kernels GPU sin ejecutar nada.
    {
        use crate::apx8::gpu_kernel_signature::{register_signature, GpuKernelSignature, GpuKernelType};

        register_signature(GpuKernelSignature {
            key: GpuKernelType::VecAdd,
            min_dims: (1, 1, 1),
            max_dims: (1_000_000, 1, 1),
            workspace_bytes: 0,
            launcher_name: "vec_add_launcher_v0",
        });
    }
}
