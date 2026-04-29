//! Atenia Engine core library
//!
//! Versioned architecture:
//! - v15: Policy & decision layer
//! - v16: Execution contracts, planning & guards
//! - v17: Model runtime & inference engine

pub mod v15;
pub mod v16;
pub mod v17;

// Hardware-probe module: cross-vendor GPU enumeration + optional NVIDIA
// augmentation. Behind a feature flag so normal builds do not pull wgpu
// or NVML. See `docs/HARDWARE_PROBE.md` for usage.
#[cfg(feature = "hw-probe")]
pub mod hw_probe;

// M4.9.b — public reproduction-surface helpers for the v20
// killer demo. Pressure probes, reactive-context factory,
// sharded BF16 load helper, argmax reduction. Default-on (see
// the `demo` feature in Cargo.toml); shared between the
// `atenia run` CLI subcommand and the `tests/m4_7_6_e_*`
// integration tests so both paths stay synchronised.
#[cfg(feature = "demo")]
pub mod demo;

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
pub mod v13;

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
// APX 9.1: expose only the high-level IR; GpuKernelIR/GpuOp are used via apx9::gpu_ir.
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
/// When true, enables verbose APX traces and planner logs.
pub fn apx_debug_enabled() -> bool {
    // First try ATENIA_DEBUG ("official" name), and as
    // an alternative also accept APX_DEBUG for convenience.
    let raw = std::env::var("ATENIA_DEBUG")
        .or_else(|_| std::env::var("APX_DEBUG"))
        .unwrap_or_else(|_| "0".to_string());

    raw == "1" || raw.to_lowercase() == "true"
}

/// Initialize global APX-related facilities (e.g. GPU memory pool).
pub fn init_apx() {
    // APX 4.12: 64MB pool with 8 blocks.
    crate::apx4_12::init_pool(64 * 1024 * 1024, 8);
}

/// Returns the current APX mode as configured by the environment.
///
/// **M4.8.b**: default lifted from `"4.19"` to `"7.2"`.
///
/// The pre-M4.8 default `"4.19"` made `apx_mode_at_least("6.3")`
/// return `false` under the legacy lexicographic comparison
/// (`'4' < '6'`), which routed every MatMul through the
/// `dispatch_matmul_apx3_8` chain at the bottom of
/// `matmul_dispatcher.rs`. Combined with `avx2_matmul` being
/// registered behind a compile-time `#[cfg(target_feature
/// = "avx2")]` (also fixed in M4.8.b), the production path
/// resolved to the scalar triple-loop registered as
/// `scalar_matmul`. M4.8.a's `bench_matmul` harness measured
/// the resulting throughput at 0.30–0.44 GFLOPS, ~600× below
/// the dev box's ~1.5 TFLOPS theoretical FP32 peak.
///
/// `"7.2"` activates the M4.7-era PGL (Parallel GEMM Layer),
/// the M4.6 ATO (Auto-Tiling Optimizer), and every AVX2 / FMA
/// branch in `matmul_dispatcher.rs`. `"7.2"` was selected
/// (rather than the higher `"7.5+"` parallel-executor modes)
/// because the parallel-executor paths
/// (`apx7::hpge` / `hls_deep` / `ule`) reshape graph
/// scheduling in ways the M4.7.5.f F64 family validation has
/// not yet re-run; M4.8.d will add explicit rayon
/// partitioning at the dispatcher layer instead, leaving the
/// graph executor on the M4.7-validated `run_plan` path.
///
/// Override via `ATENIA_APX_MODE`. Tests / benches that need
/// the legacy scalar baseline can set `ATENIA_APX_MODE=4.19`
/// and re-run.
pub fn apx_mode() -> String {
    std::env::var("ATENIA_APX_MODE").unwrap_or_else(|_| "7.2".to_string())
}

/// Compare the current APX mode against a target version.
///
/// **M4.8.b**: switched from lexicographic to numeric
/// comparison. The old implementation (`mode == target ||
/// mode > target.to_string()`) was a hidden bug: lex order
/// considers `"4.19" < "6.3"` (because `'4' < '6'`) but
/// also `"6.10" < "6.3"` (because `'1' < '3'`). The first
/// false-negative made the AVX2 path unreachable at the old
/// default mode; the second would have broken the dispatcher
/// the moment any milestone shipped a `"6.10+"` mode.
///
/// `parse_mode` parses every dot-separated segment as a
/// `u32` and compares lexicographically over the resulting
/// `Vec<u32>` — i.e. `(6, 10) > (6, 3)` and `(7, 2) > (4, 19)`
/// both hold. Non-numeric segments parse to 0, so a sentinel
/// like `"prod"` resolves to `(0,)` and never satisfies any
/// `at_least(numeric)` check (intentional — sentinels should
/// route through their own gates, not numeric comparisons).
pub fn apx_mode_at_least(target: &str) -> bool {
    fn parse_mode(s: &str) -> Vec<u32> {
        s.split('.').map(|seg| seg.parse::<u32>().unwrap_or(0)).collect()
    }
    let mode_v = parse_mode(&apx_mode());
    let target_v = parse_mode(target);
    mode_v >= target_v
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

    // Log current APX mode when debug is active.
    if crate::apx_debug_enabled() {
        eprintln!("[APX] Using mode {}", crate::apx_mode());
    }

    // APX 8.6: enable GPU mini-kernels v0 (VecAdd) when the mode allows it.
    if crate::apx_mode_at_least("8.6") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_kernels = true;
    }

    // APX 9.2/9.3: enable GPU mini-kernels v0 when using the simulated PTX toolchain.
    if crate::apx_mode_at_least("9.2") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_kernels = true;

        // Minimal PTX generation example for symbolic debugging.
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

    // APX 9.6: enable the simulated GPU memory planner.
    if crate::apx_mode_at_least("9.6") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_memory_planner = true;
    }

    // APX 9.7: enable the simulated GPU execution planner.
    if crate::apx_mode_at_least("9.7") {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_gpu_execution_planner = true;
    }

    // APX 9.8: enable the simulated GPU executor.
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

    // M4.8.b: AVX2 matmul registered when the **runtime CPU**
    // supports it, regardless of the compile-time
    // `target_feature` cfg. The previous gate
    // (`#[cfg(target_feature = "avx2")]`) only fires when the
    // build itself enables AVX2 — i.e. only with
    // `RUSTFLAGS="-C target-cpu=native"` or
    // `-C target-feature=+avx2`. Default `cargo build --release`
    // on `x86_64-pc-windows-msvc` does not set `target_feature
    // = "avx2"`, so the registration was silently a no-op and
    // the dispatcher fell through to `scalar_matmul`. M4.8.a
    // measured the consequence: 0.30 GFLOPS on a 1×5120×5120
    // shape on default builds.
    //
    // The runtime check below uses
    // `is_x86_feature_detected!`, which inspects CPUID at
    // program start and is hoisted by LLVM to a single
    // boolean load on every call. The kernel itself is
    // `unsafe fn` over raw `_mm256_*` intrinsics, which the
    // backend compiles regardless of the build's enabled
    // target features (the `unsafe` is the contract that the
    // operator has gated the call site on the runtime
    // detection). The Closure body re-asserts the precondition
    // via `is_x86_feature_detected!` for defence in depth on
    // a hypothetical future caller path that bypasses the
    // dispatcher gate.
    if std::is_x86_feature_detected!("avx2") {
        reg.register(
            "avx2_matmul",
            Arc::new(|a, b, out, m, k, n| {
                debug_assert!(
                    std::is_x86_feature_detected!("avx2"),
                    "avx2_matmul reached without AVX2 — registry should have fallen through to scalar_matmul"
                );
                unsafe {
                    crate::simd_kernels::avx2::matmul_avx2(a, b, out, m, k, n)
                }
            }),
        );
    }

    // APX 8.7: register GPU mini-kernels v0 in the GPU kernel registry.
    {
        use crate::apx8::gpu_kernels::gpu_vec_add;
        use crate::apx8::kernel_registry::{KERNEL_REGISTRY as APX8_KERNEL_REGISTRY, KernelKey as APX8KernelKey};

        APX8_KERNEL_REGISTRY.register(APX8KernelKey::VecAdd, gpu_vec_add);
    }

    // APX 8.8: register GPU kernel signatures without executing anything.
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
