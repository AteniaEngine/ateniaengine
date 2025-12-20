use crate::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use crate::tensor::Device;
use crate::kernels::matmul_tiled_cpu::matmul_tiled_cpu;
use crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b;
use crate::apx6_4::matmul_4x8_avx2;
use crate::apx5::kernel_planner::KernelTarget;
use crate::apx6_5::matmul_tiled_6_5;
use crate::matmul::matmul_tiled_flex::matmul_tiled_flex;

pub fn matmul_dispatch(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let mode = crate::apx_mode();

    // APX 7.x: read runtime flags to enable PEX paths.
    let (enable_pex, _enable_workstealing) = {
        let flags = crate::config::get_runtime_flags();
        (flags.enable_pex, flags.enable_workstealing)
    };

    // APX 6.8: block-size selector based on BlockSizePredictor. If there is
    // a known optimal block for this process, use it with the flexible tiled
    // kernel, only in mode >= 6.8. If there is no prediction, follow the
    // normal flow (6.7 ABL, 6.5, 6.4, etc.).
    if crate::apx_mode_at_least("6.8") {
        if let Ok(pred) = crate::global_block_predictor().lock() {
            if let Some((bm, bn, bk)) = pred.best_block() {
                matmul_tiled_flex(a, b, out, m, k, n, bm, bn, bk);
                return;
            }
        }
    }

    // APX 6.7: Auto-Bench Learning (ABL). In mode >= 6.7, run an
    // initial micro-benchmark (only once) to decide whether on this
    // machine it is better to use baseline 3.8 or the 6.4 microkernel for
    // typical sizes, and use that profile to select the kernel.
    if crate::apx_mode_at_least("6.7") {
        let profile_mutex = crate::global_runtime_profile();
        if let Ok(mut guard) = profile_mutex.lock() {
            if guard.entries.is_empty() {
                crate::apx6_7::auto_bench::run_initial_bench(&mut guard);
            }

            if let Some(best) = crate::apx6_7::auto_bench::estimate_best_kernel(m, &guard) {
                match best {
                    KernelTarget::CpuFastAvx2 => {
                        if std::is_x86_feature_detected!("avx2") && m >= 256 && n >= 256 {
                            matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
                            return;
                        }
                    }
                    KernelTarget::Cpu => {
                        let ctx = DeviceContext::new(Device::CPU);
                        dispatch_matmul_apx3_8(a, b, out, m, k, n, &ctx);
                        return;
                    }
                    _ => {
                        // Other targets are not handled here; let the
                        // normal flow decide.
                    }
                }
            }
        }
    }

    // APX 7.3: if adaptive PGL mode is enabled via runtime flags
    // (e.g. using `matmul_adaptive`), run an internal micro-benchmark that
    // measures seq/PEX/WS on temporary buffers and feeds the adaptive runtime.
    // This does not modify `out` nor the math of the result.
    {
        let flags = crate::config::get_runtime_flags();
        if flags.enable_adaptive_pgl {
            let bucket = crate::apx7::adaptive_pgl::bucket_for(n);
            let len_out = m * n;

            let mut tmp_seq = vec![0.0f32; len_out];
            let mut tmp_pex = vec![0.0f32; len_out];
            let mut tmp_ws = vec![0.0f32; len_out];

            fn quick_time<F: FnOnce()>(f: F) -> f64
            where
                F: FnOnce(),
            {
                let t0 = std::time::Instant::now();
                f();
                t0.elapsed().as_secs_f64() * 1000.0
            }

            // Internal measurements using 6.3b kernels (seq/PEX/WS) without
            // affecting the real result.
            let t_seq = quick_time(|| {
                matmul_tiled_6_3b(a, b, &mut tmp_seq, m, k, n);
            });
            let t_pex = quick_time(|| {
                crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(
                    a, b, &mut tmp_pex, m, k, n,
                );
            });
            let t_ws = quick_time(|| {
                crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(
                    a, b, &mut tmp_ws, m, k, n,
                );
            });

            if let Ok(mut guard) = crate::apx7::adaptive_pgl::ADAPTIVE_BUCKETS.write() {
                guard[bucket].record(t_seq, t_pex, t_ws);
            }
        }
    }

    // APX 6.5: tiled AVX2 8x8 kernel with double packing, only when
    // - modo == "6.5",
    // - hay AVX2 disponible,
    // - sufficiently large size (>=128x128x128).
    if mode == "6.5"
        && std::is_x86_feature_detected!("avx2")
        && m >= 128
        && n >= 128
        && k >= 128
    {
        matmul_tiled_6_5(a, b, out, m, k, n);
        return;
    }

    // APX 6.4: BLIS-style AVX2 4x8 microkernel, only when
    // - modo >= 6.4,
    // - hay AVX2 disponible,
    // - sufficiently large size (>=256x256),
    // - y estamos en CPU FP32 contiguo (garantizado por las slices &[f32]).
    if crate::apx_mode_at_least("6.4")
        && std::is_x86_feature_detected!("avx2")
        && m >= 256
        && n >= 256
    {
        matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
        return;
    }

    // APX 6.3B / 7.x: kernel tiled AVX2/FMA (32x32, unrolling en K).
    if crate::apx_mode_at_least("6.3") && std::is_x86_feature_detected!("avx2") {
        // APX 7.4: dynamic adaptation to system load. If mode is
        // >= 7.4, query a load snapshot and, based on it,
        // potentially force seq/PEX/WS before delegating to PGL 7.2.
        if crate::apx_mode_at_least("7.4") {
            let snap = crate::apx7::dynamic_load::sample_system_load();
            let strategy = crate::apx7::dynamic_load::choose_strategy(&snap);

            match strategy {
                "seq" => {
                    matmul_tiled_6_3b(a, b, out, m, k, n);
                    return;
                }
                "pex" => {
                    crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
                    return;
                }
                "ws" => {
                    // In this implementation, the PEX path already includes an
                    // internal work-stealing scheduler, so the WS strategy
                    // shares the kernel with PEX.
                    crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
                    return;
                }
                _ => {
                    // "pgl" or other: let PGL 7.2 decide.
                }
            }
        }

        // APX 7.2+: let the Parallel GEMM Layer (PGL) decide the
        // concrete strategy (seq / PEX / WS) over the same base kernel.
        if crate::apx_mode_at_least("7.2") {
            use crate::apx7_2::pgl::{decide_pgl, PGLStrategy};

            let threads = crate::cpu_features::cpu_features().threads.max(1) as usize;
            let decision = decide_pgl(m, k, n, threads);

            match decision.strategy {
                PGLStrategy::Seq => {
                    matmul_tiled_6_3b(a, b, out, m, k, n);
                }
                PGLStrategy::Pex | PGLStrategy::WorkStealing => {
                    // Actualmente PEX v2 (7.1) ya implementa WS interno
                    // sobre tiles, por lo que ambas estrategias comparten
                    // the same numeric implementation.
                    crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
                }
            }
            return;
        } else {
            // Previous behavior for APX < 7.2.
            if enable_pex {
                crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
            } else {
                matmul_tiled_6_3b(a, b, out, m, k, n);
            }
            return;
        }
    }

    // APX 6.1: deterministic tiled CPU kernel (no threads, no SIMD) when
    // mode is >= 6.1.
    if mode.starts_with("6.1") || mode > "6.1".to_string() {
        matmul_tiled_cpu(a, b, out, m, k, n);
        return;
    }

    // APX < 6.1 or without enough AVX2 for 6.3: use the registered kernel
    // dispatcher (AVX512/AVX2/scalar) over a CPU context. The MatMul GPU path
    // is still handled by apx4::gpu_dispatch and gpu_hooks.
    let ctx = DeviceContext::new(Device::CPU);
    dispatch_matmul_apx3_8(a, b, out, m, k, n, &ctx);
}

pub fn batch_matmul_dispatch(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let batch_stride_a = m * k;
    let batch_stride_b = k * n;
    let batch_stride_out = m * n;

    let ctx = DeviceContext::new(Device::CPU);

    for b_i in 0..batch {
        let a_b = &a[b_i * batch_stride_a..][..batch_stride_a];
        let b_b = &b[b_i * batch_stride_b..][..batch_stride_b];
        let out_b = &mut out[b_i * batch_stride_out..][..batch_stride_out];

        // Reuse the APX 3.8 dispatcher for each batch slice.
        dispatch_matmul_apx3_8(a_b, b_b, out_b, m, k, n, &ctx);
    }
}
