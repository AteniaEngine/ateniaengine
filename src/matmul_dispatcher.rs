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

    // APX 7.x: leer flags de runtime para habilitar rutas PEX.
    let (enable_pex, _enable_workstealing) = {
        let flags = crate::config::get_runtime_flags();
        (flags.enable_pex, flags.enable_workstealing)
    };

    // APX 6.8: selector de block-size basado en BlockSizePredictor. Si hay
    // un bloque óptimo conocido para este proceso, lo usamos con el kernel
    // tiled flexible, sólo en modo >= 6.8. Si no hay predicción, se sigue
    // el flujo normal (6.7 ABL, 6.5, 6.4, etc.).
    if crate::apx_mode_at_least("6.8") {
        if let Ok(pred) = crate::global_block_predictor().lock() {
            if let Some((bm, bn, bk)) = pred.best_block() {
                matmul_tiled_flex(a, b, out, m, k, n, bm, bn, bk);
                return;
            }
        }
    }

    // APX 6.7: Auto-Bench Learning (ABL). En modo >= 6.7, se realiza un
    // micro-benchmark inicial (una sola vez) para decidir si en esta
    // máquina conviene más el baseline 3.8 o el microkernel 6.4 para
    // tamaños típicos, y se usa ese perfil para seleccionar el kernel.
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
                        // Otros targets no se tocan aquí; dejamos que el
                        // flujo normal decida.
                    }
                }
            }
        }
    }

    // APX 7.3: si el modo adaptive PGL está habilitado vía flags de
    // runtime (por ejemplo usando `matmul_adaptive`), realizar un
    // micro-benchmark interno que mida seq/PEX/WS en buffers
    // temporales y alimente el runtime adaptativo. Esto no modifica
    // `out` ni la matemática del resultado.
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

            // Medidas internas usando los kernels 6.3b (seq/PEX/WS) sin
            // afectar al resultado real.
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

    // APX 6.5: kernel tiled AVX2 8x8 con doble packing, sólo cuando
    // - modo == "6.5",
    // - hay AVX2 disponible,
    // - tamaño suficientemente grande (>=128x128x128).
    if mode == "6.5"
        && std::is_x86_feature_detected!("avx2")
        && m >= 128
        && n >= 128
        && k >= 128
    {
        matmul_tiled_6_5(a, b, out, m, k, n);
        return;
    }

    // APX 6.4: microkernel AVX2 4x8 estilo BLIS, sólo cuando
    // - modo >= 6.4,
    // - hay AVX2 disponible,
    // - tamaño suficientemente grande (>=256x256),
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
        // APX 7.4: adaptación dinámica al load del sistema. Si el modo es
        // >= 7.4, se consulta un snapshot de carga y, en función de él,
        // se puede forzar seq/PEX/WS antes de delegar en el PGL 7.2.
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
                    // En esta implementación, la ruta PEX ya incorpora un
                    // planificador con work-stealing interno, por lo que la
                    // estrategia WS comparte kernel con PEX.
                    crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
                    return;
                }
                _ => {
                    // "pgl" u otra: dejamos que el PGL 7.2 decida.
                }
            }
        }

        // APX 7.2+: dejar que el Parallel GEMM Layer (PGL) decida la
        // estrategia concreta (seq / PEX / WS) sobre el mismo kernel base.
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
                    // la misma implementación numérica.
                    crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
                }
            }
            return;
        } else {
            // Comportamiento previo para APX < 7.2.
            if enable_pex {
                crate::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b_pex(a, b, out, m, k, n);
            } else {
                matmul_tiled_6_3b(a, b, out, m, k, n);
            }
            return;
        }
    }

    // APX 6.1: kernel tiled CPU determinista (sin hilos, sin SIMD) cuando el
    // modo es >= 6.1.
    if mode.starts_with("6.1") || mode > "6.1".to_string() {
        matmul_tiled_cpu(a, b, out, m, k, n);
        return;
    }

    // APX < 6.1 o sin AVX2 suficiente para 6.3: usar el dispatcher de
    // kernels registrado (AVX512/AVX2/scalar) sobre un contexto CPU. La ruta
    // GPU de MatMul sigue gestionada por apx4::gpu_dispatch y gpu_hooks.
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

        // Reutilizar el dispatcher APX 3.8 para cada slice de batch.
        dispatch_matmul_apx3_8(a_b, b_b, out_b, m, k, n, &ctx);
    }
}
