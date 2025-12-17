# Guía de Tests de Atenia Engine

Este documento resume los tests de integración de `tests/`, organizados por fase **APX** y por categoría.
La idea es que los usuarios puedan:

- Entender **qué valida cada grupo de tests**.
- Elegir fácilmente **qué versión APX** o **qué subsistema** quieren probar.

> Nota: todos los comandos supondrán que estás en la raíz del proyecto `atenia-engine`.

---

## Cómo filtrar tests por versión APX

- Ejecutar **todos los tests** (completo):

  ```bash
  cargo test -- --nocapture
  ```

- Ejecutar sólo tests de una **fase APX** (por patrón de nombre de archivo/binario):

  ```bash
  # Ejemplos
  cargo test --test apx_4_*   -- --nocapture   # APX 4.x
  cargo test --test apx_6_*   -- --nocapture   # APX 6.x
  cargo test --test apx_7_*   -- --nocapture   # APX 7.x
  cargo test --test apx_8_*   -- --nocapture   # APX 8.x
  cargo test --test apx_9_*   -- --nocapture   # APX 9.x
  cargo test --test apx_11_*  -- --nocapture   # APX 11.x
  cargo test --test apx_12_*  -- --nocapture   # APX 12.x
  ```

- Ejecutar sólo tests cuyo **nombre de test** coincide con un prefijo:

  ```bash
  cargo test apx_12_3_ -- --nocapture    # Sólo tests etiquetados como APX 12.3
  ```

---

## APX 2.x – Fused Kernels iniciales

- **`apx_2_5_fused_kernels_test.rs`**  
  - **Qué valida**: primeras fusiones de kernels básicos (ej. combinaciones MatMul + activación).  
  - **Objetivo**: asegurar que las rutas fusionadas producen resultados equivalentes a las rutas naïve CPU.

---

## APX 3.x – Backward fusion y transferencia

- **`apx_3_0_fused_backward_test.rs`**  
  - **Qué valida**: backward fusion (cadenas simples fusionadas en backward) vs backward clásico.
- **`apx_3_5_transfer_test.rs`**  
  - **Qué valida**: rutas de transferencia (CPU↔GPU lógicas o abstracciones) en contexto de APX 3.x.
- **`apx_3_8_dispatch_test.rs` / `apx_3_9_router_test.rs`**  
  - **Qué valida**: dispatch/router inicial para seleccionar rutas fusionadas vs no fusionadas.

---

## APX 4.x – GPU core (CUDA bring-up, MatMul, Linear, Attention)

### 4.0–4.6: GPU básico y primeras fusiones

- **`apx_4_0_dispatch_test.rs`**  
  - **Qué valida**: dispatch básico de operaciones hacia GPU cuando está habilitado.
- **`apx_4_1_cuda_test.rs`**  
  - **Qué valida**: que el entorno CUDA mínimo está operativo (driver, dispositivo, etc.).
- **`apx_4_2_matmul_test.rs`**  
  - **Qué valida**: correctness de MatMul GPU vs referencia CPU en tamaños pequeños.
- **`apx_4_3_gpu_segment_test.rs` / `apx_4_3_linear_chain_compare.rs`**  
  - **Qué valida**: segmentación del grafo en segmentos GPU y comparación con rutas puramente CPU.
- **`apx_4_4_linear_test.rs`**  
  - **Qué valida**: operación `Linear` (MatMul + bias) en GPU vs CPU.
- **`apx_4_5_batch_matmul_test.rs`**  
  - **Qué valida**: batch matmul (multi-batch) GPU vs CPU.
- **`apx_4_6_full_pipeline_test.rs`**  
  - **Qué valida**: pipeline completo de forward con ruta GPU habilitada, equivalencia global con CPU.

### 4.7–4.9: Fusion Engine

- **`apx_4_7_fusion_test.rs`, `apx_4_8_fusion_test.rs`, `apx_4_9_fusion_test.rs`**  
  - **Qué valida**: distintos escenarios de fusión (linear + activación, cadenas de ops) y equivalencia numérica.

### 4.10–4.18: QKV, Self-Attention y backward

- **`apx_4_10_fused_linear_silu_gpu_test.rs`**  
  - **Qué valida**: kernel fusionado Linear+SiLU en GPU vs composición naive.
- **`apx_4_11_gpu_hooks_test.rs`**  
  - **Qué valida**: hooks GPU (instrumentación alrededor de kernels).
- **`apx_4_12_memory_pool_test.rs`**  
  - **Qué valida**: pool de memoria GPU simple (reservas/liberaciones y reutilización).
- **`apx_4_13_fusion_engine_test.rs`**  
  - **Qué valida**: FusionEngine global (registro y selección de fusiones).
- **`apx_4_14_qkv_forward_test.rs` / `apx_4_14_qkv_fusion_test.rs`**  
  - **Qué valida**: QKV forward y su versión fusionada, equivalencia con ruta no fusionada.
- **`apx_4_15_qkv_backward_math_test.rs`**  
  - **Qué valida**: matemática de backward de QKV contra referencias analíticas/numéricas.
- **`apx_4_16_qkv_backward_integration_test.rs`**  
  - **Qué valida**: backward de QKV en un grafo más grande (integración).
- **`apx_4_17_self_attention_forward_test.rs`**  
  - **Qué valida**: forward de Self-Attention (incluyendo QKV) y comparaciones con referencia CPU.
- **`apx_4_18_self_attention_backward_test.rs`**  
  - **Qué valida**: backward de Self-Attention completo (incluyendo acumulación de gradientes).

---

## APX 5.x – Kernel planner y adaptive matmul

- **`apx_5_1_kernel_planner_test.rs`**  
  - **Qué valida**: planner simple de kernels (elección de variantes según tamaño).
- **`apx_5_3_matmul_plan_test.rs` / `apx_5_3_batch_matmul_plan_test.rs` / `apx_5_3_planner_test.rs`**  
  - **Qué valida**: planificación concreta de MatMul/BatchMatMul (bloques, tiles, etc.).
- **`apx_5_4_adaptive_test.rs` / `apx_5_4_batch_matmul_adaptive_test.rs`**  
  - **Qué valida**: adaptación dinámica de planes de MatMul/BatchMatmul en función de forma/tamaño.

---

## APX 6.x – MatMul Perf & Fusion runtime

Hay muchos tests en APX 6.x; en resumen:

- **`apx_6_1_*`**  
  - Benchmarks y correctness de MatMul CPU vs GPU, con variantes tiled y AVX2.
- **`apx_6_2_matmul_avx2_benchmark.rs`**  
  - Bench de MatMul AVX2 puro.
- **`apx_6_3*`, `apx_6_4*`, `apx_6_5*`**  
  - Correctness + benchmarks de diversas variantes de MatMul y tiling.
- **`apx_6_6_*`, `apx_6_7_*`, `apx_6_8_*`, `apx_6_9_*`**  
  - Auto-tiling, profiling ligero y selección de rutas de fusión.
- **`apx_6_10_*`**  
  - `apx_6_10_fused_full_correctness_test.rs` / `apx_6_10_fusion_benchmark_test.rs` / `apx_6_10_selector_test.rs` / `apx_6_10_fused_full_benchmark_test.rs`  
  - Validan la correcta fusión de pipelines completos y su rendimiento relativo.
- **`apx_6_11_policy_test.rs`, `apx_6_12_scheduler_test.rs`, `apx_6_13_tempered_*`, `apx_6_14_temperature_test.rs`, `apx_6_15_stabilizer_test.rs`**  
  - Política global de fusión y ajustes de “temperatura”/estabilidad para seleccionar rutas.

---

## APX 7.x – Paralelismo en grafo y schedulers

La familia `apx_7_*` cubre:

- **`apx_7_0_*`, `apx_7_1_*`, `apx_7_2_*`, `apx_7_3_*`, `apx_7_4_*`**  
  - Schedulers paralelos básicos (PEX/WS), equivalencia vs ejecución secuencial, benchmarks ligeros.
- **`apx_7_5_hpge_test.rs`**  
  - **HPGE v1**: ejecución paralela a nivel de grafo, respetando dependencias.
- **`apx_7_6_hpge_priority_test.rs` / `apx_7_7_hpfa_test.rs`**  
  - **HPGE v2 + HPFA**: priorización por camino crítico y afinidad de fusión.
- **`apx_7_8_tlo_*`, `apx_7_9_hls_*`, `apx_7_10_hls_*`, `apx_7_11_pfls_*`, `apx_7_12_ule_*`**  
  - Optimizadores de localidad, scheduling jerárquico y predicción de futuros cuellos de botella.

---

## APX 8.x – GPU Simulation Stack (100% CPU)

Todos los tests `apx_8_*` validan la **pila GPU simbólica** (dual graph, IR, codegen mock, router multi-arch, planners, HXO).  
En general:

- **`apx_8_1_dualgraph_test.rs`**  
  - Construcción de DualGraph CPU/GPU.
- **`apx_8_2_hybrid_dispatcher_test.rs`**  
  - Dispatcher híbrido CPU/GPU simbólico.
- **`apx_8_3_gpu_transfer_test.rs`, `apx_8_4_mirror_test.rs`, `apx_8_5_persistent_cache_test.rs`**  
  - Estado de mirroring/persistencia GPU en `Tensor`.
- **`apx_8_6_gpu_kernels_test.rs`, `apx_8_7_kernel_registry_test.rs`, `apx_8_8_kernel_signatures_test.rs`, `apx_8_9_kernel_generator_test.rs`**  
  - Stubs de kernels, registro, firmas y generación de IR.
- **`apx_8_10_codegen_mock_test.rs`, `apx_8_11_gpu_compiler_stub_test.rs`, `apx_8_12_metalayer_test.rs`, `apx_8_13_gpu_codegen_test.rs`**  
  - Codegen simbólico, compilador stub, MetaLayer y rutas de generación multi-arch.
- **`apx_8_14_autoselector_test.rs`, `apx_8_15_precompile_cache_test.rs`, `apx_8_16_multiarch_router_test.rs`, `apx_8_17_gpu_finalizer_test.rs`**  
  - Auto-selector de backend, cache de precompilación, router multi-arch y finalizador stub.
- **`apx_8_18_device_planner_test.rs`, `apx_8_19_gpu_partition_test.rs`, `apx_8_20_hxo_test.rs`**  
  - Device planner simulado, políticas de partición y orquestador híbrido (HXO).

---

## APX 9.x – Virtual GPU (vGPU) y SM model

Todos los `apx_9_*` tests verifican la **vGPU pipeline** y el modelo de SM:

- **9.1–9.5**: IR GPU, PTX emitter, validator y traductor/optimizador SASS.  
- **9.6–9.10**: planificador de memoria, ejecutor GPU mock y autotuner simulado.  
- **9.11–9.14**: ejecutor vGPU de alto nivel, memoria virtual y `VGpuRunner`.  
- **9.15–9.19**: launcher de bloques, sincronización, warps y scheduler SIMT.  
- **9.20–9.25**: pipeline SIMT, scoreboard, OOW scheduler y dual-issue simbólico.

---

## APX 11.x – AutoDiff GPU infra

- **`apx_11_0_backward_ir.rs`**  
  - Generación de IR simbólico de backward (`BackwardKernelSpec`) para operaciones básicas.
- **`apx_11_1_linear_backward.rs`**  
  - Backward symbolico de `LinearOp`.
- **`apx_11_2_matmul_backward.rs`**  
  - Backward simbólico de `MatMulOp`.
- **`apx_11_3_attention_backward.rs`**  
  - Backward simbólico de atención (QKV + softmax + proyección).
- **`apx_11_4_fusion_test.rs`**  
  - Fusiones de backward en grafos simples.
- **`apx_11_5_tensor_gpu_roundtrip.rs` / `apx_11_6_tensor_bridge_api.rs`**  
  - Roundtrip CPU↔GPU para tensores y API de puente hacia autodiff GPU.
- **`apx_11_7_matmul_backward_real.rs`, `apx_11_8_linear_backward_real.rs`, `apx_11_9_attention_backward_real.rs`**  
  - Validan backward “realista” (en presencia de más ops) contra referencias CPU.

---

## APX 12.x – Runtime introspection y estabilidad GPU

- **`apx_12_0_kernel_normalization.rs`**  
  - Normalización de kernels antes de NVRTC (p. ej. `KernelNormalizer::normalize_kernel`).
- **`apx_12_1_autoplanner.rs`**  
  - `AutoPlanner::plan_square_matmul` y variantes: calcula grid/block/shared_mem simbólicos.
- **`apx_12_2_profiler.rs`**  
  - Profiling básico de ejecuciones GPU (tiempos promedio de kernels).
- **`apx_12_2_5_compat_layer.rs`**  
  - GPU Loader Compatibility Layer: paths nvJitLink, PTX directo, normalizado, downgraded y `CpuFallback`.
- **`apx_12_3_autotuner.rs`**  
  - Autotuner de MatMul:
    - `apx_12_3_autotuner_basic`: usa un runner simulado que es “más rápido con bloques grandes”.
    - `apx_12_3_cpu_fallback_mode`: valida el modo CPU (sin GPU) con parámetros por defecto.
- **`apx_12_4_device_profiler.rs`**  
  - Device Profiling v2: propiedades detalladas del dispositivo (SMs, warp size, etc.).
- **`apx_12_5_timeline.rs`**  
  - Línea de tiempo de lanzamientos de kernels (timestamps, orden de ejecución).
- **`apx_12_6_fingerprint.rs`**  
  - Fingerprinting de ejecuciones GPU: genera `u64` estables por kernel/configuración.
- **`apx_12_7_execution_recorder.rs`**  
  - Execution Recorder: registra cada ejecución con fingerprint, duración y parámetros.
- **`apx_12_8_kvh.rs`**  
  - Kernel Validation Harness: compara kernel GPU contra referencia CPU y reporta diferencias.
- **`apx_12_9_autotag.rs`**  
  - Auto-Tagging: añade tags como `"matmul"`, `"fused"`, `"autogen"` a fingerprints y registros.
- **`apx_12_10_heatmap.rs`**  
  - GPU Behavior Heatmap: frecuencia y latencias promedio de kernels.
- **`apx_12_11_perf_buckets.rs`**  
  - Performance buckets (T0–T4) basados en latencias promedio del heatmap.
- **`apx_12_12_consistency.rs`**  
  - GPU Consistency Scanner: ejecuta kernels pequeños repetidos y clasifica estado (`GpuConsistency`).
- **`apx_12_13_stability_map.rs`**  
  - GPU Stability Memory Map (VRAM Noise Detector):  
    - Usa `StabilityScanner::scan(total, step)` y verifica:
      - Que hay entradas.
      - Que `read_ms` son finitos.
      - Que `offset` < `total`.

- **`apx_12_module_cache_test.rs`**  
  - Cache de módulos NVRTC/CUDA: mismos PTX no recargan módulo, reutilizan handle cacheado.

### APX 12.x – Semántica de fallback y estabilidad de tests (12.0 → 12.14)

Esta fase evolucionó para soportar entornos reales (especialmente Windows) donde:

- El loader puede fallar por motivos ajenos al código (driver, nvJitLink, NVRTC, PTX versioning).
- Puede activarse `CpuFallback` como modo válido.

Reglas adoptadas:

- **`CompatLoader` (APX 12.2.5)** define el contrato de carga:
  - Intenta múltiples rutas (nvJitLink → PTX directo → PTX normalizado → PTX downgraded).
  - Si ninguna ruta funciona, devuelve `Err(CpuFallback)` y marca fallback global.
- **Tests GPU**:
  - Si el test valida “infra/robustez”, **fallback cuenta como éxito**.
  - Si el test depende de GPU real, debe **skippear** cuando el loader marca fallback.
- **Correctness GPU vs CPU** no se exige bit-perfect para infraestructura APX 12.x:
  - Los tests se vuelven **smoke tests** (forma + finitud), evitando falsos negativos.

### APX 12.14 – GPU Safety Layer v3 (No-Panic Fallback)

Objetivo: eliminar “random panics” por `unwrap()`/`expect()` en la ruta GPU.

- **Qué valida**:
  - Que la ruta GPU no paniquea ante fallos de loader/launch.
  - Que `CpuFallback` no rompe `cargo test`.
- **Qué se implementó**:
  - En ops GPU clave se reemplazó `unwrap()` por manejo no-panic.
  - Ante `CpuFallback`, los ops retornan temprano o dejan que el caller/test trate el fallback.
  - Los tests relacionados se mantienen en modo smoke/skip según corresponda.


---

## Tests GPU “core” fuera de APX

- **`gpu_batch_matmul_test.rs`**  
  - **Qué valida**: ruta GPU real de `BatchMatMulOp` usando `GpuMemoryEngine`.  
  - **Modo actual**: **smoke test** APX 12.x:
    - Compara longitud de salida vs referencia CPU.
    - Verifica que todos los valores son finitos.
- **`gpu_linear_test.rs`**  
  - **Qué valida**: ruta GPU de `LinearOp` (x @ W^T + b) usando `GpuMemoryEngine`.  
  - Usa `CompatLoader::is_forced_fallback()` para saltar el test si el loader cae a CPU.  
  - **Modo actual**: smoke test (longitud y finitud de valores, no igualdad exacta CPU).
- **`gpu_matmul_test.rs`**  
  - **Qué valida**: MatMul GPU directo vs referencia CPU.
- **`gpu_memory_test.rs`**  
  - **Qué valida**: asignación, copia H2D/D2H y liberación con `GpuMemoryEngine`.
- **`gpu_pipeline_test.rs`**  
  - **Qué valida**: pipeline GPU completo (compilación, carga de módulo, lanzamiento).
- **`gpu_runtime_test.rs`**, **`gpu_safety_test.rs`**, **`gpu_vec_ops_test.rs`**  
  - Varias validaciones del runtime GPU, safety checks y operaciones vectoriales básicas.

---

## Otros tests importantes (no APX)

- **Tensores y NN core**  
  - `tensor_test.rs`, `tensor_layout_test.rs`, `tensor_dtype_test.rs`, `tensor_ops_test.rs`, `tensor_fastpath_test.rs`, `nn_ops_test.rs`.
- **Entrenamiento / mini modelos**  
  - `mini_flux_*`, `mini_transformer_training_test.rs`, `trainer_test.rs`, `trainer_v2_test.rs`, `amg_*`, `amm_*`.
- **Infra general**  
  - `cpu_parallel_test.rs`, `apx_parallel_*`, `apx_trace_test.rs`, `nightly_check.rs`, `memory_manager_test.rs`, `gpu_runtime_test.rs`, `gpu_safety_test.rs`, `launch_engine_test.rs`, `dynamic_loader_test.rs`, `nvrtc_*`, `cuda_device_info_test.rs`, etc.

---

## Cómo extender esta guía

Cuando añadas nuevos tests:

- Nómbralos siguiendo `apx_<vers>_<subvers>_...`.
- Añade una entrada en esta guía con:
  - **Archivo de test**
  - **Versión APX**
  - **Qué valida en 1–3 líneas**
- Si son tests GPU realmente ejecutados en hardware, indica claramente si son:
  - **Correctness estricto**
  - **Smoke test / infraestructura**
  - **Benchmark** (no se chequean valores, sólo tiempos)
