# Atenia Engine — APX Modes

## 🟦 APX 6.9 / 6.10 — Fusion Profiling & Selector

- ✔ Introducen un `FusionSelector` global que mide tiempos de rutas fusionadas vs no fusionadas (ej. FusedQKV).
- ✔ Registran métricas por tipo de fusión (tiempos unfused vs fused) y derivan una preferencia global.
- ✔ No cambian los kernels base de MatMul/Linear/Attention; sólo deciden cuándo usar las variantes fusionadas existentes.
- ✔ Integrados de forma transparente en `Graph::execute` y en la ruta de Self-Attention.

**DISCLAIMER APX 6.9 / 6.10 — Fusion Profiling & Selector**

APX 6.9/6.10 no introducen nuevos kernels ni cambian la matemática de los ya existentes. Su objetivo es medir, de forma estructural, el coste relativo de ejecutar ciertos patrones de manera fusionada (por ejemplo, FusedQKV) frente a la ruta naive separada, y almacenar estas estadísticas en un selector global. Las decisiones derivadas afectan únicamente a qué variante de kernel se llama (fusionada o no), pero siempre respetando las mismas ecuaciones y la misma semántica de forward/backward. La cinta de autograd y la representación de tensores permanecen inalteradas.

---
## 🟦 APX 6.11 / 6.12 — Runtime Policy & Scheduling Bias

- ✔ Derivan una política de ejecución global (`FusionRuntimePolicy`) a partir de las estadísticas de 6.9/6.10.
- ✔ APX 6.11 fija una política determinista (PreferFull / PreferQKV / Baseline) usada como hint global.
- ✔ APX 6.12 añade un sesgo de scheduling (`AdaptiveScheduleBias`) para influir en planificadores posteriores.
- ✔ No modifican computación real, sólo establecen flags/hints de alto nivel.

**DISCLAIMER APX 6.11 / 6.12 — Runtime Policy & Bias**

APX 6.11/6.12 no añaden nuevos kernels ni modifican la matemática de los existentes. La política global y el sesgo de scheduling se usan exclusivamente como hints para seleccionadores posteriores (fusiones, orden de ejecución), sin tocar los valores de los tensores ni la estructura del grafo. Todas las rutas siguen produciendo resultados numéricamente equivalentes dentro de los umbrales validados, y la cinta de autograd no se ve afectada.

---

- ✔ Detecta automáticamente la carga real del sistema (uso de CPU, hilos disponibles) en tiempo de ejecución.
- ✔ Ajusta dinámicamente el paralelismo efectivo del MatMul: número de hilos utilizados por el scheduler PEX/WS y estrategia preferida (Seq / PEX / WS).
- ✔ Aprovecha APX 7.2/7.3 para delegar en PGL cuando no hay una preferencia fuerte dictada por la carga externa.
- ✔ Integrado en el scheduler PEX v2 y en la rama 6.3B del dispatcher para modos `ATENIA_APX_MODE >= "7.4"`.
- ✔ Tests específicos que simulan estados de carga alta/media/baja y bench ligero de diagnóstico.

**DISCLAIMER APX 7.4 — Dynamic Workload Adaptation**

APX 7.4 — Dynamic Workload Adaptation no cambia kernels, no altera la matemática, no toca forward/backward, ni modifica tensores reales. La adaptación se limita exclusivamente a seleccionar cuántos hilos y qué scheduler usar (Seq / PEX / WS) en base a la carga externa del sistema, pero siempre ejecutando los mismos operadores MatMul originales. Es totalmente segura, no invasiva y reversible.

---

## 🟦 APX 7.5 — HPGE (Hierarchical Parallel Graph Executor)

- ✔ Primer scheduler paralelo a nivel de grafo (`Graph`), por encima de `execute_single`.
- ✔ Construye una topología simple (padres/hijos) y ejecuta waves de nodos `ready`.
- ✔ Sólo reordena nodos independientes; respeta estrictamente las dependencias.
- ✔ Fallback seguro a `run_plan(true)` si detecta inconsistencias.

**DISCLAIMER APX 7.5 — HPGE**

APX 7.5 no introduce kernels nuevos, no cambia la matemática de los nodos, no altera backward, ni modifica la representación interna de tensores. El Hierarchical Parallel Graph Executor sólo decide el orden en el que se invoca `Graph::execute_single` sobre nodos que ya están listos (sin padres pendientes), respetando 100% las dependencias topológicas. Si en cualquier momento detecta un estado inconsistente (ciclos, nodos no ejecutados, ausencia de nodos ready en un grafo no vacío), hace fallback inmediato a la ejecución secuencial estándar mediante `run_plan(true)`.

---

## 🟦 APX 7.6 / 7.7 — HPGE v2 + HPFA (Critical-Path Optimizer)

- ✔ Extiende HPGE con un modelo de prioridad por nodo (Critical-Path Optimizer).
- ✔ Señales de prioridad: coste estimado, tamaño de subárbol, tiempo histórico y afinidad de fusión (HPFA).
- ✔ Integra Hot-Path Fusion Awareness (APX 7.7) como bonificación de prioridad; no toca fusiones reales.
- ✔ Mantiene las mismas garantías de HPGE: sólo reordena nodos independientes.

**DISCLAIMER APX 7.6 / 7.7 — HPGE v2 + HPFA**

APX 7.6/7.7 no introducen kernels nuevos ni cambian los existentes. No modifican backward, ni la cinta de autograd, ni la estructura del grafo. El Critical-Path Optimizer usa métricas estáticas (coste estimado, tamaño del subárbol) e históricas (tiempos medios por nodo) junto con la afinidad de fusión de APX 6.9/6.10 (HPFA) exclusivamente para ordenar el conjunto de nodos `ready`. Las dependencias topológicas se respetan en todo momento y, ante cualquier anomalía, el sistema cae de vuelta a `run_plan(true)`.

---

## 🟦 APX 7.8 — TLO (Temporal Locality Optimizer)

- ✔ Introduce hints de localidad temporal por nodo (`branch_id`, `depth`) en `Graph::build`.
- ✔ Reordena el conjunto de nodos `ready` según estos hints antes de ejecutar cada batch.
- ✔ No cambia el grafo ni sus dependencias; sólo el orden de nodos independientes.
- ✔ Se integra de forma transparente sobre HPGE / HPGE v2.

**DISCLAIMER APX 7.8 — TLO**

APX 7.8 no modifica kernels, no altera la matemática de forward/backward, ni cambia la representación de tensores. El Temporal Locality Optimizer se limita a reordenar nodos que ya son independientes basándose en hints de localidad sintéticos derivados de la estructura del grafo (profundidad aproximada y rama lógica). Todos los tests de equivalencia deben seguir pasando, y ante cualquier comportamiento sospechoso siempre es posible desactivar TLO o forzar el uso de `run_plan(true)`.

---

## 🟦 APX 7.9 — HLS (Hierarchical Locality Scheduler)

- ✔ Agrupa nodos en clusters jerárquicos según su firma estructural (número de padres/hijos).
- ✔ Refina clusters por localidad: inputs compartidos, padre común, mismo tipo de op.
- ✔ Ordena clusters por score, tamaño y fanout para guiar el scheduler de prioridades.
- ✔ No modifica kernels ni backward; sólo aporta un orden jerárquico adicional.

**DISCLAIMER APX 7.9 — HLS**

APX 7.9 no añade operadores nuevos ni toca la matemática de las operaciones existentes. El Hierarchical Locality Scheduler sólo construye una agrupación lógica de nodos (clusters) basada en información estructural del grafo y la usa como hint adicional para priorizar nodos `ready` dentro de HPGE v2. No se alteran dependencias, no se introducen nuevos caminos de ejecución, y el sistema mantiene la posibilidad de caer en `run_plan(true)` si la topología calculada resultara inconsistente.

---

## 🟦 APX 7.10 — HLS-Deep (Deep SuperLevel Scheduling)

- ✔ Calcula profundidades topológicas por nodo y construye SuperLevels por anchura/profundidad.
- ✔ Ejecuta SuperLevels secuencialmente y nodos dentro de cada uno en paralelo (o en batch ordenado).
- ✔ Integra TLO como heurística de orden dentro de SuperLevels anchos.
- ✔ Mantiene equivalencia numérica con la ejecución clásica / HPGE.

**DISCLAIMER APX 7.10 — HLS-Deep**

APX 7.10 no introduce kernels nuevos, no altera backward ni la forma de los tensores. El HLS Deep Pass sólo reagrupa nodos en "SuperLevels" y decide el orden de llamada a `execute_single` dentro de cada uno, respetando siempre las dependencias topológicas originales del grafo. Si el algoritmo de construcción de SuperLevels o el scheduler interno detectan una situación incoherente (por ejemplo, nodos que nunca llegan a estar listos), se activa un fallback inmediato a la ruta secuencial clásica (`run_plan(true)`), preservando la corrección.

---

## 🟦 APX 7.11 — PFLS (Predictive Future-Level Scheduling)

- ✔ Registra tiempos y congestión por SuperLevel en un historial ligero (`PFLSHistory`).
- ✔ Predice futuros "hotspots" (SuperLevels con mayor tiempo/congestión acumulados).
- ✔ Reordena nodos en SuperLevels previos para despejar cuellos de botella futuros.
- ✔ No usa datos reales del modelo; sólo métricas estructurales de ejecución.

**DISCLAIMER APX 7.11 — PFLS**

APX 7.11 PFLS no modifica kernels, matemáticas, backward ni la representación interna de tensores. El sistema observa únicamente tiempos de ejecución por SuperLevel y el número de nodos activos para predecir posibles cuellos de botella futuros y reordenar nodos independientes en niveles previos. Todas las dependencias topológicas se respetan estrictamente y, si la heurística predictiva detecta un estado inconsistente o no confiable, el sistema cae inmediatamente en la ruta segura de HPGE/HLS secuencial (`run_plan(true)`). PFLS nunca utiliza valores de tensores ni información de datos del modelo, sólo tiempos y congestión estructural.

---

## 🟦 APX 7.12 — ULE (Unified Level Executor)

- ✔ Unifica en un solo módulo las heurísticas de scheduling introducidas en 7.5–7.11.
- ✔ Usa SuperLevels (HLS-Deep), TLO, prioridades estructurales y PFLS en una misma ruta.
- ✔ Selecciona automáticamente la "estrategia" (Seq / PEX / Work-Stealing) más adecuada por SuperLevel.
- ✔ Reemplaza en tiempo de ejecución a HPGE/HLS/HLS-Deep/PFLS, manteniendo los módulos previos para tests.

**DISCLAIMER APX 7.12 — ULE**

APX 7.12 no incorpora kernels nuevos, no cambia la matemática, no altera backward ni la representación de tensores. El Unified Level Executor se limita a unificar la lógica de scheduling de APX 7.5–7.11 en un único módulo estructural que decide, por SuperLevel, cómo ordenar y ejecutar nodos independientes. Las dependencias topológicas del grafo se respetan en todo momento y, ante cualquier inconsistencia detectada, el sistema realiza fallback inmediato a la ejecución secuencial estándar (`run_plan(true)`). PFLS sigue usando exclusivamente tiempos y congestión estructural; no se emplea información de datos reales del modelo.

---

## 🟦 APX 8.x — GPU Simulation Stack (CPU-only)

- ✔ Introduce una cadena completa de **infraestructura GPU simulada**: desde IR y registros de kernels hasta planificación de dispositivo, partición, codegen y orquestación híbrida.
- ✔ Todos los módulos APX 8.x funcionan **exclusivamente sobre CPU**, generando strings y metadatos; **no hay llamadas reales a CUDA/HIP/Metal/Vulkan**, ni reservas de VRAM.
- ✔ No se modifican kernels CPU existentes, ni rutas de `Tensor`, ni backward/autograd; toda la lógica GPU es simbólica y reversible.
- ✔ Cada subversión 8.x añade una capa estructural: dual graph, dispatcher híbrido, estimadores de transferencia, mirror/persistencia, registro de kernels, IR, codegen mock, router multi-arch, planners de dispositivo/partición y un orquestador híbrido (HXO).

**DISCLAIMER APX 8.x — GPU Simulation Stack**

APX 8.x no ejecuta kernels GPU reales, no reserva memoria de dispositivo, no realiza transferencias H2D/D2H reales, no modifica la matemática de los kernels CPU ni altera backward ni la cinta de autograd. Toda la infraestructura de GPU introducida en 8.1–8.20 es puramente simbólica: construye grafos duales, estima costes de transferencia, mantiene espejos y metadatos de persistencia, registra kernels y firmas, genera IR y código mock (strings), simula routers multi-arch y planificadores de dispositivo/partición, y finalmente orquesta un "HybridOpPlan" mediante HXO. Los tensores reales siguen viviendo y operando en CPU; todos los tests de APX 8.x incluyen comprobaciones explícitas de **equivalencia numérica** frente a las rutas anteriores.

### APX 8.1–8.5 — Dual Graph, Hybrid Dispatcher y Estado GPU simbólico

- **8.1 DualGraph Builder (`apx8::dualgraph`)**
  - Construye un grafo dual CPU/GPU duplicando nodos y manteniendo mapeos entre ellos.
  - Uso: puramente estructural, base para dispatchers posteriores.
- **8.2 Hybrid Dispatcher (`apx8::hybrid_dispatcher`)**
  - `ExecDevice::{CPU,GPU}` y `HybridDispatcher::dispatch` deciden simbólicamente CPU vs GPU.
  - La ruta GPU llama a `exec_gpu_stub`, que internamente ejecuta la misma ruta CPU (`execute_single_inner`).
- **8.3 GPU Transfer Estimator (`apx8::gpu_transfer_estimator`)**
  - `GPUTransferEstimator::estimate` devuelve un `TransferEstimate` sintético según tamaño de tensor y `DevicePlacement`.
  - `HybridDispatcher::choose_device_for` usa ese criterio para mejorar la decisión CPU/GPU sin mover datos.
- **8.4 GPU Mirroring Layer (`apx8::mirror`)**
  - Añade `GPUMirror` y `MirrorState` como marca de espejo GPU en `Tensor` (campo `gpu: Option<GPUMirror>`).
  - No copia buffers a GPU; sólo marca estados logical-clean/dirty.
- **8.5 GPU Persistence Layer (`apx8::persistent`)**
  - Introduce `GPUPersistenceInfo` y un contador global de pasos para heurísticas de reutilización y limpieza del mirror.
  - Integrado en `Tensor` como `persistence: Option<GPUPersistenceInfo>`.

### APX 8.6–8.12 — Kernels simulados, Registry, IR y MetaLayer

- **8.6 GPU Kernels v0 (`apx8::gpu_kernels`)**
  - Define un kernel stub `gpu_vec_add` que opera sobre datos CPU y marca el mirror GPU como dirty.
  - Controlado por `RuntimeFlags::enable_gpu_kernels`.
- **8.7 Kernel Registry v1 (`apx8::kernel_registry`)**
  - `KernelRegistry`, `KernelKey`, `KernelFn` y `RegisteredKernel`.
  - Registra kernels CPU y stubs GPU, así como plantillas de kernels (`GpuKernelTemplate`).
- **8.8 GPU Kernel Signatures v0 (`apx8::gpu_kernel_signature`)**
  - Registra firmas textuales `GpuKernelSignature` para diferentes tipos de kernels GPU.
  - Sirve como capa de identificación/metadata, sin ejecución real.
- **8.9 GPU Kernel Generators v0 + 8.12 KernelIR (`apx8::kernel_generator`)**
  - V0: `GpuKernelOp`, `GpuKernelTemplate` → `GpuKernelIR` vía `to_ir()`.
  - 8.12: `KernelOp` y `KernelIR { ops, name, params }` con `mock_add`, `new_mock`, `hash`, `signature`.
  - Este `KernelIR` es la representación central usada por el resto de la pipeline GPU simulada.
- **8.12 GPU MetaLayer (`apx8::gpu_metalayer`)**
  - `optimize_ir(ir)` produce un `OptimizedIR` simbólico (p.ej., filtrando NOPs).
  - No altera ejecución real; sólo reescribe IR.

### APX 8.10–8.17 — Codegen Mock, Compiler Stub, Router y Finalizer

- **8.10 GPU Codegen Mock (`apx8::codegen_mock`)**
  - Implementa el trait `GpuCodegen` para backends sintéticos (CUDA, HIP, Metal) que producen código como string, sin compilación real.
- **8.11 GPU Compiler Stub (`apx8::gpu_compiler_stub`)**
  - `GpuCompilerStub` mantiene un cache de `CompiledKernelStub` indexado por `GpuTarget` y firmas.
  - Simula la existencia de un compilador/driver sin tocar hardware.
- **8.13 GPU Codegen v1 (`apx8::codegen::gpu_codegen_v1`)**
  - `GPUCodegenV1` genera código sintético a partir de `KernelIR` mediante `generate_kernel`.
  - `with_autoselect` usa `GPUAutoSelector` (8.14) y normaliza vendors (`"nvidia"`, `"amd"`, `"intel"`) a `"cuda"`, `"hip"`, `"metal"`.
  - `codegen_with_cache` integra `PrecompileCache` (8.15) como cache de strings.
  - `codegen_multiarch` usa `route_kernel` (8.16) para anotar la arquitectura seleccionada.
  - `codegen_with_finalizer` combina `generate_kernel` con `gpu_finalize` (8.17).
- **8.14 GPU Auto-Selector v0 (`apx8::gpu_autoselector`)**
  - `GPUAutoSelector::detect` y `choose_backend(ir)` seleccionan un backend textual según el nombre del IR.
  - No inspecciona hardware real; es un heurístico determinista.
- **8.15 Pre-Compilation Cache v0 (`apx8::precompile_cache`)**
  - `PrecompileCache` guarda strings `compiled::<signature>` por `KernelIR`.
  - Usado en rutas como `codegen_with_cache` para simular kernels ya compilados.
- **8.16 Multi-Arch Kernel Routing v0 (`apx8::multiarch_router`)**
  - `TargetArch` y `route_kernel(ir)` determinan simbólicamente si el IR iría a CPU, CUDA, HIP, Metal o Vulkan según su firma.
- **8.17 GPU Finalizer Stub (`apx8::gpu_finalizer`)**
  - `gpu_finalize(ir)` genera strings como `"FINALIZED CUDA KERNEL for {name}"` o `"CPU fallback"` según `TargetArch`.
  - Se integra en `GPUCodegenV1::codegen_with_finalizer` como última etapa simbólica.

### APX 8.18–8.20 — Device Planner, Partitioning Simulator y HXO

- **8.18 GPU Device Planner v0 (`apx8::device_planner`)**
  - `SimulatedGPU` y `DevicePlan` describen dispositivos y hints de split totalmente ficticios.
  - `detect_simulated_gpus` devuelve GPUs fake (ej. `FakeCUDA_4090`).
  - `plan_for_ir(ir_name)` decide simbólicamente si un IR se asociaría a una GPU concreta o a CPU.
- **8.19 GPU Partitioning Simulator (GPS) (`apx8::gpu_partition`)**
  - `PartitionPolicy` y `PartitionPlan` definen políticas 1D/2D/Auto basadas en la forma de un tensor (`shape`).
  - `suggest_partition(shape)` devuelve una política y un `estimated_speedup` puramente simbólico.
  - `HybridDispatcher::choose_gpu_strategy(shape)` usa esta función para describir una estrategia, sin particionar datos reales.
- **8.20 Hybrid Execution Orchestrator (HXO) (`apx8::hxo`)**
  - `HybridOpPlan` agrega en una sola estructura: `device`, `partition`, `backend`, `codegen` y un flag `precompiled`.
  - `build_hxo_plan(ir, shape)` orquesta:
    - planner de dispositivo (8.18), planner de partición (8.19), router multi-arch (8.16), cache de precompilación (8.15) y codegen+finalizer (8.13+8.17).
  - `hybrid_dispatch(ir, shape)` expone un `HybridDispatchResult::Pseudo` sólo con metadatos, sin tocar `Graph` ni `Tensor`.

En conjunto, APX 8.x prepara una cadena GPU completa a nivel de arquitectura —grafo dual, dispatch híbrido, IR, registros, codegen, planners y orquestador— mientras mantiene la ejecución real estrictamente en CPU y numéricamente equivalente a las versiones previas.

---

## 🟦 APX 9.x — Virtual GPU Pipeline & SM Model

La fase APX 9.x está **completamente implementada** de la 9.1 a la 9.25 e introduce una pila de simulación GPU de nivel arquitectural, todavía 100% CPU-only:

- **APX 9.1–9.5**
  - IR GPU de alto nivel, generación de PTX simbólico y toolchain de validación/optimización (PTX emitter, validator, traductor SASS, optimizador SASS).

- **APX 9.6–9.10**
  - Planificador de memoria GPU simulado, planificador de ejecución, ejecutor GPU mock, autotuner basado en tiempos simulados y codegen GPU real stub (orientado a APX 10).

- **APX 9.11–9.14**
  - Traductor CPU→PTX, ejecutor vGPU de alto nivel, modelo de memoria virtual (`VGpuMemory`) y runner vGPU (`VGpuRunner`) que ejecuta el IR sobre CPU.

- **APX 9.15–9.19**
  - Lanzador de bloques (`VGpuBlockLauncher`), capa de sincronización (`VGPUBarrier` / `VGPUBlockContext`), modelo SIMT de warp (`VGPUWarp`), scheduler de warps y pila de divergencia/reconvergencia (`WarpMask`, `DivergenceStack`).

- **APX 9.20–9.23**
  - Pipeline SIMT Fetch/Decode/Execute (`VGPUPipeline`), scoreboard de registros (`VGPUScoreboard`), out‑of‑order warp scheduler (`VGPUOOWarpScheduler`) y unidad de dual‑issue simbólica (`VGPUDualIssue`).
