# INVESTIGATION_M6 — Hallazgos del intento M6.a → M6.c.7 (revertido)

**Estado**: M6.a → M6.c (incluyendo los 5 hotfixes M6.c.7) **revertido completo**.
Tree restaurado a `M5.f.a` (commit `8b8253d`). Esta nota preserva los hallazgos
de investigación para que el próximo intento M6 no rehaga el trabajo desde cero.

---

## 1. Contexto

- **Objetivo M6**: 2-5 tok/s en Llama 2 13B Chat sobre RTX 4070 Laptop
  (8 GiB VRAM) + 32 GB RAM, partiendo del baseline M5.f.a de 14 s/tok (CPU).
- **Hardware del dev box**: RTX 4070 Laptop, 32 GiB RAM (8 GiB VRAM dedicada,
  16 GiB compartida). 24 hilos, AVX2+FMA, sin AVX-512. Driver 32.0.15.9621
  (DirectX FL 12.2).
- **Modelo de referencia**: Llama 2 13B Chat F32, 363 parámetros,
  24.24 GiB residentes en RAM tras carga BF16.

---

## 2. Sub-fases ejecutadas (commits revertidos)

| Sub-fase | Commit revertido | Contenido |
|---|---|---|
| M6.a | `f520b23` | Bench instrumentation per-NodeType (cfg-gated `bench-trace`) + R1 falsifier `examples/bench_cuda_matmul.rs` |
| M6.b | `49fb81c` | Lift double GPU gate (`gpu_available()` runtime detect + `cuda_matmul_non_pooled` con cudaMalloc directo) |
| M6.c.1 + .2 | `01d6d0e` | `Backend` trait vendor-neutral (`src/gpu/backend.rs`) + `LayerResidencyPlanner` (`src/gpu/residency_planner.rs`) — plumbing only |
| M6.c.3 + .4 | `05c9086` | `WeightStore::upload_resident_layers` + `SharedParam::Gpu` + mixed-storage matmul path `(a=Cpu, b=Cuda, out=Cpu)` |
| M6.c.7 fixes 1-3 | `3505c90` | Cache `cuda_available` (`OnceLock<bool>`) + reorder `ATENIA_GPU=0` antes de cuda_available + non-pooled restringido a residency |
| M6.c.7 fix 4 | `847bee6` | Revert `apx4::gpu_context::gpu_available` a hardcoded `false` |
| M6.c.7 fix 5 | `fca7544` | Revert `gpu::utils::gpu_enabled` a hardcoded `false` |

Todos los reverts conservan historial vía `git revert` (no `git reset`).

---

## 3. Resultados de bench (válidos — preservar)

### R1 — CUDA-vs-CPU falsifier (M6.a)

`examples/bench_cuda_matmul.rs` midió GPU vs CPU sobre tamaños matmul
representativos del hot-path Llama:

- **5120 × 5120 (Q/K/V/O proj)**: GPU 1.26× más rápido que CPU.
- **5120 × 13824 (FFN gate/up)**: GPU 2.10× más rápido.
- **13824 × 5120 (FFN down)**: GPU 2.75× más rápido.

**Conclusión**: el GPU compute es genuinamente más rápido en aislamiento.
El problema de M6 NO está en la velocidad del kernel — está en el coste
de orquestación (driver overhead, asignaciones, transferencias H↔D).

### ADR-004 — F64 4-model validation

Mantuvo 4/4 verde durante toda la sub-fase M6.a-M6.c. La validación numérica
no regresó.

---

## 4. Bug catastrófico encontrado (motivo del revert)

### Síntoma

Bajo `ATENIA_GPU=0` (kill-switch que desactiva todas las superficies GPU):

- M5.f.a baseline: **14 s/tok**, RAM ~26 GiB, sin pagefile thrashing.
- M6.c.7 (post fix 5): **278 s/tok** (835.5 s para 3 tokens), **94% RAM
  (29.7/31.7 GiB)**, **disco C: 100%**, GPU 0%.

20× regresión en el camino CPU **con todas las superficies GPU
demostrablemente desactivadas**.

### Diagnóstico (incompleto al momento del revert)

La captura del Administrador de Tareas mostró:
- GPU 0% (kill-switch verificado activo).
- RAM 94% (29.7 / 31.7 GiB) → cerca del límite físico.
- Disco C: al **100%** durante toda la generación.

**Conclusión**: M6 introdujo ~5-6 GiB de residencia adicional
(o no liberación de transitorios) que en una máquina de 32 GiB
empuja el sistema a paginar al pagefile de C:. La latencia 20×
no es un bug en el dispatch de matmul — es coste de pagefile thrashing.

### Hipótesis no investigadas (orden de probabilidad)

1. **Materialización F32 transitoria sin liberar**: algún path en
   M6.c.3/.4 (`ensure_cpu` sobre `CpuBf16Shared` para preparar
   subida que luego no ocurre con kill-switch on, pero el
   F32 transitorio queda referenciado en alguna estructura).

2. **`cuda_matmul_non_pooled`**: aunque restringido a residency en
   fix 3, el primer arranque del lazy `OnceLock` o del pool podría
   reservar memoria GPU mapeada a RAM compartida (16 GiB de los 23.9
   GiB totales del GPU son shared memory en este laptop) —
   el sistema operativo presta RAM al GPU, reduciendo lo disponible
   para el modelo CPU.

3. **Bench-trace cfg gate no es zero-overhead**: aunque se compiló sin
   `--features bench-trace`, alguna macro o `static` puede haber
   añadido overhead aún apagado.

4. **Cambio en `matmul_dispatcher::matmul_dispatch`**: poco probable
   (no se tocó en M6.a-c), pero merece ser descartado.

**Próximo intento M6 debe medir baseline DESPUÉS de cada activación**
para localizar el commit culpable en menos de 5 fixes incrementales.

---

## 5. Lecciones de proceso

### Qué falló en M6.c.7

- 5 hotfixes incrementales sin medir baseline entre ellos.
- Cada fix asumió que el bug estaba en una superficie GPU específica.
  Tras desactivar las 3 superficies (`try_gpu_matmul`, `dispatch_matmul_gpu`,
  `exec_gpu_segment`), el bug persistió → era CPU-path / RAM, no GPU.
- El último commit honesto donde el sistema funciona bien es M5.f.a.
  Cualquier auditoría hacia adelante debió empezar comparando
  M5.f.a vs M6.a directamente.

### Protocolo estricto para el próximo intento M6

1. **Una sola activación por commit** (no agrupar M6.c.3+.4 ni
   M6.c.1+.2 ni "fixes 1-3"; cada cambio observable es su propio commit).
2. **Medir baseline inmediatamente después** del commit, en la misma
   máquina del dev box, con `ATENIA_GPU=0` y con `ATENIA_GPU=1`.
3. **Rollback automático** si el baseline `ATENIA_GPU=0` regresa
   más de 1.5× sobre M5.f.a (14 s/tok).
4. **Registrar uso de RAM** en cada smoke (`Get-Process atenia |
   Select-Object WorkingSet64,PrivateMemorySize64`) además del
   tiempo de generación.
5. **Empezar por la activación más pequeña posible**: probablemente
   M6.f BF16 GPU kernel (un solo matmul GPU sobre weight ya residente),
   no la sub-fase M6.b de "lift double gate" que abre tres caminos
   a la vez.

---

## 6. Activos preservados (por si futuros intentos los reusan)

Aunque revertidos del `main`, los commits siguen accesibles vía
sus hashes:

- `examples/bench_cuda_matmul.rs` (R1 falsifier) — `f520b23`.
- `src/gpu/backend.rs` (Backend trait) — `01d6d0e`.
- `src/gpu/residency_planner.rs` (planner pure-fn) — `01d6d0e`.
- `src/cuda/matmul.rs::cuda_matmul_non_pooled` — `49fb81c`.
- `src/amg/weight_store.rs::upload_resident_layers` + `SharedParam::Gpu` — `05c9086`.

Los algoritmos del planner (split simétrico first-K + last-K') y
los bytes-per-layer F32 son correctos y reusables.

---

**Cierre**: M6 sigue abierto. M5.f.a es el baseline al que volvemos.
El próximo intento M6 empieza con la lección clara: orchestration
overhead y memory pressure son los riesgos reales en este hardware,
no la velocidad del kernel. Tracking en M6.f / v21 según se decida.
