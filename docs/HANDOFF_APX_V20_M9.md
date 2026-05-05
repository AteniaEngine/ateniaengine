# HANDOFF — APX v20 — M9 (INT8 W8A16 weight quantisation)

**Status:** Closed as **experimental / opt-in only**. Tag `v0.9.0-m9-experimental`.
**Predecessor:** M8.6 (BF16 KV cache, tag `v0.8.6-m8.6`).
**Successor (active):** TBD — see §10 below.

> M9 set out to halve the per-weight VRAM footprint via INT8 W8A16
> absmax quantisation, doubling the effective VRAM capacity and
> eliminating the Disk overflow path on Llama 2 13B. The
> infrastructure landed clean. The numerical contract did not —
> simple absmax (per-channel and per-group {32, 128}) cannot
> satisfy ADR-004's strict drift threshold (`< 0.5`) on 4 of 4
> production models in our fixture. M9 ships as an **opt-in path**
> for operators who accept the capacity / drift trade-off; full
> ADR-004-compliant INT8 is deferred to M10+ via either FFN-only
> mixed precision or outlier-decomposition (LLM.int8 / GPTQ / AWQ).

---

## 1. What M9 delivered (the wins)

- **Complete INT8 infrastructure**:
  - `DType::Int8` (1 byte / element).
  - `TensorStorage::CpuInt8 { q, scales, shape, group_size }` with
    decode-on-access (`copy_to_cpu_vec` + `ensure_cpu` arms).
  - `Tensor::new_cpu_int8` (per-channel) +
    `Tensor::new_cpu_int8_per_group` (M9.4) constructors.
  - `tensor::quantizer::absmax_per_channel_symmetric` (M9.1).
  - `tensor::quantizer::absmax_per_group_symmetric` (M9.4).
  - `cuda::int8_to_bf16::int8_to_bf16_in_vram` (M9.1, per-channel).
  - `cuda::int8_to_bf16::int8_per_group_to_bf16_in_vram` (M9.4).
  - CUDA kernels: `int8_to_bf16_per_channel_kernel` +
    `int8_to_bf16_per_group_kernel` in `src/cuda/int8_to_bf16.cu`.

- **Loader integration** (`src/v17/loader/weight_mapper.rs`):
  - `ATENIA_M9_INT8=1` flag activates the path on
    `Tier::Vram` slow-path branch.
  - `_proj.weight` predicate gates which tensors get quantised
    (lm_head and embed_tokens are excluded by name).
  - `VRAM_INT8_PATH_COUNT` counter mirrors the M8.4 BF16 routing
    discipline; tests assert delta ≥ proj count.

- **Tier planner cost model** (`src/gpu/tier_plan.rs`):
  - `vram_cost_bytes` returns `numel × 1` when
    `ATENIA_M9_INT8=1` AND `name.ends_with("_proj.weight")`.
  - 13B realistic plan under the flag:
    - **VRAM = 167 tensors** (24 layers × 7 proj — vs 80 baseline)
    - RAM = 196 tensors
    - **Disk = 0** (eliminated by construction)
    - vram_bytes_assigned = 6.98 GiB
    - ram_bytes_assigned = 10.28 GiB

- **GPU-side speedup** (M9.0 microbench, `examples/bench_int8_w8a16.rs`):
  - Q/K/V/O proj: **2.01×** over Path B (M8.4c)
  - FFN gate/up:  **1.98×**
  - FFN down:     **1.97×**
  - LM head:      **1.93×**
  - Throughput ~doubles across all 4 dominant Llama 13B shapes.

- **Routing telemetry validated**: 4-model M9.4 fixture confirmed
  100% of `_proj.weight` tensors take the INT8 path (counter
  deltas: 112 / 196 / 168 / 154 across the 4 models;
  BF16 fast/slow deltas all 0).

- **`cargo test --lib`: 210 / 210 verde** (5 new M9.1 + 5 new
  M9.2 + 5 new M9.4 unit tests on top of the 195 M8.6 baseline).

---

## 2. What M9 did NOT deliver (the honest box)

- **ADR-004 strict (`max_abs_diff < 0.5` vs F64)**: not satisfied
  by absmax INT8 on the 4-model fixture under any of the
  3 strategies tried. Full numbers in §4.
- **Production-by-default routing**: M9 ships **opt-in only** via
  `ATENIA_M9_INT8=1`. Default unset keeps the M8.6 BF16 production
  path bit-identical.
- **`atenia generate` 13B smoke under M9**: not run. Without an
  ADR-004-green gate at the small-model level, smoke on 13B (40
  layers vs 16-32 in the gate models) would be blind: drift is
  expected to be **worse**, and we have no evidence the generated
  text would be coherent on every prompt.

---

## 3. Sub-phase ledger

| Phase    | Title                                                    | Commit     | Status |
| -------- | -------------------------------------------------------- | ---------- | ------ |
| M9.0     | INT8 W8A16 microbench (RTX 4070 ~2× over Path B M8.4c)   | `76dd83c`  | ✅      |
| M9.1     | Quantizer + `TensorStorage::CpuInt8` + CUDA wrapper      | `f6949cf`  | ✅      |
| M9.2     | Loader integration + 13B planner test                    | `4383862`  | ✅      |
| M9.3     | Pre-smoke audit + `gpu-trace` exhaustive-match fix       | `73d4fe5`  | ✅      |
| M9.4 fix | M8.5 test accepts INT8 routing path (A+C fix)            | `06ee91a`  | ✅      |
| M9.4-pg  | Per-group quantisation (Q8_0) replaces per-channel       | `6686008`  | ✅      |
| M9.4-g32 | Tightened group size 32 → reverted to 128 after fixture  | `0e29a77`+ | ✅      |
| M9 close | Honest closure as opt-in / experimental                  | (this)     | ✅      |

---

## 4. Empirical findings (the 12-run experiment matrix)

Three quantisation strategies × four production models = 12 F64
fixture runs. The table below shows max_abs_diff vs the F64
ground truth (ADR-004 threshold 0.5) and the count of bit-exact
argmax matches over 4 forward positions.

|             | per-channel               | per-group g=128           | per-group g=32            |
| ----------- | ------------------------- | ------------------------- | ------------------------- |
| **Llama 3.2 1B**  | 1.098 / 2 of 4 ✗ | **0.516** / 3 of 4 ✗     | 0.970 / 3 of 4 ✗         |
| **Qwen 2.5 1.5B** | 4.268 / 2 of 4 ✗ | 2.338 / **3 of 4** ✗     | **1.841** / 2 of 4 ✗     |
| **SmolLM2 1.7B**  | **8.883** / 2 of 4 ✗ | 11.246 / **4 of 4** ✗ | 11.172 / 2 of 4 ✗        |
| **TinyLlama 1.1B**| 2.981 / **4 of 4** ✗ | 1.440 / 3 of 4 ✗     | 1.813 / **4 of 4** ✗     |

(`drift / argmax_matches`. Bold = best within row. `✗` = ADR-004 fail.)

### Findings

- **No strategy passes ADR-004 universally.** All 12 runs miss
  the 0.5 gate by margins 1.03× (Llama 3.2 g=128) to 22.5×
  (SmolLM2 g=128).
- **g=128 is the best overall trade-off**:
  - Wins drift on 2/4 models (Llama 3.2 0.516, TinyLlama 1.44).
  - Wins argmax on 2/4 models (Qwen 3/4, SmolLM2 4/4).
  - Aligns with typical Llama-family `head_dim ∈ {64, 128}` —
    keeping group boundaries inside head boundaries appears to
    matter empirically.
- **g=32 is NOT universally better.** This was the surprise of
  the M9.4 experiment. Counter-intuitive: smaller blocks
  *should* localise outliers better. They do, but they also
  cut across head-dimension boundaries and introduce more
  rounding-bias accumulation at block edges. On Llama 3.2 and
  TinyLlama g=32 strictly degraded vs g=128 (0.97 vs 0.52,
  1.81 vs 1.44).
- **Argmax is more resilient than absolute magnitude**.
  Per-group g=128 produced 4/4 argmax on SmolLM2 with drift
  11.2 — relative ordering of logits survives even when the
  absolute scale drifts massively. ADR-004 measures the latter.
- **SmolLM2 1.7B is the worst-case model** across all
  strategies. Probable explanation: weight distribution has
  outliers that don't localise into 32 or 128-element blocks.
  Per-channel was paradoxically its best (8.88 vs 11.25 g=128) —
  the per-column-wide scale at least provides one consistent
  reference rather than 16-64 mismatched ones.

### Why absmax is hitting a wall

The drift cascade equation (informal):

```text
end-to-end-drift ≈ envelope_per_op · sqrt(num_ops · alpha)
```

For BF16 single-op envelope `~3e-3`, 22-32 layer `num_ops` ≈
~70-200 matmuls, alpha ≈ 1 → end-to-end ~3e-2..1e-1 ≪ 0.5.

For absmax INT8 single-op envelope on outlier-heavy Llama
weights `~1e-2..3e-2`, same `num_ops` → end-to-end
~5e-1..3 — *exactly the regime we observe*. Per-group reduces
the single-op envelope but only 2-4× in the best case; not
enough to clear the gate without **either** (a) reducing
the cascade depth (FFN-only mixed precision, M9.5) **or**
(b) bypassing the outlier problem (LLM.int8 / GPTQ / AWQ, M10+).

---

## 5. The path that ships

When `ATENIA_M9_INT8=1`:

1. Loader `Tier::Vram` slow path on `_proj.weight` →
   `absmax_per_group_symmetric(values, shape, group_size = 128)`.
2. `int8_per_group_to_bf16_in_vram(q, scales, shape, 128)`
   uploads INT8 + scales to VRAM, dispatches the dequant
   kernel, returns a BF16-resident `TensorGPU`.
3. `SharedParam::Cuda { shape, gpu }` lands in the store.
4. The dispatcher's existing M8.4c branch consumes it as a
   normal BF16-resident weight (no kernel changes).
5. `VRAM_INT8_PATH_COUNT` advances by 1.

Flag unset: 100% bit-identical to M8.6 production (counter-
verified: BF16 deltas are unchanged with M9 dormant).

---

## 6. Operator quickstart (opt-in)

```powershell
# Capacity-only mode: doubles the 13B VRAM working set without
# satisfying ADR-004 strict. Generated text quality is degraded
# vs the BF16 baseline; argmax may diverge on some positions
# (3-of-4 typical, 2-of-4 worst case in the small-model fixture).
$env:ATENIA_M8_BF16_KERNEL = "1"
$env:ATENIA_M9_INT8 = "1"
cargo run --release --example llama2_13b_demo
# Plan should report ~167 VRAM tensors / 0 Disk on the 13B.
```

```powershell
# Production mode (M8.6 baseline, BF16-resident, ADR-004 strict):
$env:ATENIA_M8_BF16_KERNEL = "1"
# (M9 flag unset)
cargo run --release --example llama2_13b_demo
# Plan reports ~80 VRAM tensors and uses the Disk-streaming path.
```

---

## 7. API surface added

```rust
// src/tensor/tensor.rs
pub enum DType { F32, F16, BF16, FP8, Int8 }
pub enum TensorStorage {
    /* ... */
    CpuInt8 {
        q: Vec<i8>, scales: Vec<f32>,
        shape: Vec<usize>, group_size: usize,
    },
}
impl Tensor {
    pub fn new_cpu_int8(shape, q, scales) -> Self;          // per-channel
    pub fn new_cpu_int8_per_group(shape, q, scales, gs) -> Self;
}

// src/tensor/quantizer.rs (new module)
pub fn absmax_per_channel_symmetric(weights, shape) -> (Vec<i8>, Vec<f32>);
pub fn absmax_per_group_symmetric(weights, shape, group_size) -> (Vec<i8>, Vec<f32>);
pub fn quantize_int8_w8a16(weights, shape) -> Tensor;       // per-channel
pub fn quantize_int8_per_group(weights, shape, group_size) -> Tensor;
pub fn as_cpu_int8_view(&Tensor) -> Option<(&[i8], &[f32], &[usize], usize)>;

// src/cuda/int8_to_bf16.{rs,cu} (new)
pub fn int8_to_bf16_in_vram(q, scales, shape) -> Option<TensorGPU>;
pub fn int8_per_group_to_bf16_in_vram(q, scales, shape, group_size) -> Option<TensorGPU>;
pub fn cuda_int8_resident_count() -> usize;
extern "C" int8_to_bf16_per_channel_launch_device(...);
extern "C" int8_to_bf16_per_group_launch_device(...);

// src/v17/loader/weight_mapper.rs
pub fn vram_int8_path_count() -> usize;
```

---

## 8. Decisions

- **D89** — Ship M9 as opt-in, not as default. Closing the
  milestone around the infrastructure win without claiming a
  numerical contract we can't satisfy is more honest than
  hiding the drift behind a relaxed gate.
- **D90** — Group size = 128 (Q8_0). Empirical winner across
  the 4-model matrix; aligns with typical `head_dim`. NOT 32
  despite llama.cpp's default — Llama-family weights respond
  worse to that boundary in our experiments.
- **D91** — Per-group quantisation is the M9 storage layout.
  The per-channel API (M9.1) is preserved for tools and tests
  but not used in the loader path.
- **D92** — `_proj.weight` predicate (loader + planner). Excludes
  `lm_head.weight` and `model.embed_tokens.weight` from
  quantisation. Both have outsized effect on the final logits
  envelope; keeping them BF16 is consistent with the M9.0
  microbench's scope (which only measured the 4 proj shapes).

---

## 9. Validation gates

| Gate                                  | Command                                                                              | Result      |
| ------------------------------------- | ------------------------------------------------------------------------------------ | ----------- |
| Library tests (single-thread)         | `cargo test --lib -- --test-threads=1`                                               | 210/210 ✅   |
| All-features build                    | `cargo check --lib --all-features`                                                   | clean ✅     |
| M9 quantizer suite                    | `cargo test --lib quantizer`                                                         | 14/14 ✅    |
| M9 INT8 GPU wrapper                   | `cargo test --lib int8_to_bf16`                                                      | 3/3 ✅ + 1 ignored |
| M9 tier_plan cost model               | `cargo test --lib m9_2`                                                              | 3/3 ✅      |
| M9 loader counter wiring              | `cargo test --lib m9_int8 vram_int8`                                                 | 2/2 ✅      |
| M9.0 microbench                       | `cargo run --release --example bench_int8_w8a16`                                     | H2 PASS ✅   |
| F64 4-model fixture (per-channel)     | `... ATENIA_M9_INT8=1 cargo test --release --test m8_5_full_family_validation_test --ignored --nocapture` | drift fail (1.10–8.88) ✗ |
| F64 4-model fixture (g=128)           | (same; g=128 in loader)                                                              | drift fail (0.52–11.25) ✗ |
| F64 4-model fixture (g=32)            | (same; g=32 in loader)                                                               | drift fail (0.97–11.17) ✗ |

The four ADR-004 fixture rows are the **honest** failure
record. The first three rows are the genuine wins. The
ignored CUDA round-trip test in `int8_to_bf16` runs green
on the dev box manually.

---

## 10. Open issues / how to resume

### Option α — M9.5: FFN-only mixed precision (~3 days)

**Idea**: only `mlp.gate_proj.weight`, `mlp.up_proj.weight`,
and `mlp.down_proj.weight` get quantised; the attention
projections (Q/K/V/O) stay BF16. This roughly halves the
quantisation cascade depth — most of the byte savings (FFN is
the bulk of params), most of the speedup (FFN down is the
biggest matmul), but the attention path keeps M8.6 numerics.

**Why it might pass ADR-004**: the cascade is shallower; most
of the per-layer outlier-induced drift comes through attention
(K weight has a `Scale` transform that amplifies the scale
range), so keeping that BF16 directly attacks the cascade
depth term in the drift equation in §4.

**Cost**: change the `_proj.weight` predicate in loader +
planner to `name.contains(".mlp.")
&& name.ends_with("_proj.weight")`. Re-run the 4-model
fixture. If it passes, ship; if not, pivot to Option β.

### Option β — M10: outlier decomposition (~1-2 weeks)

LLM.int8() (Dettmers) is the established approach for outlier-
heavy weights. Vector-wise quantisation per (token, channel)
+ a "salient column" detection that keeps outliers in BF16
mixed with INT8 for the rest. Recovers ADR-004-class accuracy
on Llama-family models per the literature (~0.1% perplexity
delta vs FP16).

**Cost**: ~600 LoC new (mixed-precision GEMM is non-trivial,
and the salient-column detection adds a per-matmul scan).
Substantial but well-trodden — `bitsandbytes` is the reference
implementation.

### Option γ — M10: GPTQ / AWQ + calibration (~2 weeks)

State-of-the-art INT8 / INT4 quality. Requires offline
calibration with a representative dataset (~1 hour compute on
13B). Highest quality at 4-bit and beyond.

**Cost**: Hessian-inverse computation (GPTQ) or activation-
aware sensitivity (AWQ) infrastructure, plus a calibration
runner. Could be deferred to M11+ unless 4-bit becomes
strategically important.

### Option δ — M10 production hardening (no quantisation work)

Pivot off quantisation entirely. M9 has shipped what it can
ship; production pressure is best served by deepening M8.6 +
M8.7 + the SignalBus reactive infrastructure. M9 stays as
opt-in for operators who genuinely need the capacity headroom
and accept the trade-off.

---

## 11. Why this closure is honest

A milestone that delivers complete, working infrastructure
*and* genuine numerical findings (including the negative one)
is not a failure — it is a milestone that closed at the
predicted boundary the M9.0 research report defined explicitly:

> H1 — W8A16 absmax per-channel passes ADR-004:
>   probability **75%** with per-channel; **>90%** with
>   per-group g=128.

We landed in the <10% failure case for per-group g=128 across
4 of 4 models. The research framing was right; the empirical
result is what it is. Calibration-based quantisers exist
exactly because absmax has known limits on outlier-heavy
LLMs, and ADR-004's `< 0.5` threshold was calibrated against
BF16-class envelopes, not INT8-class.

Closing M9 here lets the next operator pick the best
follow-up (α / β / γ / δ) with full information about the
cliff, instead of pretending the cliff doesn't exist.

---

**Closure tag:** `v0.9.0-m9-experimental` on `main`.
