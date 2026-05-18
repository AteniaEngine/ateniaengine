# Milestones

A chronological narrative of every shipped milestone in Atenia Engine, from the
first native graph primitives (M1) through the GGUF production unlock (M11.D.5),
the adapter-layer migration (Phases 11–15), and the M12 operability-hardening
series.

This document is the *history*. For where the engine stands **right now** —
what is cabled to production signals versus what is still scaffolding — see
[STATUS.md](./STATUS.md). For the *direction*, see [ROADMAP.md](../ROADMAP.md).
Each milestone closes with a `HANDOFF_APX_V20_M*.md` deep dive in this
directory; the relevant one is linked inline.

> **A note on the hour estimates.** Where the roadmap attaches "hours" to a
> sub-phase, treat them as *story points* — useful to compare two items,
> useless to add into a calendar. Actual execution has consistently run
> dramatically faster than the estimates.

---

## APX v20 — Real Model Runtime Integration

APX v20 connects the completed telemetry and decision infrastructure (v12–v19)
to real external model execution. Earlier versions are complete; the narrative
below starts where real checkpoints enter the picture.

### M1 — Native Conv2D / MaxPool2D

`Conv2D` and `MaxPool2D` landed natively in the Adaptive Model Graph (AMG) with
forward, backward, tape integration, and finite-difference gradient checking.

### M2 — Reactive execution context

A reactive execution context was attached to the graph. The executor consults
guard state before each node and returns typed abort reasons on guard verdicts.
Existing APIs were preserved as backward-compatible wrappers.

### M3 — Real GPU allocation

Real GPU allocation behind a vendor-neutral storage abstraction
(`TensorStorage`), real host↔device transfers, and the M3-e reaction loop that
moves real VRAM to RAM on guard `Degrade` verdicts.
See [HANDOFF M3](./HANDOFF_APX_V20_M3.md).

### M4 — Safetensors loader mechanics

A `safetensors` reader (header + body, by-name and iterator access), a
`WeightMapper` with shape validation and structured `LoadReport` diagnostics,
and BF16 / F16 → F32 decode. Validated against a real HuggingFace gpt2
checkpoint. See [HANDOFF M4](./HANDOFF_APX_V20_M4.md).

### M4.5 — End-to-end real model execution

The engine loaded a HuggingFace `TinyLlama-1.1B-Chat-v1.0` checkpoint and ran
forward on CPU. New graph primitives: rotary positional embedding, general
permute, broadcast multiplication, rank-4 batched matmul. A complete Llama-2
graph builder consumes the HuggingFace parameter naming convention directly.
See [HANDOFF M4.5](./HANDOFF_APX_V20_M4.5.md).

### M4.6 — Llama-family compatibility expansion

Three production checkpoints added on top of TinyLlama:

- **SmolLM2 1.7B** (Phase A) — tied word embeddings, configurable RmsNorm eps.
- **Qwen 2.5 1.5B** (Phase B) — Q/K/V projection biases, `model_type`-aware
  config defaults.
- **Llama 3.2 1B Instruct** (Phase C) — `rope_scaling: "llama3"` piecewise
  frequency scaling, explicit `head_dim`.

Each model was validated against PyTorch **F64 mathematical ground truth** per
[ADR-004](./decisions/ADR-004-f64-reference-as-default.md), with Atenia F32 max
drift between 1.32×10⁻⁴ and 1.45×10⁻³ — three to four orders of magnitude
closer to truth than industry-default BF16 inference on the same checkpoints.
Argmax MATCH 4/4 positions on every model. **M4.6.1** added retroactive F64
validation for TinyLlama (drift 1.41×10⁻⁴). These four small models
(TinyLlama, SmolLM2, Qwen 2.5 1.5B, Llama 3.2 1B) became the permanent F64
regression fixture exercised under every later numerical-contract milestone.
See [HANDOFF M4.6](./HANDOFF_APX_V20_M4.6.md).

### M4.7 — Beyond-VRAM

The killer demo for the v20 thesis: run a 13B-class model in BF16 end-to-end on
the dev hardware (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, NVMe). A 13B model in
BF16 (~26 GB on disk) fits in neither VRAM nor RAM alone, but is executable via
VRAM ↔ RAM ↔ NVMe offload.

Six sub-phases, each closed under budget:

- **M4.7.1** — Sharded safetensors loader (multi-file +
  `model.safetensors.index.json`, drop-after-decode RAM bound). Mistral 7B
  v0.3: 291 tensors / 3 shards / 14.5 GB BF16 loaded with peak RAM bounded per
  shard.
- **M4.7.2** — Native BF16 parameter storage with decode-on-access
  (`TensorStorage::CpuBf16`). 50.0 % RAM savings, drift bit-exact to the
  precision-floor spike across the four-model fixture.
- **M4.7.3** — GPU MatMul with resident operands + executor device dispatch.
  `cuda_matmul_inplace`, `Tensor::zeros_new_cuda`; defensive `ensure_cpu` /
  `ensure_decoded` audit on every executor arm.
- **M4.7.4** — RAM↔SSD streaming primitive: BF16-aware disk format, chunked
  streaming reader (4 MiB chunks, no whole-file read), dtype-aware `ensure_cpu`
  Disk arm. 50 % saving on disk too; spill + restore mathematically
  transparent.
- **M4.7.5** — M3-e policy upgrade: per-tensor LRU eviction inside the
  `DeepDegrade` arm (`SPILL_FRACTION = 0.5`), probe-cache amortisation audit
  (100 ms TTL), `ensure_cpu` consumer-side audit closure.
- **M4.7.6** — Llama 2 13B Chat end-to-end **momento guau**. The transparency
  contract holds bit-exactly at 13B scale:

  | Mode | argmax | logit | Notes |
  |------|-------:|------:|-------|
  | A — clean RAM | 1 | 4.7747 | No reactive context. |
  | B — autonomous LRU spill | — | — | 4 DeepDegrade events, 26 GB spilled. |
  | C — forced 50 % LRU spill | **1** | **4.7747** | 866/1732 LRU entries spilled. **argmax(C) == argmax(A) bit-exactly.** |

See [HANDOFF M4.7](./HANDOFF_APX_V20_M4.7.md).

### M4.8 — Performance optimisation

The momento guau wall-clock was impractical for public reproduction. M4.8
delivered **3.49× on the 13B Mode A forward** (18.75 min → 5.38 min) and
**49.5× on the production matmul shape** (`4×5120×13824`), via: lifting the
default `apx_mode` to `7.2`, numeric segment comparison, runtime AVX2 dispatch,
a SIMD BF16 decode kernel (2.76×), rayon batch/row partitioning, and
`matrixmultiply 0.3` (vendor-agnostic BLIS-style sgemm, **no MKL**). Drift
*improved* on all four fixture models. See
[HANDOFF M4.8](./HANDOFF_APX_V20_M4.8.md).

### M4.9 — Public CLI demo

The single-binary `atenia` CLI (`clap` derive). `atenia run --mode c`
reproduces the momento guau bit-exact in **≈6.9 min** via one command on the
dev box. `atenia probe` replaced the legacy `hardware_probe` binary (parity
verified bit-exact). Full CLI reference: [CLI.md](./CLI.md).
See [HANDOFF M4.9](./HANDOFF_APX_V20_M4.9.md).

### M5 — Tokenizer + KV cache + token-by-token generation

**Atenia chats.** `atenia generate --prompt "Hello, how are you?" --model
models/llama-2-13b-chat` returns a coherent conversational reply. 24.24 GiB
resident for BF16 13B with two graphs sharing weights via
`Arc<TensorStorage>` (vs ~52 GiB naïve). R2 graph-level falsifier (cache-aware
attention bit-exact to no-cache), R6 generation contract, D67 determinism
fixture all green. See [HANDOFF M5](./HANDOFF_APX_V20_M5.md).

### M6 — Tier-aware GPU loader

The tier-aware loader (VRAM → RAM → NVMe) routed 60 projection weights of
Llama 2 7B Chat to VRAM at load time; the rest stayed in RAM; bit-identical
output; **1.46× faster** end-to-end (8.22 vs 12.02 s/tok). The planner is a
pure function of `(metadata, free_ram, free_vram)`; placement happens at load
time, never as post-load migration (see
[INVESTIGATION_M6_REPLAN.md](../INVESTIGATION_M6_REPLAN.md)).
See [HANDOFF M6](./HANDOFF_APX_V20_M6.md).

### M7 — 13B-friendly tiers

Disk fast-path (raw BF16 bytes → NVMe, zero F32 transient) + adaptive RAM
headroom. **Llama 2 13B Chat ran end-to-end** with 38 tensors on VRAM, 126 on
RAM, 239 on NVMe, 7.36 GiB free RAM throughout, no BSOD, 36.6 s/tok.
See [HANDOFF M7](./HANDOFF_APX_V20_M7.md).

### M8 — BF16-resident VRAM kernels

**Path B**: BF16 weight resident in VRAM + F32 upcast per-matmul +
`cublasGemmEx(F32, F32, F32)`. Llama 2 7B **1.31×** over M6; Llama 2 13B
**1.36×** over M7.3. F64 four-model contract preserved (drift 4.0e-5 to
2.40e-2, all under the 0.5 gate with 21–12,500× margin). The Path B upcast
eliminated the per-matmul activation truncation that cascaded through layers in
the rejected M8.4-original BF16-input path. See
[HANDOFF M8](./HANDOFF_APX_V20_M8.md).

### M8.6 — BF16 KV cache

Runtime ledger casts F32 → BF16 on harvest, BF16 → F32 on reinject; the graph
stays F32. TinyLlama 1.1B-Chat 8-token determinism fixture bit-identical under
the BF16 ledger, so the default flipped on. 1.6 GiB saved at seq=2048 on 13B.
Opt-out: `ATENIA_LEGACY_F32_KV_CACHE=1`. See
[HANDOFF M8.6](./HANDOFF_APX_V20_M8.6.md).

### M8.7 — Disk → GPU JIT pipeline

154 disk-tier weights stream per forward through the BF16 GPU dispatch with a
98.7 % CPU prefetch hit rate, dropping Llama 2 13B from 27.0 s/tok (M8) to
**20.7 s/tok (1.30×)**, argmax bit-exact with M8. M8.7.1.b/c (dedicated
copy/compute streams) deferred for VRAM-budget reasons, scaffolding left in
place. See [HANDOFF M8.7](./HANDOFF_APX_V20_M8.7.md).

### M9 — INT8 W8A16 quantisation (opt-in / experimental)

The full INT8 path landed cleanly (quantizer → `TensorStorage::CpuInt8` → CUDA
per-group dequant → loader → tier-planner cost model); M9.0 microbench measured
~2× over M8.4c on the four dominant 13B shapes. End-to-end on the dev box: **M9
default 18.8 s/tok (−9 % vs M8.7); with `ATENIA_RAM_HEADROOM_OVERRIDE_GIB=8`,
17.7 s/tok (−14 %)**. The numerical contract was **not** met — simple absmax
INT8 (per-channel, per-group g∈{32,128}) misses ADR-004's `< 0.5` gate on 4/4
fixture models (12-run experiment matrix). M9 ships **opt-in for the drift
reason, not the performance reason**; default unset is bit-identical to M8.6.
ADR-004-strict INT8 deferred to track β (LLM.int8 outlier decomposition). See
[HANDOFF M9](./HANDOFF_APX_V20_M9.md).

### M10 — Real velocity (two-mode execution + per-checkpoint certificates)

M10's structural deliverable: **Atenia ships two execution modes plus a
per-tensor dispatch infrastructure driven by per-checkpoint numeric
certificates that no other inference engine publishes.**

- **M10.0** — baseline characterisation across the four velocity brackets.
- **M10.1** — CPU GEMV specialisation: attempted, reverted (hypothesis wrong,
  both paths memory-bound on strided B access; analysis in `bench_logs/`).
- **M10.2.0** — TF32 Tensor Cores in certified mode
  (`CUBLAS_COMPUTE_32F_FAST_TF32`). ADR-004 strict preserved 4/4.
- **M10.2.1** — BF16-TC native **fast mode** (`ATENIA_FAST_MODE=1`,
  `cublasGemmEx(BF16, BF16, F32)`).
  [ADR-005](./decisions/ADR-005-fast-mode-bf16-tc-envelope.md) documents the
  per-checkpoint drift envelope.
- **M10.3** — numeric-contract schema v1.0.0 + per-checkpoint manifests in
  [numcert/](./numcert/) + [CERTIFICATION.md](./CERTIFICATION.md).
- **M10.3.1.0** — manifest-driven mode selection
  (`ATENIA_FAST_MODE > manifest.recommended_mode > certified`).
- **M10.3.1.1** — per-tensor dispatch: schema v2.0.0 with
  `per_tensor_policy`, `WeightStore::apply_per_tensor_policy` stamps each
  VRAM-resident tensor with a precision byte; dispatcher routes per-matmul.
  Confirmed mixed dispatch within a single model (TinyLlama: 66 fast FFN
  tensors + 88 certified attention tensors).

Velocity matrix (dev box, post-M10.3.1.1): TinyLlama 0.37 s/tok ✅,
SmolLM2 1.52 s/tok ✅, Mistral 7B ~8 s/tok (fast; hardware-limited on 8 GB
VRAM), Llama 2 13B 15.12 s/tok (fast). The residual 7B/13B gap is hardware-
bound on 8 GB VRAM and closes through track β, not more kernel work.

### M11 — Top-10 certification + GGUF support

**M11.D.5** completed the GGUF production unlock: Q4_K_M decoder, CLI
auto-detection (`.gguf` extension), and a **functional certification schema
v2.0.0** (smoke-based, 5-token greedy, documented drift). ADR-004 strict does
not apply to GGUF by construction — Q4_K_M is aggressive 4-bit quantisation
with intrinsic precision loss. Certified models: TinyLlama Q4_K_M
(lm_head drift ~10.19) / Q8_0 (~2.28), Llama 3.2 1B Q4_K_M (norm drift 0.0,
F16), SmolLM2 1.7B Q4_K_M (q_proj drift ~0.287). Findings: Q8_0 ~4.5× more
stable than Q4_K_M; F16 tensors decode bit-exact; all models generate coherent
text. `cargo test --lib` 320/320 green. Tag `v0.11.0-m11-d5`.
See [HANDOFF M11.D.5](./HANDOFF_APX_V20_M11_D_5.md).

---

## Adapter Phases 11–12 — Multi-family adapter layer

After M11, the family-specific logic was migrated out of the execution core
into a contained adapter layer (`src/model_adapters/`: `common.rs`,
`llama_family.rs`, `phi3.rs`, `gemma2.rs`, `mod.rs`). Adding a new family
becomes a contained change; the execution core stays family-agnostic.

**Phase 11 — adapter layer foundation**

- Internal model adapter layer, phase 1.
- Adapter metadata and residency hints; adapters split by model family.
- Residency hints wired into tier planning; residency policy logged;
  adapter LM-head residency override.
- Adapter registry contracts; resolution diagnostics backed by registry
  contracts; pipeline adapter metadata centralised; legacy Phi-3 store-builder
  dispatcher removed.
- **Phase 11.1** — `demo::build_and_load_llama` routed through the adapter
  layer.

**Phase 12 — config policy migration**

- **12.1** — `ConfigPolicy` trait added to the adapter layer.
- **12.2** — `get_tie_word_embeddings` migrated to consult the adapter
  registry.
- **12.3** — `effective_attention_bias` migrated to consult the adapter
  registry.
- **12.4** — Phi-3 LongRope parsing moved to the adapter layer; contract
  revised to **fail-fast on LongRope under a non-Phi3 `model_type`**.

**Phase 13 — family validation → the adapter**

- The family-semantic `hidden_size % num_attention_heads` check (V2) moved
  out of `LlamaConfig::validate()` into `ConfigPolicy::validate_config`
  (default `Ok(())`), backed by a shared helper and wired into
  Llama / Qwen2 / Mistral; Phi-3 / Gemma 2 keep the default (they ship an
  explicit `head_dim` the derived-head_dim assumption would wrongly reject).
- Structural fix bundled: `validate()` now checks `effective_head_dim()`
  (the value the kernel uses) instead of the derived `head_dim()`.
- `validate()` stays public and family-agnostic; the full contract is
  `parse → validate() → adapter.validate_config()`. The pre-existing
  M11.B `RopeScaling` non-exhaustive-match debt in `tinyllama_config_test`
  was fixed in an isolated commit (`4a09198`).

**Phase 14 — llama3 `rope_scaling` → the adapter**

- The Llama 3 piecewise inverse-frequency parser moved out of
  `config.rs::get_rope_scaling` into a shared
  `common::parse_llama3_rope_scaling`, wired into the Llama-family
  adapters' `parse_rope_scaling` (parallel to Phi-3 LongRope at 12.4).
- `get_rope_scaling` is now orchestration only: delegate to the adapter,
  then a fail-fast guard. A recognised discriminator declared under the
  wrong family now errors (`"llama3"` under non-Llama, symmetric with the
  longrope-under-non-Phi3 guard) instead of silently downgrading.

**Phase 15 — GGUF family config semantics → the adapter**

- `gguf_config.rs` is now a pure format parser: the `if arch == "phi3"`
  and `if arch == "gemma2"` blocks are gone. Phi-3 LongRope is parsed by
  translating the GGUF rope-scaling metadata into the same HF-shaped
  `serde_json::Value` the adapter already consumes, routed through the
  existing `Phi3Adapter::parse_rope_scaling` — no HF/GGUF semantic
  duplication, `GgufReader` never enters the adapter.
- New `ConfigPolicy::apply_config_defaults` (default no-op) owns family
  default *values* for both formats: `Gemma2Adapter` injects the 50/30
  softcaps, `query_pre_attn_scalar`, and explicit `head_dim`;
  `Phi3Adapter` makes `head_dim` explicit. Called from both
  `from_json_str` (HF) and `llama_config_from_gguf` (GGUF) before
  validate; HF Gemma 2 configs missing the caps now get the family
  defaults (accepted parity change; explicit values always win). The
  GGUF arch → architecture/model_type mapping stays as a documented
  format-identity bridge.

**Boundary status.** The **config** boundary is closed (Phases 13–15):
`LlamaConfig` and `gguf_config.rs` carry no family semantics. The
symmetric **weight-mapping** boundary (the `gguf_to_hf_naming.rs` /
`build_gguf_name_map` `if arch ==` branch) was the **Phase 16**
candidate, since closed (see the Phase 16 section below). A 2-tier CI
(blocking `cuda-toolkit` + visible non-blocking `cpu-only`, the latter
guarding the vendor-agnostic invariant) plus this doc refresh is the
consolidation pass taken before M12. The M12 series that followed is
below. See [STATUS.md](./STATUS.md).

---

## M12 — Operability & diagnostics hardening

A tightly-scoped, sub-phased series that made the engine surface *why* it
fails instead of swallowing the cause. Each sub-phase followed the same
contract: audit → microplan → gate → implement → local tests → single
commit → CI. No control-flow, numeric, tiering, or placement changes; the
success path is byte-stable throughout.

**M12.1 — CUDA root-cause / VRAM-probe diagnostics**

The CUDA upload helpers and the VRAM probe used to collapse real driver
errors into a generic `None` / `0`. Split into `_detailed` sibling
functions returning typed errors (`Bf16UploadError` reused,
`VramProbeError` added with a pure `nvidia-smi` parser); the legacy
signatures stay as thin `.ok()` / `.unwrap_or(0)` wrappers so the
value-on-failure (and therefore the tier plan) is unchanged. The root
cause is now propagated to the caller and the operator.

**M12.2 — `atenia run` load panics → clean errors + exit 2**

The `atenia run` load path `.expect()`-panicked on a bad model. Introduced
`DemoLoadError` and `build_and_load_llama_checked` returning a `Result`;
the three CLI callers stop the heartbeat, print `error: {e}`, and return
exit code 2. The legacy `build_and_load_llama` is preserved as a
panic-wrapper so the `m4_7_6_e` momento-guau path is untouched.

**M12.3 — env/hardware diagnostics + tier-plan summary**

One consolidated, read-and-echo env/hardware diagnostics block
(`src/diag.rs`, a pure `render_env_diagnostics` + 6 tests, suppressed
under `apx_is_silent()`) printed once on the `generate` / `run` paths, and
a shared `gpu::tier_plan::log_tier_plan_summary` so the run path emits the
same tier-plan summary the pipeline already did. `[APX] Using mode` was
un-gated from `apx_debug_enabled()` to `!apx_is_silent()`.

**M12.4 — visible fallbacks + actionable CLI errors**

Silent degrades became visible: H1 — a once-per-run `[ATENIA][warn]` when
`cuda_matmul` falls back to CPU (latch, no decode-loop spam); H2 — the M6
legacy-residency summary now appends a failed-layer count; H3 —
`try_all_paths` accumulates per-attempt reasons and flushes them on the
terminal CPU fallback (success path unchanged); H5 — `impl Display for
LoaderError` (root cause of a raw `{:?}` leak), `PipelineError::Loader`
switched to it, plus a remediation hint on the `atenia generate` load
failure.

**M12.5 — `cuInit` checked + JSON render failures exit ≠ 0**

The `cuInit` CUresult was discarded at three loader sites; a pure
`map_cu_init_rc` helper now turns a non-zero code into a specific
`LoadError` (which also flows into the M12.4 `[COMPAT][warn]` reasons).
`cli_run`'s `render` / `render_mode_b` return `bool`, and the three Mode
A/B/C callers map a JSON serialise failure to a non-zero exit instead of
exiting `0` with empty stdout.

**Series close.** `cargo test --lib` 369/369, `tinyllama_config_test`
15/0/3, blocking `cuda-toolkit` CI green per sub-phase. The
vendor-agnostic CPU-only link drift that M12 carried has since been
closed by the CPU-1 → CPU-5 series (below); the Phase 16
weight-mapping boundary that [STATUS.md](./STATUS.md) still listed has
since been closed by the Phase 16 series (below). No tracked debt
remains. This marks the transition from "engine that runs" to "engine
that explains itself".

---

## CPU-only vendor-agnostic build (CPU-1 → CPU-5)

Closes the long-standing build-system debt where a CUDA-less
`cargo build --lib` failed at **link** (*"could not find native static
library `batch_matmul`"*): `build.rs` early-returned on a CUDA-less
host without emitting link directives, yet the Rust FFI declared the
CUDA kernel static libraries unconditionally. The ROADMAP /
ADR-003 ("GPU as inference baseline") vendor-agnostic invariant ("the
core never assumes NVIDIA-specific behaviour") is now **enforced in
CI**, not merely asserted.

- **CPU-1** — `build.rs` declares `cargo::rustc-check-cfg=cfg(atenia_cuda)`
  on every path and *sets* `cargo::rustc-cfg=atenia_cuda` only after the
  CUDA kernels actually compiled and linked. Auto-detected; zero flags
  on a CUDA box. No source read it yet — pure foundation.
- **CPU-2** — every CUDA `extern` / `#[link]` block across `src/cuda/*`
  and `apx4_12/gpu_memory_pool.rs` gated `#[cfg(atenia_cuda)]`, each with
  an identical-signature `#[cfg(not(atenia_cuda))]` stub. Landed in three
  sub-commits: **C2a** (gpu_memory_pool, mod, linear, batch_matmul,
  fused_linear_silu), **C2b** (bf16_to_f32, int8_to_bf16, pool_helpers),
  **C2c** (matmul.rs + `cuda_available()` hardened to `false` without
  the backend + `vec_add_gpu` exact CPU sum). CUDA-less wrappers return
  `None` / `Err` / the exact CPU result; VRAM-only paths are
  `unreachable!`. The CUDA build stays byte-identical (lib 369/369,
  `tinyllama_config_test` 15/0/3) at every sub-commit.
- **CPU-5** — once the first `cpu-only` CI run went green, the job was
  promoted from non-blocking (`continue-on-error`) to **blocking**, so a
  future vendor-agnostic regression fails the run. CI is now a two-blocking-
  job matrix: `cuda-toolkit` + `cpu-only / no-CUDA`.

This is a build-system / FFI boundary fix only — it does **not** add a
non-NVIDIA compute backend (Intel iGPU / AMD ROCm / Apple Metal remain
roadmap v22–v24). The CUDA success path is unchanged throughout.

---

## Phase 16 — weight-mapping boundary closed

Closes the last `core depends on arch` weight-mapping break. The
*config* boundary closed in Phases 13–15; the symmetric GGUF→HF
tensor-name mapping still ran through an arch-string free function
(`gguf_to_hf_name(name, arch)` with `if arch == "phi3" / "gemma2"`)
called by `pipeline::build_gguf_name_map`.

- **16.1** (`85010b6`) — pure refactor: `gguf_to_hf_name` decomposed
  into `split_blk` + the arch-agnostic `gguf_to_hf_name_common` + the
  family extras `phi3_gguf_extra` / `gemma2_gguf_extra`; the composing
  fn kept transitionally. Behaviour-identical.
- **16.2** (`3340cda`) — new adapter-owned `GgufNameMapper` trait
  (default = common; Phi-3 / Gemma 2 override), added to the
  `AteniaModelAdapter` supertrait. Behaviour-inert until wired.
- **16.3** (`b19f726`) — `build_gguf_name_map` takes the resolved
  adapter and calls `adapter.gguf_to_hf_name(...)`; the arch-branching
  free fn is removed; `arch` survives only for the missing-metadata
  guard and the diagnostic message.

Scope discipline: `model_type_for_arch` (the documented
arch→model_type label bridge) and `gguf_weight_loading.rs` (family
transform fns already invoked only via the adapters) were
intentionally left untouched — they are not `core depends on arch`
breaks. CUDA build byte-identical (lib 376/376,
`tinyllama_config_test` 15/0/3, zero numeric / behaviour change);
both blocking CI jobs green. The execution core is now fully
family-agnostic for config **and** weight mapping.

---

## Beyond v20

The roadmap horizons (v21/M12 production hardening → v22 Intel iGPU → v23 AMD
ROCm → v24 Apple Metal → v25 distributed → training at v25+) are tracked in
[ROADMAP.md](../ROADMAP.md). None are shipped; the codebase is structured to
make them possible, not to claim them.
