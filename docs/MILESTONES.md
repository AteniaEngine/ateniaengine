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

## Phi-3.5 GGUF end-to-end — consolidation validation

The post-Phase-16 consolidation validation battery surfaced a real
Phi-3.5 GGUF load failure. Closing it served as the first
end-to-end exercise of the GGUF path on a fused-tensor LongRope
family: `Phi-3.5-mini-instruct` Q4_K_M now loads and generates
coherent text. Four isolated, regression-tested fixes landed
(`a712f28`, `e67f627`, `b423f56`, `345482d`): Phi-3 LongRope read
from the GGUF `rope_factors_{short,long}` tensors with the factor
tensors skipped by the weight-name map; a Q5_K block decoder
mirroring `dequantize_row_q5_K`; the Phi-3 GGUF transform table
delegated to the single safetensors source of truth; and the fused
gate/up name mapping (`ffn_up` → `gate_up_proj`, Phi-3 overrides
ordered before the common Llama-layout table) with the non-weight
skips accepted by both GGUF completeness gates. GGUF k-quant
coverage is now Q4_K / Q5_K / Q6_K; the GGUF path is validated for
LongRope and the fused QKV / gate_up tensor layout. CUDA build
green (lib 382/382, `tinyllama_config_test` 15/0/3); both blocking
CI jobs green. No tracked debt remains.

---

## Adapter Toolkit + Gemma 2 GGUF correctness

The Phi-3.5 GGUF bring-up motivated ADR-006 (the Adapter Toolkit):
the per-family GGUF->HF name maps and load-transform tables, until
then imperative `if name.contains(...)` ladders duplicated across
five files, became declarative Rust data. Landed in order: **AT-0**
(`20bc651`, ADR-006); **AT-2** (`d9634a6`, a conformance harness
that froze current adapter behaviour as an executable oracle —
written before the refactor); **AT-1a/b/c** (`536a506`, `914ae2a`,
`60311ea`) — `FamilyTensorSpec` data plus golden A/B tests, then
the GGUF->HF name maps and the load transforms rewired onto it,
behaviour-preserving (lib + conformance + TinyLlama end-to-end
green on both the HF and GGUF paths). AT-2 flagged GAP-1: the
Gemma 2 GGUF transform table was a never-validated pre-`.rev()`
table; AT-1 froze it verbatim with a `KNOWN BUG` marker and
deferred the fix.

The dedicated **Gemma 2 GGUF correctness** phase then closed it
against a real checkpoint (bartowski/gemma-2-2b-it Q4_K_M), in six
isolated regression-tested layers: `post_ffw_norm.weight`
unmapped (GAP-N1, `907f1ca`); wrong GGUF config keys for head_dim
/ softcaps — `attention.key_length` and `attn_logit_softcapping`,
not `head_size` / `logit_softcap` (GAP-C1/C2, `455983c`); the
4-norm `ffn_norm` -> `pre_feedforward_layernorm` mapping with
extra-first adapter composition (GAP-N2, `a33d44f`); and the root
cause — llama.cpp pre-folds the RMSNorm `1+gamma` into the Gemma 2
GGUF weights (measured exactly +1.0 element-wise vs the HF
safetensors across every norm class), so the corrected Gemma 2
GGUF transform table is the HF table minus the norm `+1` fold
(GAP-T1, `36bbfe0`). The buggy `GEMMA2_GGUF_GAP1` table and the
GAP-1-only `TileKvDim1` recipe were removed; `gguf_gap1_transforms`
was renamed `gguf_transforms`. No `RopeUnpermute` is required
(Phi-3 bracket). Gemma 2 GGUF now generates text identical to the
HF reference ("The capital of France is Paris."); TinyLlama and
Phi-3.5 GGUF regressions unaffected. Lib 406/0/0,
`tinyllama_config_test` 15/0/3, build clean; the AT-2 snapshot and
AT-1a golden that pinned the buggy table were updated as an
intentional, checkpoint-validated change. No tracked debt remains.

**v1 closure** (`8a18cbf`, `30314b1`, `<docs>`). AT-3 landed:
the two hand-copied GGUF load-completeness gates in `pipeline.rs`
collapsed into a single `is_unexpected_gguf_skip` helper (AT-3a);
`FamilyTensorSpec` gained a `required_gguf_dtypes` field per
family with a conformance test that asserts every declared dtype
is decodable — converting a future `UnsupportedDType` (e.g. a
new family declaring Q2_K) from a runtime failure into a
test-time failure (AT-3b). With AT-3 the ADR-006 microplan is
fully delivered, validated end-to-end against two real families
(Phi-3.5 + Gemma 2 GGUF, both identical to HF) without using the
imperative escape hatch. AT-4 (YAML / JSON templates, scaffolding
generator, public Adapter SDK) remains explicitly deferred — no
serialized contract is frozen and no automatic / magic model
support is promised; a new family still requires a graph
builder, numeric validation and explicit review. Lib 409/0/0;
`tinyllama_config_test` 15/0/3; CI dual-blocking green; no
tracked debt remains.

**Post-v1 local validation battery.** A load-and-generate sweep
over the 18 checkpoints under `models/` on the dev box (RTX 4070
Laptop, 8 GB VRAM, default certified mode, 12-token greedy
continuation) returned 17 PASS / 1 WARN / 0 FAIL. The mandated
regressions (TinyLlama Q4_K_M GGUF, Phi-3.5-mini Q4_K_M GGUF,
gemma-2-2b-it Q4_K_M GGUF) all generated text identical to their
HF safetensors references. The single WARN was `gpt2-safetensors`,
a directory missing `config.json` (incomplete checkpoint, not a
load-path defect; no GPT-2 adapter is registered). One drift in
the STATUS known-limitations was detected: the previously
documented "Mistral 7B and Falcon 3 7B fail in
certified/manifest mode with `BF16->VRAM slow-path upload
failed`" no longer reproduces on the local
`mistral-7b-v0.3` and `falcon3-7b-instruct` safetensors — both
load and generate coherent text. The limitation is downgraded in
STATUS to "previously observed, currently dormant" pending
broader coverage; no new F64 numcert was added for these two
checkpoints. No regression in any previously-validated family.

---

## Phase Q — Qwen3 family support

The Adapter Toolkit v1 closure validated against five families
(Llama, Qwen2, Mistral, Phi-3, Gemma 2). Phase Q adds **Qwen3**
as the first new family on top of the toolkit, exercising the
"add a family" path against a real safetensors checkpoint
(`Qwen/Qwen3-0.6B`) on the dev box.

Topological deltas vs Llama (confirmed by reading the safetensors
header directly): **per-head QK-Norm** RMSNorm applied to Q and K
after the reshape-to-heads and before RoPE — γ shape `[head_dim]`,
broadcast across heads — plus **no QKV biases** and an explicit
`head_dim` (128 in the 0.6B checkpoint, not equal to
`hidden_size / num_attention_heads = 64`), so q/k/v/o projection
shapes use `n_heads * head_dim` instead of `hidden`. No new AMG
NodeTypes are introduced; the per-head RmsNorm is expressible
with the existing `RmsNorm` + `BroadcastMul` over the
`[batch, seq, n_heads, head_dim]` layout. The
`1 / sqrt(head_dim)` attention scale is moved from `k_proj` into
the K-Norm γ via the load transform (a pre-normalize scale would
be stripped by RMSNorm; a post-normalize γ scale survives),
preserving the toolkit's "scale lives in the weight" convention.

Landed in three layered commits (`2d76076`, `5809253`, `40e0633`):
`ModelFamily::Qwen3` + `Qwen3Adapter` + registry entry; declarative
`QWEN3_SPEC` with `ReshapeHeadDim4D` recipe and the per-head γ rules
in `tensor_spec.rs` plus the conformance routing; the self-
contained `nn/llama/qwen3.rs` graph builder + weight mapper
mirroring the Phi-3 / Gemma 2 self-contained pattern. Qwen3's
LM head is always registered as a separate parameter (independent
of `tie_word_embeddings`) because Qwen3 safetensors physically
ships `lm_head.weight`; the math matches the tied case when
checkpoint weights coincide. The imperative escape hatch from
ADR-006 Decision 5 was **not** used — the spec expressed the
QK-Norm transform purely as declarative data.

Validated end-to-end against `Qwen3-0.6B`: load 6.4 s on the dev
box, 12-token greedy generation produces coherent text
(`<think>\nOkay, the user is asking about the capital of France.
Let me think. I know that France's capital is Paris...`). Lib
409/0/0, `tinyllama_config_test` 15/0/3, build clean (0 warnings).
Mandatory regressions all green at the same HEAD: TinyLlama
Q4_K_M GGUF, Phi-3.5-mini Q4_K_M GGUF, gemma-2-2b-it Q4_K_M GGUF,
Qwen2.5-Coder-1.5B-Instruct safetensors. No regression in any
previously-validated family.

What Phase Q deliberately does NOT include: Qwen3 GGUF (out of
scope; `required_gguf_dtypes` empty, no certified checkpoint), and
Qwen3 numcert / F64 fixture (no manifest added for the 0.6B
checkpoint). Larger Qwen3 variants share the same topology and
should resolve through the same adapter without further code,
subject to hardware fit; none validated end-to-end yet.

---

## Model family validation

The per-family mastery validation sections that follow are
consolidated — per-checkpoint tables, formats, quants, real fixes,
out-of-scope notes — in `docs/MODEL_FAMILY_VALIDATION.md`. That
document is the single-page reference; the sections below keep the
full per-phase narrative. Both describe *functional / family
validation*, which is distinct from ADR-004 numeric certification.

## Llama-family mastery validation

Pre-Adapter-Toolkit-v2 audit of the Llama family: confirm the
existing `LlamaFamilyAdapter` (closed at Phase 16) handles every
checkpoint in the target Llama scope without regressions and
without family-specific code.

**Phase 2 (read-only audit) findings.** Config parsing
(`rope_theta` legacy default 10000 for Llama 1/2; `eos_token_id`
array → first element for Llama 3.x; `rope_scaling.llama3`
adapter-owned in `common::parse_llama3_rope_scaling`); adapter
resolution (`LlamaForCausalLM` + `model_type=llama` → both
DeepSeek-R1-Distill-Llama and plain Llama route here); tokenizer
(`tokenizer_config.json::eos_token` resolves `<|eot_id|>` → 128009
for Llama 3.x instruct, so the single-EOS `GenerationConfig`
contract holds correctly without multi-EOS support); safetensors
+ GGUF weight mappers (`llama_weight_mapper` /
`llama_gguf_weight_mapper` with `LlamaRopeUnpermuteRows` for q/k);
GGUF quant decoders (F16, Q8_0, Q4_K, Q5_K, Q6_K all decoded).
**No gap surfaced.**

**Phase 5 smokes (`target/validation/llama_mastery/`).** Greedy,
certified mode, prompt `"What is the capital of France?"`
(`"What is recursion in programming?"` for the DeepSeek distill so
its chain-of-thought is exercised), max-tokens 80 (120 for
DeepSeek). **9 / 9 PASS:**

| # | Checkpoint                              | Format       | Output                                | Load   | tok/s  |
|---|-----------------------------------------|--------------|---------------------------------------|--------|--------|
| A | TinyLlama 1.1B                          | safetensors  | "The capital of France is Paris."     |  70.9s | 2.33   |
| B | Llama 3.2 1B Instruct                   | safetensors  | "The capital of France is Paris."     |  62.4s | 2.19   |
| C | Llama 3.2 1B Instruct                   | Q4_K_M GGUF  | "The capital of France is Paris."     |  25.4s | 2.52   |
| D | Llama 3.2 3B Instruct                   | safetensors  | "The capital of France is Paris."     | 119.3s | 0.28   |
| E | Llama 3.1 8B Instruct                   | safetensors  | "The capital of France is Paris."     | 507.9s | 0.07   |
| F | Llama 3.1 8B Instruct                   | Q4_K_M GGUF  | "The capital of France is Paris."     | 186.5s | 0.07   |
| H | Llama 3.1 8B Instruct                   | Q5_K_M GGUF  | "The capital of France is Paris."     | 215.7s | 0.07   |
| I | Llama 3.1 8B Instruct                   | Q6_K GGUF    | "The capital of France is Paris."     |  83.5s | 0.10   |
| J | DeepSeek-R1-Distill-Llama-8B            | Q4_K_M GGUF  | coherent CoT reasoning (120 tok)      | 158.7s | 0.10   |

A–C halt on EOS; D–I halt on EOS (Llama 3.x `<|eot_id|>` = 128009);
J halts on max-tokens (CoT model deliberately keeps reasoning).
The 8B-class throughput (0.07–0.10 tok/s) is dominated by
tier-aware spill on an 8 GB VRAM + 32 GB RAM box, not a kernel
or adapter regression. `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **409 / 409 PASS**;
`cargo test --test tinyllama_config_test` → **15 PASS / 3 ignored**.

**Non-claims.** Llama 2 7B Chat was **not** validated end-to-end
in this battery: gated on HuggingFace, no token available, not
downloaded. Llama 2 13B Chat coverage stays via
`m5_dc_llama2_13b_coherence_test` (locked integration test,
ship since M5.d.c). DeepSeek-R1 distill is validated as a
Llama-family derivative; the `<think>...</think>` special-token
behaviour was not exhaustively checked beyond a single coherent
120-token sample. No numcert manifest was added for any of the
new 8B GGUF checkpoints — the smoke is functional, not the
ADR-004 strict gate.

**Conclusion.** The Llama family is dominated in the M11–16
sense for the validated scope. No production code touched in
this phase; the Adapter Toolkit v1 (`LlamaFamilyAdapter` +
`llama_weight_mapper` + `llama_gguf_weight_mapper` +
`common::parse_llama3_rope_scaling`) already covered every
checkpoint, including the DeepSeek-R1 distill derivative,
without family-specific branches.

---

## Qwen-family mastery validation

Pre-Adapter-Toolkit-v2 audit of the Qwen family (Qwen2 + Qwen3):
confirm the adapters handle every checkpoint in the target Qwen
scope. Unlike the Llama battery, this one surfaced **real gaps**
and required adapter/config fixes — all generalizable, none
touching the execution core.

**Phase 2 audit + Phase 6 smokes** ran together: the safetensors
smokes passed unchanged, the GGUF smokes failed and drove the
fixes. Final result, prompt `"What is the capital of France?"`
(reasoning prompt for the distill), greedy, certified mode —
**11 / 11 PASS**:

| Checkpoint                          | Format        | Output                            |
|-------------------------------------|---------------|-----------------------------------|
| Qwen2.5 1.5B Instruct               | safetensors   | "...is Paris." (EOS)              |
| Qwen2.5 3B Instruct                 | safetensors   | "...is Paris." (EOS)              |
| Qwen2.5 7B Instruct                 | safetensors   | "...is Paris." (EOS)              |
| Qwen2.5 7B Instruct                 | Q4_K_M GGUF   | "...is Paris." (EOS)              |
| Qwen2.5 7B Instruct                 | Q5_K_M GGUF   | "...is Paris." (EOS)              |
| Qwen2.5 7B Instruct                 | Q6_K GGUF     | "...is Paris." (EOS)              |
| Qwen2.5-Coder 1.5B Instruct         | safetensors   | "...is Paris." (EOS)              |
| Qwen3 0.6B                          | safetensors   | coherent `<think>` CoT            |
| Qwen3 4B                            | safetensors   | coherent `<think>` CoT            |
| Qwen3 8B                            | Q4_K_M GGUF   | coherent `<think>` CoT            |
| DeepSeek-R1-Distill-Qwen-7B         | Q4_K_M GGUF   | coherent CoT reasoning            |

**Gaps found and fixed (adapter / config / transform — no core):**

1. **Qwen3 LM-head tie (adapter).** `build_qwen3` /
   `build_qwen3_with_store` always registered a separate
   `lm_head.weight`. The small variants (0.6B / 1.7B) ship a
   physical (redundant) `lm_head.weight` so this happened to
   work; the larger variants (4B / 8B / 14B / 32B) are genuinely
   tied and ship none, so the loader left the graph parameter
   zero-initialised → uniform logits → the model emitted a
   constant degenerate token (vocab_size−1) every step. Root
   cause confirmed by inspecting the safetensors headers
   (Qwen3-4B has no `lm_head.weight`). Fixed to honour
   `config.tie_word_embeddings` exactly like `build_llama`
   (tied → reuse `embed_tokens` transposed). Re-validated:
   Qwen3-4B coherent, Qwen3-0.6B still coherent (310 params,
   was 311).

2. **Qwen2 / Qwen3 GGUF architecture (config).**
   `gguf_config.rs` rejected `general.architecture = "qwen2"` /
   `"qwen3"`. Added both to `architecture_from_gguf`,
   `arch_prefix` (metadata-key prefix), and `model_type_for_arch`.

3. **Qwen2 GGUF QKV biases (config + name mapping).** Qwen2
   carries q/k/v biases. `COMMON_NAME_TABLE` gained the
   `attn_{q,k,v}.bias` → `self_attn.{q,k,v}_proj.bias` suffixes
   (additive — no other family's GGUF has them). The hard-coded
   `attention_bias = Some(false)` in `llama_config_from_gguf`
   became `None`, so `effective_attention_bias()` resolves the
   family default (Qwen2 → `true`); byte-identical for every
   previously-supported GGUF family.

4. **Qwen3 GGUF QK-Norm + weight mapper (adapter).**
   `QWEN3_SPEC.name_extra` maps `blk.N.attn_{q,k}_norm.weight`;
   the Qwen3 adapter's `GgufNameMapper` composes it with the
   common table, and `map_gguf_weights` now uses the
   QK-Norm-aware HF transform table (`qwen3_weight_mapper`)
   instead of the Llama GGUF mapper. llama.cpp does **not**
   row-permute Qwen2/Qwen3 q/k (the permute is Llama-arch
   specific), and the residency loader's `.rev()` leaves tensors
   in HF orientation, so both Qwen GGUF paths reuse their HF
   transform table with **no** `LlamaRopeUnpermuteRows` — the
   same relationship Phi-3 / Gemma 2 GGUF already have. The
   coherent GGUF smoke output is the empirical confirmation that
   the no-permute decision is correct (a permute error scrambles
   attention into incoherent output).

**Files changed.** `src/nn/llama/qwen3.rs` (lm_head tie),
`src/v17/loader/gguf_config.rs` (arch whitelist + attention_bias),
`src/model_adapters/tensor_spec.rs` (COMMON QKV bias suffixes +
QWEN3_SPEC QK-Norm extras + required_gguf_dtypes),
`src/v17/loader/gguf_to_hf_naming.rs` (`qwen3_gguf_extra`),
`src/model_adapters/llama_family.rs` (Qwen2 GGUF mapper),
`src/model_adapters/qwen3.rs` (Qwen3 GGUF mapper + name mapper).

**Tests.** `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **409 / 409 PASS** (run
both after the lm_head fix and after the GGUF fixes);
`cargo test --test tinyllama_config_test` → 15 PASS / 3 ignored.

**Non-claims.** The 7–8B GGUF checkpoints have no numcert
manifest — the smoke is functional, not the ADR-004 strict gate.
Qwen3 variants 14B / 32B were not downloaded (hardware fit);
they share the topology and should resolve through the same
adapter. Throughput on the 7–8B class (0.07–0.12 tok/s) is
bounded by tier-aware spill on the 8 GB-VRAM dev box.

**Conclusion.** Qwen is dominated for the validated scope:
Qwen2 safetensors + GGUF (Q4_K_M / Q5_K_M / Q6_K), Qwen3
safetensors + GGUF, and the DeepSeek-R1 distill-Qwen derivative
all load and generate coherently, with no per-model hacks and
no core changes.

---

## Gemma-family mastery validation

Pre-Adapter-Toolkit-v2 audit of the Gemma family (Gemma 2 +
Gemma 3 text). Gemma 2 was already supported; **Gemma 3 (text)
is a new family added in this phase**. Text-only focus —
multimodal Gemma 3 is explicitly out of scope.

**Phase 6 smokes** — prompt `"What is the capital of France?"`,
greedy, certified mode — **8 / 8 in-scope PASS**:

| Checkpoint                  | Format       | Output                         |
|-----------------------------|--------------|--------------------------------|
| Gemma 2 2B Instruct         | safetensors  | "...is **Paris**. 🇫🇷" (EOS)    |
| Gemma 2 9B Instruct         | safetensors  | "...is **Paris**. 🇫🇷" (EOS)    |
| Gemma 2 9B Instruct         | Q4_K_M GGUF  | "...is **Paris**." (EOS)       |
| Gemma 2 9B Instruct         | Q5_K_M GGUF  | "...is **Paris**. 🇫🇷" (EOS)    |
| Gemma 2 9B Instruct         | Q6_K GGUF    | "...is **Paris**." (EOS)       |
| Gemma 3 1B Instruct         | safetensors  | "...is **Paris**." + note (EOS)|
| Gemma 3 1B Instruct         | Q4_K_M GGUF  | "...is **Paris**." (EOS)       |
| Gemma 3 4B Instruct         | Q4_K_M GGUF  | coherent (text path)           |

**Gemma 3 (text) — new family.** `Gemma3ForCausalLM` /
`model_type = gemma3_text`. It is Gemma 2's topology — dual-norm
per block, `(1+γ)` RMSNorm, GeGLU FFN, `× sqrt(hidden)` embedding
scale, tied LM head — with three deltas:

1. **Per-head QK-Norm** — RMSNorm γ `[head_dim]` on q and k after
   reshape-to-heads, before RoPE (as Qwen3). The
   `1/√query_pre_attn_scalar` attention scale is folded into the
   k_norm γ (post-`+1`).
2. **No soft-cap** — Gemma 3 dropped Gemma 2's attention/final
   logit soft-caps; the builder's conditional `Some/None` already
   skips the `SoftCap` node.
3. **Dual RoPE base frequency** — local (sliding-window) layers
   use `rope_local_base_freq` (10 000), global layers use
   `rope_theta` (1 000 000), selected per layer by
   `sliding_window_pattern` (every `pattern`-th layer is global).

Implemented as a new `Gemma3Adapter` + `build_gemma3` /
`build_gemma3_with_store` builder + `GEMMA3_SPEC` (HF and GGUF
transform tables) + the GGUF `gemma3` architecture route. No new
AMG ops — the dual RoPE is just a per-layer `theta` argument to
the existing `RoPE` node. Sliding-window attention is deferred
(full causal — equivalent below the window, as for Gemma 2); the
dual RoPE base **is** applied per layer.

**Gaps found and fixed (adapter / config / tokenizer / decoder —
no core change):**

1. **Gemma 3 unsupported (adapter + config + graph).** Added the
   two config fields, the adapter, the builder, the spec, and the
   GGUF architecture.

2. **Multi-EOS (config + tokenizer).** Generation collapsed a
   model's `eos_token_id` to the first array element. A Gemma
   instruct turn ends with `<end_of_turn>` (id 106 in Gemma 3's
   262 k vocab, a *different* id in Gemma 2's 256 k vocab), not
   `<eos>`. Collapsing to the first element left Gemma 3 running
   past its natural stop into off-distribution garbage —
   confirmed by a token-id diagnostic (the model correctly
   emitted `<end_of_turn>` right after the answer; the generator
   did not halt). Fix: `LlamaConfig` keeps the full
   `eos_token_ids` set, `GenerationConfig.eos_token_ids` is a
   `Vec`, the generator halts on any member, and the pipeline
   additionally resolves the standard chat turn-terminators
   (`<end_of_turn>` / `<|eot_id|>` / `<|im_end|>`) by **name**
   from the vocabulary — the GGUF metadata carries only a scalar
   `eos_token_id`, so the array is unavailable there. Generalises
   to every chat family; Llama 3.x and Qwen were already correct
   (their `eos_token` *is* the turn-terminator) and the name
   lookup is a dedup no-op for them.

3. **Q5_0 decoder (loader).** The Gemma 3 1B Q4_K_M GGUF mixes
   `Q5_0` into its attention tensors — the standard llama.cpp
   quant recipe for small models (verified identical across the
   ggml-org and bartowski conversions). Added `decode_q5_0` for
   the legacy ggml `block_q5_0` format (32-element, 22-byte
   blocks: f16 scale + 4-byte high-bit field + 16-byte nibbles)
   and threaded `Q5_0` through the loader's dtype gate.

**Out of scope.** Gemma 3 4B *safetensors* is the multimodal
`Gemma3ForConditionalGeneration` wrapper — it has a `vision_config`
and nests the text config under `text_config`, so the config
parser hard-errors on the absent top-level `vocab_size`.
Classified out of scope per the text-only focus (no vision /
encoder work). The text-only path is covered by the Gemma 3 4B
Q4_K_M GGUF (llama.cpp's GGUF conversion extracts the text model),
which loads and generates coherently.

**Files changed.** `src/nn/llama/config.rs` (eos_token_ids +
rope_local_base_freq + sliding_window_pattern + parser helper),
`src/nn/llama/gemma3.rs` (new — builder + weight mappers),
`src/model_adapters/gemma3.rs` (new — `Gemma3Adapter`),
`src/model_adapters/tensor_spec.rs` (`GEMMA3_SPEC` + transform
tables), `src/model_adapters/mod.rs` (`ModelFamily::Gemma3` +
registry), `src/model_adapters/conformance.rs` (Gemma 3 +
Q5_0 entries), `src/v17/loader/gguf_config.rs` (gemma3 arch +
eos_token_ids + dual-RoPE metadata), `src/v17/loader/gguf_to_hf_naming.rs`
(`gemma3_gguf_extra`), `src/v17/loader/gguf_decode.rs`
(`decode_q5_0`), `src/nn/llama/generator.rs` +
`src/nn/llama/pipeline.rs` (multi-EOS), `src/tokenizer/mod.rs`
(`token_to_id`), `src/nn/llama/mod.rs` (`pub mod gemma3`).

**Tests.** `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **409 / 409 PASS**;
`cargo test --test tinyllama_config_test` → 15 PASS / 3 ignored;
TinyLlama Q4_K_M GGUF regression smoke coherent ("...is Paris.").

**Non-claims.** The 9B / 4B GGUF checkpoints have no numcert
manifest — the smoke is functional, not the ADR-004 strict gate.
Gemma 3 sliding-window attention is deferred (full causal below
the window). Gemma 3 12B / 27B were not downloaded; the 12B/27B
are multimodal-wrapped like the 4B. CodeGemma and Gemma 2 27B
(optional targets) were not exercised.

**Conclusion.** Gemma is dominated for the validated text scope:
Gemma 2 safetensors + GGUF (Q4_K_M / Q5_K_M / Q6_K), Gemma 3
(text) 1B safetensors + GGUF and 4B GGUF all load and generate
coherently, with SoftCap / GeGLU / dual-norm / QK-Norm / dual-RoPE
correct, no per-model hacks, and no core changes. Multimodal
Gemma 3 remains explicitly out of scope.

---

## Phi-family mastery validation

Pre-Adapter-Toolkit-v2 audit of the Phi family (Phi-3 / Phi-3.5 /
Phi-4). Phi-3 / Phi-3.5 were already supported — fused QKV, fused
`gate_up`, LongRope — via `Phi3Adapter` / `build_phi3`. This phase
extended the adapter to the rest of the family and exercised the
GGUF quant paths.

**Phase 6 smokes** — prompt `"What is the capital of France?"`,
greedy, certified mode — **8 / 8 PASS**:

| Checkpoint                | Format       | Output                          |
|---------------------------|--------------|---------------------------------|
| Phi-3.5-mini-instruct     | safetensors  | coherent (max-tokens)           |
| Phi-3.5-mini-instruct     | Q4_K_M GGUF  | coherent (max-tokens)           |
| Phi-3.5-mini-instruct     | Q5_K_M GGUF  | coherent (max-tokens)           |
| Phi-3.5-mini-instruct     | Q6_K GGUF    | "…Notre-Dame Cathedral." (EOS)  |
| Phi-3-mini-4k-instruct    | safetensors  | coherent (max-tokens)           |
| Phi-3-mini-128k-instruct  | safetensors  | "…Notre-Dame Cathedral." (EOS)  |
| Phi-4-mini-instruct       | safetensors  | coherent (max-tokens)           |
| Phi-4-mini-instruct       | Q4_K_M GGUF  | coherent (max-tokens)           |

All four Phi checkpoints carry HF architecture `Phi3ForCausalLM` /
`model_type = phi3`, so they route to the existing `Phi3Adapter`
unchanged. Phi-4-mini is the same architecture class with three
config-driven differences (GQA, partial rotary, a 200 k vocab).

**Gaps found and fixed (builder / config — no core change, no
new AMG ops):**

1. **Plain-RoPE Phi-3 (graph/builder).** `build_phi3` /
   `build_phi3_with_store` `panic!`'d unless the config declared a
   `rope_scaling.type = "longrope"` block. Phi-3-mini-4k is the
   4k-context model — it ships **no** `rope_scaling` (plain RoPE)
   — so it crashed on load. Fix: `resolve_phi3_longrope` maps an
   absent `rope_scaling` to a unit-factor `RopeScalingLongRope`
   (all-1.0 short/long factors, `original == max` position
   embeddings), which degenerates exactly to standard RoPE — the
   inverse frequencies are unscaled and `attention_factor` is 1.0.
   One builder code path; the `panic!` survives only for a
   genuinely wrong scaling type (`llama3` on a Phi checkpoint).

2. **Phi-4 partial rotary (config + graph).** Phi-4-mini sets
   `partial_rotary_factor = 0.75`: RoPE rotates only the first
   `round(0.75 · head_dim) = 96` of the 128 head dimensions; the
   remaining 32 pass through un-rotated (the layout HF
   `Phi3Attention` uses when `partial_rotary_factor < 1`). Added
   `LlamaConfig::partial_rotary_factor` + the `rotary_dim()`
   helper; `apply_phi3_rope` / `apply_phi3_rope_with_offset` slice
   q / k into a rotary head and a pass-through tail, RoPE the
   rotary head, and `Concat` the two back — built from existing
   `SliceLastDim` / `Concat` / `RoPE` nodes, full rotary
   (`rotary_dim == head_dim`, Phi-3 / 3.5) takes the unchanged
   single-`rope` path. A Phi-4 GGUF stores the rotated count as
   `phi3.rope.dimension_count`; the GGUF config parser converts it
   back to the HF-style fraction.

3. **Phi-4 grouped-query attention (graph/builder).** Phi-4-mini
   is GQA — 24 query heads, 8 KV heads — where Phi-3 / Phi-3.5 are
   MHA (32 / 32). The Llama builder expands grouped-query K/V by
   tiling the *standalone* `k_proj` / `v_proj` weight at load time
   (`TileGroupedDim`). Phi-3 fuses Q/K/V into one `qkv_proj` and
   slices the *activation*, so the builder produced 8-head K/V
   against 24-head Q → `BatchMatMul4D dim 1 mismatch: 24 vs 8`.
   Fix: `expand_kv_heads_flat` repeats each KV head `kv_groups`
   times in the graph, in the interleaved order HF `repeat_kv`
   produces (`[kv0×g, kv1×g, …]`), from `SliceLastDim` / `Concat`.
   K/V then carry `n_heads_q` heads through reshape / scale / RoPE
   / cache — identical to how the Llama path caches load-time
   tiled K/V. No-op for the MHA variants (`kv_groups == 1`).

Phi-4-mini's 200 k tiktoken-derived tokenizer, `tie_word_embeddings
= true`, and fused QKV/gate_up were already handled by the
existing adapter (`build_phi3` honours the tie; `split_fused_qkv`
is GQA-aware). The chat turn-terminator `<|end|>` was added to the
pipeline's stop-token resolution (Phi instruct ends a turn with
`<|end|>`, not its `eos_token`) — the same multi-EOS mechanism the
Gemma phase introduced.

**Files changed.** `src/nn/llama/config.rs`
(`partial_rotary_factor` field + parse + `rotary_dim()`),
`src/nn/llama/phi3.rs` (`resolve_phi3_longrope`, `apply_phi3_rope`
/ `apply_phi3_rope_with_offset`, `expand_kv_heads_flat`, both
builders), `src/v17/loader/gguf_config.rs`
(`rope.dimension_count` → `partial_rotary_factor`),
`src/nn/llama/pipeline.rs` (`<|end|>` turn-terminator),
`src/nn/llama/gemma2.rs` (struct-literal field).

**Tests.** `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **409 / 409 PASS**
(one flaky failure on the first run — the GPU VRAM-probe test
`legacy_probe_is_unwrap_or_zero_of_detailed`, contending with
concurrent smokes; clean on a re-run with no GPU load);
`cargo test --test tinyllama_config_test` → 15 PASS / 3 ignored.

**Non-claims.** The Phi GGUF checkpoints have no numcert manifest
— the smokes are functional, not the ADR-004 strict gate. Phi-4
(14B) and Phi-3-medium (optional targets) were not downloaded.
Phi-4-multimodal / Phi-vision are out of scope (multimodal) and
were not touched.

**Conclusion.** Phi is dominated for the validated text scope:
Phi-3-mini-4k / Phi-3-mini-128k / Phi-3.5-mini safetensors,
Phi-3.5-mini GGUF (Q4_K_M / Q5_K_M / Q6_K), and Phi-4-mini
safetensors + GGUF all load and generate coherently, with
LongRope, plain RoPE, partial rotary, fused QKV, fused gate_up,
and grouped-query attention all correct — no per-model hacks and
no core changes.

---

## Mistral-dense mastery validation

Pre-Adapter-Toolkit-v2 audit of the Mistral dense family
(7B v0.1 / v0.2 / v0.3). Mixtral / Mistral-MoE are explicitly out
of scope — the focus is the dense causal text models.

**Phase 2 audit — no gaps.** Mistral dense is pure Llama
topology: GQA (32 query / 8 KV heads), SwiGLU MLP, RMSNorm,
standard RoPE, no QKV bias, untied LM head. `MistralAdapter`
(in `model_adapters/llama_family.rs`) already delegates the
scratch graph, the store-backed graph, and both the HF and GGUF
weight mappers to the Llama path; Mistral GGUF exports under
`general.architecture = "llama"`, so a Mistral GGUF resolves to
the Llama-family adapter directly. The config audit (`mistral-7b-v0.3`:
`MistralForCausalLM` / `model_type = mistral`, GQA 32/8, head_dim
derived 128, `rope_theta = 1e6`, `sliding_window = null`,
`attention_bias` absent → `false`, `tie_word_embeddings = false`)
surfaced nothing the existing path does not already handle. Like
the Llama mastery phase, this was a **validation-only** phase —
no source file was modified.

**Phase 6 smokes** — prompt `"What is the capital of France?"`,
greedy, certified mode — **7 / 7 PASS**:

| Checkpoint                    | Format       | Output                       |
|-------------------------------|--------------|------------------------------|
| Mistral-7B-v0.3 (base)        | safetensors  | "Paris is the capital…"      |
| Mistral-7B-Instruct-v0.3      | safetensors  | "…is Paris. …Eiffel Tower…"  |
| Mistral-7B-Instruct-v0.3      | Q4_K_M GGUF  | "…is Paris. …Louvre…"        |
| Mistral-7B-Instruct-v0.3      | Q5_K_M GGUF  | "…is Paris. …Eiffel Tower…"  |
| Mistral-7B-Instruct-v0.3      | Q6_K GGUF    | "…is Paris. …Notre-Dame…"    |
| Mistral-7B-Instruct-v0.2      | safetensors  | "…city of France is Paris…"  |
| Mistral-7B-Instruct-v0.2      | Q4_K_M GGUF  | "…city of France is Paris…"  |

The base model was smoked with `--no-chat-template` (raw
completion); the instruct variants apply the `[INST] … [/INST]`
template from `tokenizer_config.json`. All outputs were coherent
— no loops, no degenerate tokens, no empty output. Each ran to
the 60-token cap (Mistral instruct is verbose; the text is fully
coherent throughout — not a stop defect). v0.2 carries
`sliding_window = 4096` and v0.3 dropped it; both are below the
smoke context length, where sliding-window attention is
equivalent to full causal — no divergence observed (the
sliding-window restriction itself remains deferred, as for
Gemma 2 / 3).

**Files changed.** None — `scripts/dl_mistral.py` was added to
fetch the checkpoints, but no engine source was modified.

**Tests.** `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **409 / 409 PASS**;
`cargo test --test tinyllama_config_test` → 15 PASS / 3 ignored
(baseline re-run — confirms no regression in the other families).

**Non-claims.** The Mistral GGUF checkpoints have no numcert
manifest — the smokes are functional, not the ADR-004 strict
gate. Mistral-7B-v0.1, Mistral-Nemo, and Ministral-8B (optional
targets) were not downloaded; Nemo and Ministral are dense and
would route through the same Llama path, but were not validated
end-to-end. Mixtral / Mistral-MoE / Pixtral are out of scope and
were not touched — no MoE code path exists or was added.

**Conclusion.** Mistral dense is dominated for the validated
scope: 7B v0.2 / v0.3 base + instruct safetensors and GGUF
(Q4_K_M / Q5_K_M / Q6_K) all load and generate coherently through
the existing Llama-family path, with no per-model hacks, no core
changes, and no accidental MoE support.

---

## SmolLM-family mastery validation

Pre-Adapter-Toolkit-v2 audit of the SmolLM / SmolLM2 family, plus
a Falcon-3 regression. The focus is dense causal text models.

**Phase 2 audit — no gaps.** SmolLM and SmolLM2 are built
directly on the Llama architecture: every checkpoint declares
`architectures = ["LlamaForCausalLM"]` and `model_type =
"llama"`, so they resolve to `LlamaFamilyAdapter` with no
SmolLM-specific code — there is no SmolLM adapter and none is
needed. Config audit (`smollm2-1.7b-instruct`: MHA 32/32,
`head_dim` derived 64, `rope_theta = 130000`,
`rms_norm_eps = 1e-5`, `tie_word_embeddings = true`,
`vocab_size = 49152`, no `rope_scaling`) is plain Llama topology
with a tied LM head — already handled by `build_llama`'s
`matmul_rhs_transposed` tied branch. Falcon-3 is also
`LlamaForCausalLM` (GQA 12/4, explicit `head_dim = 256`) and
rides the identical path. As with the Llama and Mistral mastery
phases, this was a **validation-only** phase — no source file was
modified.

**Phase 6 smokes** — prompt `"What is the capital of France?"`,
greedy, certified mode — **9 / 9 PASS** (+ 1 Falcon-3 regression):

| Checkpoint                  | Format       | Output                       |
|-----------------------------|--------------|------------------------------|
| SmolLM2-135M-Instruct       | safetensors  | coherent (verbose, max-tok)  |
| SmolLM2-360M-Instruct       | safetensors  | coherent refusal*            |
| SmolLM2-1.7B-Instruct       | safetensors  | "…is Paris." (EOS)           |
| SmolLM2-1.7B-Instruct       | Q4_K_M GGUF  | "…is Paris." (EOS)           |
| SmolLM2-1.7B-Instruct       | Q5_K_M GGUF  | "…is Paris." (EOS)           |
| SmolLM2-1.7B-Instruct       | Q6_K GGUF    | "…is Paris." (EOS)           |
| SmolLM-135M-Instruct        | safetensors  | coherent (verbose, max-tok)  |
| SmolLM-360M-Instruct        | safetensors  | "…city of Paris…" (EOS)      |
| SmolLM-1.7B-Instruct        | safetensors  | "…is Paris…" (EOS)           |
| Falcon-3-7B-Instruct        | safetensors  | "…is Paris." (EOS)           |

\* SmolLM2-360M produced a well-formed "I can't provide
geographical information" refusal — a genuine small-model
behaviour (coherent English, no loop, no degenerate token), not
an engine defect. The 135M models are verbose and mildly
repetitive at the tail (greedy decoding on a 135M model), again a
model-capacity artifact — the first sentence is correct and
coherent in every case. The 1.7B variants halt cleanly on EOS.

**Files changed.** None — `scripts/dl_smollm.py` was added to
fetch the checkpoints; no engine source was modified.

**Tests.** `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **409 / 409 PASS**;
`cargo test --test tinyllama_config_test` → 15 PASS / 3 ignored
(baseline re-run — no regression in any other family).

**Non-claims.** The SmolLM2 GGUF checkpoints have no numcert
manifest beyond SmolLM2-1.7B's existing ADR-004 F64 fixture entry
— the GGUF smokes are functional, not the strict gate. SmolLM
base (non-instruct) models were not exercised. Falcon-3 was
smoked as a single-checkpoint regression, not a full sweep of the
Falcon-3 size range.

**Conclusion.** SmolLM is dominated for the validated scope:
SmolLM2 135M / 360M / 1.7B and SmolLM 135M / 360M / 1.7B instruct
safetensors, plus SmolLM2-1.7B GGUF (Q4_K_M / Q5_K_M / Q6_K), all
load and generate coherently through the existing Llama-family
path — tied LM head correct, no empty output, no degenerate
loops, no per-model hacks, no core changes. Falcon-3 confirmed as
a clean Llama-path regression.

---

## Falcon-family mastery validation

Pre-Adapter-Toolkit-v2 audit of the Falcon family, with an
explicit classification of modern Falcon3 versus classic Falcon.
Falcon multimodal / vision / MoE / encoder-decoder variants are
out of scope — the focus is dense causal text models.

**Phase 2 audit — Falcon3 is Llama, classic Falcon is not.**
Falcon3 (1B / 3B / 7B-Instruct) declares
`architectures = ["LlamaForCausalLM"]` and `model_type =
"llama"`, so it resolves to `LlamaFamilyAdapter` with no
Falcon-specific code — there is no Falcon adapter and Falcon3
needs none. Config audit (`falcon3-7b-instruct`: GQA 12/4,
explicit `head_dim = 256`, `rope_theta = 1000042`,
`rms_norm_eps = 1e-6`, `tie_word_embeddings = false`,
`vocab_size = 131072`, no `rope_scaling`) is plain Llama topology;
Falcon3 GGUF exports under `general.architecture = "llama"`.
Classic Falcon (`falcon-7b-instruct`: `FalconForCausalLM`,
`model_type = "falcon"`, `multi_query = true`,
`parallel_attn = true`, `new_decoder_architecture = false`,
`alibi = false`, LayerNorm via `layer_norm_epsilon`) is a
genuinely distinct architecture — LayerNorm instead of RMSNorm,
a single shared norm feeding parallel attention + MLP, and a
multi-query fused QKV layout. Supporting it would require a new
graph builder and new AMG nodes, which trips the phase STOP
rule, so classic Falcon is **classified out of scope** rather
than forced onto the Llama path.

**Phase 4 fix — one config-layer tolerance.** Falcon3-1B and
Falcon3-3B-Instruct omit `bos_token_id` from `config.json` (it
is kept only in `generation_config.json`). The config parser
required the field and rejected the load. Fix: `get_bos_token_id`
falls back to `eos_token_id` when `bos_token_id` is absent —
generalizable, and behaviour-neutral because
`LlamaConfig.bos_token_id` is not consumed by the generation
path (the tokenizer owns BOS via its own `bos_token`). An
explicit `bos_token_id` is still honoured verbatim. No core,
adapter, or graph-builder change.

**Phase 6 smokes** — prompt `"What is the capital of France?"`,
greedy, certified mode — **6 / 6 PASS**:

| Checkpoint               | Format       | Output               |
|--------------------------|--------------|----------------------|
| Falcon3-1B-Instruct      | safetensors  | "…is Paris." (EOS)   |
| Falcon3-3B-Instruct      | safetensors  | "…is Paris." (EOS)   |
| Falcon3-7B-Instruct      | safetensors  | "…is Paris." (EOS)   |
| Falcon3-7B-Instruct      | Q4_K_M GGUF  | "…is Paris." (EOS)   |
| Falcon3-7B-Instruct      | Q5_K_M GGUF  | "…is Paris." (EOS)   |
| Falcon3-7B-Instruct      | Q6_K GGUF    | "…is Paris." (EOS)   |

Every Falcon3 checkpoint emitted *"The capital of France is
Paris."* and halted cleanly on EOS — no loops, no degenerate
tokens, no empty output. Classic Falcon was load-tested for
classification: `falcon-7b-instruct` safetensors fails loud with
a typed config error (missing `num_key_value_heads`), and
`falcon-7b-instruct` Q4_K_M GGUF fails loud with
`unsupported general.architecture = "falcon"`. Both are clean
fail-loud rejections, not crashes — the expected behaviour for an
unsupported architecture.

**Files changed.** `src/nn/llama/config.rs` — added
`get_bos_token_id` (fallback to `eos_token_id`) plus two unit
tests. `scripts/dl_falcon.py` added to fetch the checkpoints. No
other engine source modified; the adapter registry, graph
builders, and GGUF whitelist are untouched.

**Tests.** `cargo build --lib` clean;
`cargo test --lib -- --test-threads=1` → **411 / 411 PASS**
(409 baseline + 2 new bos-fallback tests);
`cargo test --test tinyllama_config_test` → 15 PASS / 3 ignored
(no regression in any other family).

**Non-claims.** The Falcon3 GGUF checkpoints have no numcert
manifest — the GGUF smokes are functional, not the ADR-004
strict gate. Falcon2-11B and Falcon3-10B (optional targets) were
not downloaded or validated. Classic Falcon
(`FalconForCausalLM` / `RWForCausalLM`) is **not supported** — no
graph builder, AMG node, ALiBi path, parallel-attention path, or
multi-query QKV transform was added; classic Falcon is recorded
as out-of-scope architecture, not a regression.

**Conclusion.** Falcon3 is dominated for the validated scope:
1B / 3B / 7B-Instruct safetensors and Falcon3-7B-Instruct GGUF
(Q4_K_M / Q5_K_M / Q6_K) all load and generate coherently through
the existing Llama-family path, with one generalizable
config-tolerance fix and no core changes. Verdict: **Falcon3
dominada; Falcon clásico fuera de scope actual.**

---

## Adapter Toolkit v2

A declarative layer **on top of** the v1 model-adapter system
(`src/model_adapters/`). It lets a model be described by a small
YAML/JSON document instead of a hand-written Rust adapter. It is a
strict addition: it never modifies v1, never touches the runtime
core or the graph builders, and never generates Rust code.

**Module — `src/adapter_toolkit/`.**

- `dsl.rs` — serde schema for the three authoring levels (simple /
  intermediate / advanced); parses `.yaml`/`.yml` (serde_yaml) and
  `.json` (serde_json). `deny_unknown_fields` makes a misspelled
  key a hard parse error.
- `spec.rs` — `ResolvedAdapterSpec`, the IR: the DSL `family`
  resolved to a v1 `ModelFamily` + architecture, the feature flags
  normalised into a `FeatureSet` (the Part-5 pattern catalog:
  `RopeKind`, `AttentionKind`, `KvHeadsResolved`, fused-QKV/MLP),
  and the per-checkpoint overrides layered.
- `generator.rs` — `GeneratedAdapter`. A v2 adapter holds a
  `&'static dyn AteniaModelAdapter` (the v1 hand-written adapter
  for the family) and implements the v1 7-trait supertrait by
  **pure delegation**. The DSL parameterises an existing family; it
  never defines a new architecture. Graph topology, weight mapping
  and GGUF naming stay v1's, untouched.
- `registry.rs` — `AdapterRegistry`: dynamic, **v2-first /
  v1-fallback**. Lookup by metadata, name, or family. v1 is never
  shadowed for models with no v2 spec.
- `validate.rs` — declarative validators. Errors (blocking): `gqa`
  without `kv_heads`, `fused_qkv` without `split_strategy`,
  `partial_rotary_factor` contradicting an explicit `rope`,
  out-of-range `partial_rotary_factor`, `mqa` with `kv_heads != 1`,
  unknown family/architecture. Warnings: fused weights / LongRope
  declared on a family whose v1 builder has no such path; an empty
  `eos_tokens` list.
- `introspect.rs` — human-readable rendering of a generated
  adapter: family, architecture, v1 base adapter, capabilities,
  feature set, overrides, and a GGUF→HF tensor-name sample.
- `inspect.rs` — auto-detection. Reads a model dir, detects format
  (`config.json` vs `*.gguf`), family/architecture, attention shape
  (MHA/GQA/MQA from the head counts), EOS set, and RoPE variant
  (`longrope`/`partial`/`standard`), and emits an `AdapterDsl`. The
  emitted YAML is self-checked to validate and resolve, so
  `atenia inspect` output feeds straight into `atenia load`.

**CLI — three subcommands in `src/bin/atenia.rs`.**

- `atenia load <file>` — parse the DSL, validate, build the v2
  adapter, print the summary. **Never runs generation.** Exit 2 on
  any toolkit error.
- `atenia debug <file>` — same, verbose (v1 capabilities + tensor
  sample) plus validation warnings.
- `atenia inspect <dir>` — auto-detect a model directory and emit a
  loadable YAML DSL plus a resolved-spec preview.

Generation stays exactly where it was — `atenia generate` and the
v1 load path are untouched.

**Examples.** `config/adapters/{llama,qwen,gemma,phi,mistral}.yaml`
— one per authoring level, each loadable directly with
`atenia load`.

**Validation.** `cargo build --lib` clean (one new dependency:
`serde_yaml 0.9`, pure-Rust). `cargo test --lib -- --test-threads=1`
→ **475 / 475 PASS** (411 v1 baseline + 64 adapter_toolkit), zero
regressions. `cargo test --test tinyllama_config_test` → 15 PASS /
3 ignored. CLI smokes: `load` / `debug` on all five examples;
`inspect` on TinyLlama, Llama 3.2, Qwen 2.5, Gemma 2, Phi-3.5, with
the emitted YAML round-tripping back through `load`; failure cases
(`gqa` without `kv_heads`, unknown family, classic Falcon dir) all
fail loud with exit 2; v1 `atenia generate` on TinyLlama unchanged
(*"The capital of France is Paris."*, EOS).

**Technical-debt audit (post-completion).** Three fragile points
were reviewed and hardened:
1. *GGUF RoPE detection.* GGUF metadata carries no RoPE-variant
   tag (llama.cpp folds LongRoPE / partial-rotary into the
   `rope_factors` tensors). `inspect` no longer guesses — it leaves
   `rope` unset (standard) and emits an explicit
   [`InspectionReport`] note, surfaced as a comment in the
   generated YAML, that long-context models may need
   `config.rope: longrope` added by hand.
2. *Declarative-vs-authoritative DSL sections.* The `config` /
   `weights` / `attention` sections are expected constraints —
   validated and used for introspection, **never** applied to the
   model's `LlamaConfig`. This is now explicit in the struct doc
   comments, the introspection output label ("declared features
   (validated, not applied — config.json is authoritative)"), and
   the example YAML comments. The YAML cannot appear to control
   something it does not.
3. *`serde_yaml 0.9` deprecation.* The crate is unmaintained
   (`0.9.34+deprecated`). Accepted, not replaced: the exposure is
   contained to the DSL front-end, off the inference hot path, and
   the maintained `serde_json` path is a complete substitute. A
   migration TODO is recorded in `Cargo.toml` and `dsl.rs`.

Hidden-debt sweep — confirmed and fixed: an explicit unrecognised
`model_type` used to be silently swapped for the family default
(now a hard `Resolution` error); a `split_strategy` with no
`fused_qkv` is now a validator warning. Reviewed and accepted: the
resolve path runs `ResolvedAdapterSpec::resolve` more than once per
`load` (cheap, pure, single source of truth — no divergence); the
`resolve_family` family→architecture table mirrors knowledge v1
owns, but `GeneratedAdapter::from_spec` re-checks against v1's real
`resolve_adapter` and asserts the family matches, so a stale table
fails loud rather than silently. `inspect` output verified
round-trippable for HF and GGUF inputs.

**Out of scope.** The DSL only parameterises the seven v1
families. An unknown family / architecture (e.g. classic Falcon
`FalconForCausalLM`) is a typed error, never a forced fallback —
the toolkit never invents a builder. `atenia load` deliberately
does not run generation.

---

## Beyond v20

The roadmap horizons (v21/M12 production hardening → v22 Intel iGPU → v23 AMD
ROCm → v24 Apple Metal → v25 distributed → training at v25+) are tracked in
[ROADMAP.md](../ROADMAP.md). None are shipped; the codebase is structured to
make them possible, not to claim them.
