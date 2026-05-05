# Atenia Engine — Roadmap

This document describes the priorities guiding Atenia Engine's development. It is organized by upcoming APX version, from in-progress work to broader horizons.

This roadmap communicates scope and priority, not calendar commitments. Versions are released when ready.

---

## Status overview

Atenia Engine is currently working through APX v20 (Real Model Runtime Integration). Earlier versions (v12 through v19) are complete. The most recently closed sub-milestone is **M8.6: BF16 KV cache (D62 resolved; default flipped on after the TinyLlama 1.1B-Chat 8-token determinism fixture came back bit-identical; 1.6 GiB RAM savings at seq=2048 on Llama 2 13B; legacy F32 path preserved behind `ATENIA_LEGACY_F32_KV_CACHE=1`)**. The previous closure was **M8.7: Disk → GPU JIT pipeline (1.30× over the M8 baseline on Llama 2 13B; 20.7 s/tok with `ATENIA_M8_7_ENABLED=1`; ADR-004 numerics preserved)**. The full M4.7 → M4.8 → M4.9 → M5 → M6 → M7 → M8 → M8.7 → M8.6 trajectory is closed: Llama 2 13B Chat runs end-to-end on dev-class commodity hardware (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, NVMe spill cache); as of M6 the tier-aware loader doubled the speed of the 7B Chat over the CPU baseline; as of M7 the 13B Chat ran the first time on the box without BSOD via automatic Disk overflow; as of M8 the BF16-resident VRAM kernels gave 1.36× over M7.3; and as of M8.7 **`ATENIA_M8_BF16_KERNEL=1 ATENIA_M8_7_ENABLED=1 atenia generate --model models/llama-2-13b-chat` streams 154 disk-tier weights per forward through the BF16 GPU dispatch with a 98.7 % CPU prefetch hit rate, dropping the 13B per-token cost from 27.0 s/tok (M8) to 20.7 s/tok — 1.30× faster than the M8 baseline, argmax bit-exact with M8**.

**Next active milestone: M9 (INT8 quantisation)**. M8.7.1.b/c (async H→D + dedicated copy/compute streams) was deferred at the close of M8.7 because the dev-box VRAM working-set budget (~540 MiB free after residents + headroom) cannot accommodate the two-buffer pipeline's peak ~670 MiB. M9 attacks the same throughput frontier from the other side: halve again from 2 to 1 byte per weight, dropping the Llama 2 13B from BF16 26 GiB to INT8 13 GiB and eliminating the M8.7 disk-overflow problem space rather than optimising within it. The 1.6 GiB freed by M8.6 in long contexts directly feeds M9's tier planner working-set budget.

See [docs/HANDOFF_APX_V20_M8.6.md](./docs/HANDOFF_APX_V20_M8.6.md) for the most recent closing notes; [docs/HANDOFF_APX_V20_M8.7.md](./docs/HANDOFF_APX_V20_M8.7.md) for the prior milestone.

Detailed closing notes per milestone live in the `docs/` directory:

- [docs/HANDOFF_APX_V20_M3.md](./docs/HANDOFF_APX_V20_M3.md) — reactive execution context, real GPU storage, M3-e migration loop
- [docs/HANDOFF_APX_V20_M4.md](./docs/HANDOFF_APX_V20_M4.md) — safetensors loader and weight mapping mechanics
- [docs/HANDOFF_APX_V20_M4.5.md](./docs/HANDOFF_APX_V20_M4.5.md) — real model execution end-to-end (`TinyLlama-1.1B`)
- [docs/HANDOFF_APX_V20_M4.6.md](./docs/HANDOFF_APX_V20_M4.6.md) — Llama-family compatibility expansion (four checkpoints, F64 validation methodology)
- [docs/HANDOFF_APX_V20_M4.7.md](./docs/HANDOFF_APX_V20_M4.7.md) — beyond-VRAM killer demo (Llama 2 13B Chat, transparency contract closed)
- [docs/HANDOFF_APX_V20_M4.8.md](./docs/HANDOFF_APX_V20_M4.8.md) — performance optimisation (3.5× on 13B; 49.5× on the production matmul shape; vendor-agnostic AVX2/FMA + matrixmultiply)
- [docs/HANDOFF_APX_V20_M4.9.md](./docs/HANDOFF_APX_V20_M4.9.md) — public CLI demo (`atenia run --mode c` reproduces the momento guau in 6.9 min via one command)
- [docs/HANDOFF_APX_V20_M5.md](./docs/HANDOFF_APX_V20_M5.md) — tokenizer + KV cache + token-by-token generation (`atenia generate` ships; Llama 2 13B Chat answers conversationally; Arc-shared weights at 24.24 GiB)
- [docs/HANDOFF_APX_V20_M6.md](./docs/HANDOFF_APX_V20_M6.md) — tier-aware GPU loader (VRAM → RAM → NVMe planner; 1.46× speedup on Llama 2 7B Chat; bit-identical output)
- [docs/HANDOFF_APX_V20_M7.md](./docs/HANDOFF_APX_V20_M7.md) — 13B-friendly tiers (Disk fast-path + adaptive RAM headroom; Llama 2 13B Chat end-to-end with 239 tensors on NVMe, 7.36 GiB RAM headroom, no BSOD)
- [docs/HANDOFF_APX_V20_M8.md](./docs/HANDOFF_APX_V20_M8.md) — BF16-resident VRAM kernels (Path B: BF16 storage + F32 upcast per-matmul; 1.31× on Llama 2 7B, 1.36× on Llama 2 13B; ADR-004 4-model F64 validation passes with margin 21–12,500×)
- [docs/HANDOFF_APX_V20_M8.7.md](./docs/HANDOFF_APX_V20_M8.7.md) — Disk → GPU JIT pipeline (M8.7.0 single-tensor staging + M8.7.1.a CPU prefetch + M8.7.1.r Path B stream-aware refactor; 13B 20.7 s/tok with 154 disk-streamed matmuls per forward and 98.7 % prefetch hit rate; M8.7.1.b/c deferred for VRAM-budget reasons, documented for future revival)
- [docs/HANDOFF_APX_V20_M8.6.md](./docs/HANDOFF_APX_V20_M8.6.md) — BF16 KV cache (D62 resolved; runtime ledger F32→BF16 cast in harvest, BF16→F32 in reinject; graph stays F32; TinyLlama 1.1B determinism fixture bit-identical under BF16 default; 1.6 GiB savings at seq=2048 on 13B; `ATENIA_LEGACY_F32_KV_CACHE=1` opt-out)

---

## Current focus: APX v20 — Real Model Runtime Integration

APX v20 connects the completed telemetry and decision infrastructure to real external model execution. The first target was HuggingFace `safetensors` checkpoints. The milestone is structured into sub-phases:

### Completed sub-phases

- **M1** — `Conv2D` and `MaxPool2D` natively in the Adaptive Model Graph, with forward, backward, tape integration, and finite-difference gradient checking.

- **M2** — Reactive execution context attached to the graph; the executor consults guard state before each node and returns typed abort reasons on guard verdicts. Existing APIs preserved as backward-compatible wrappers.

- **M3** — Real GPU allocation for tensors behind a vendor-neutral storage abstraction (`TensorStorage`), real host↔device transfers, and the M3-e reaction loop that moves real VRAM to RAM on guard `Degrade` verdicts.

- **M4** — Model loader mechanics: a `safetensors` reader (header + body, by-name and iterator access), a `WeightMapper` with shape validation and structured `LoadReport` diagnostics, and BF16 / F16 → F32 decode. Validated empirically against a real HuggingFace gpt2 checkpoint.

- **M4.5** — End-to-end real model execution. The engine loads a HuggingFace `TinyLlama-1.1B-Chat-v1.0` checkpoint and runs forward on CPU. Logits match a PyTorch reference within F32-vs-BF16 precision drift over 22 transformer blocks (max absolute diff ≈ 0.73, mean ≈ 0.06, no values diverging by more than 1.0). New graph primitives landed for this: rotary positional embedding, general permute, broadcast multiplication, and rank-4 batched matmul. A complete Llama-2 graph builder consumes the HuggingFace parameter naming convention directly.

- **M4.6** — Llama-family compatibility expansion. Three production checkpoints added on top of TinyLlama: SmolLM2 1.7B (Phase A — tied word embeddings, configurable RmsNorm eps, generic `nn::llama` rename), Qwen 2.5 1.5B (Phase B — Q/K/V projection biases, `model_type`-aware config defaults), and Llama 3.2 1B Instruct (Phase C — `rope_scaling: "llama3"` piecewise frequency scaling, explicit `head_dim`). Each model validated against PyTorch F64 mathematical ground truth per [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md), with Atenia F32 max drift between 1.32×10⁻⁴ and 1.45×10⁻³ — three to four orders of magnitude closer to truth than industry-default BF16 inference on the same checkpoints. Argmax MATCH 4/4 positions on every model. The Llama 3 scaling wiring is falsified independently with a long-context graph test (seq_len = 2048) that proves the scaled inverse-frequency vector reaches the RoPE kernel through the AMG pipeline.

- **M4.6.1** — Retroactive F64 validation for TinyLlama. The original M4.5-d.1 test (PyTorch BF16 reference) is preserved untouched as historical record of the pre-ADR-004 methodology. A new `tinyllama_f64_validation_test.rs` adds the F64-gated equivalent: Atenia max drift 1.41×10⁻⁴, ratio 5198× vs BF16. Resolves the implicit "PyTorch as ground truth" framing left by M4.5-d.1 — the BF16-argmax disagreement reported there was a near-tie quantisation artefact, not an Atenia bug. See [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md).

### Pending sub-phases

- **M4.6.2** (immediate post-momento-guau candidate) — Phi 3.5 mini Instruct (3.8B). The architectural deltas vs the Llama family are identified and tractable: a `RopeScaling::Longrope` variant with per-dim `short_factor` / `long_factor` vectors and the `attention_factor` post-multiply on cos/sin (a step llama3 does not need); a fused `qkv_proj` and a fused `gate_up_proj` that need to be split at load time; everything else (RmsNorm, SwiGLU, MHA, half-split RoPE) reuses existing primitives. Estimated ~9 calibrated hours of work — slightly above Phase C. Technically viable on the dev hardware (32 GB RAM accommodates the ~15 GB F32 weights + safetensors buffer + activations). The momento guau is closed (M4.7.6.e); M4.6.2 is now unblocked and is the natural next "fifth Llama-family checkpoint" before M5 if Phi 3.5 mini is a target deployment. Otherwise M5 (inference UX) is strictly higher leverage. See the M4.6.2 investigation notes for the full architectural diff.

- **M4.7 ✅** — Beyond-VRAM execution. The killer demo for the v20 thesis: run a 13B-class model in BF16 on the dev hardware end-to-end. Concrete target hardware: **RTX 4070 Laptop with 8 GB VRAM, 32 GB RAM, project root on an external USB SSD (drive F:), runtime data on internal NVMe (drive D:)**. A 13B model in BF16 weighs roughly 26 GB on disk — does not fit in VRAM (8 GB), does not fit in RAM alone (32 GB minus working set leaves no room for activations + KV cache), but is executable end-to-end with VRAM ↔ RAM ↔ disk offload orchestrated by the M3 reaction loop. **Closed at commit `1906415` (M4.7.6.e)**: Llama 2 13B Chat runs end-to-end with `argmax(Mode A clean RAM) == argmax(Mode C forced 50 % LRU spill) == 1, logit 4.7747` bit-exactly — the selective LRU spill + lazy-restore cycle is mathematically transparent at 13 B parameter scale. Mistral 7B is unblocked as a side effect: its architecture is identical to Llama 2 and the only blocker (memory tiering) is now in place.

  **Sub-phase plan:**

  > **A note on the hour estimates.** The numbers attached to each sub-phase are *complexity references*, not a wall-clock forecast. They use the M4.6 calibration (raw estimates / 2.5) as a way to compare relative weight across sub-phases. Actual execution has run dramatically faster — M4.7.2 was estimated at 33–44h and closed in roughly 23 minutes of focused work. Treat the hour figures the way you'd treat story points: useful to compare two items, useless to add up into a calendar.

  - **M4.7.1 ✅** — Sharded safetensors loader (multi-file + `model.safetensors.index.json`, drop-after-decode RAM bound). Closed at commits `28e7bcd` (a — index parser, 9 unit tests), `52a47b2` (b — `ShardedSafetensorsReader` driver, `WeightMapper::load_one_shard_into` extraction, 3 in-memory shard tests bit-identical to single-file load), `105dcc9` (c — Mistral 7B v0.3 integration: 291 tensors across 3 shards, 14.5 GB BF16 loaded end-to-end, `loaded=291 / skipped=0 / missing=0`, drop-after-decode confirmed in practice with peak RAM bounded by ~5 GB per-shard instead of the 15 GB sum). Closed under budget. The four M4.6 single-file checkpoints remain bit-identical; sharded loading is a pure addition.
  - **M4.7.2 ✅** — Native BF16 parameter storage with decode-on-access. Closed at commits `62661c5` (a — `TensorStorage::CpuBf16(Vec<u16>)` variant + accessors with decode-on-access semantics + panic-stubs for off-path consumers; 7 unit tests on the storage primitive), `4570d63` (b — `WeightMapper::set_store_params_as_bf16(bool)` flag with the down-convert at the final write step + cross-path equivalence test against the spike `a786837`), `9378df7` (c+d — five executor seams patched with `ensure_cpu()` decode-on-access pattern: `MatMul`, `IndexSelect`, `BroadcastMul`, `BroadcastAdd`, plus the late-discovered `Transpose2D` arm exercised by the tied-embeddings path; TinyLlama 1.1B BF16-storage smoke test green with 50.0% RAM savings exact, argmax 4/4 vs F64), `f58ada0` (e — full 4-model F64 re-validation per ADR-004). Empirical drift table under native BF16 storage:

      | Model              | Drift vs F64 | Argmax    | RAM saved |
      |--------------------|-------------:|-----------|-----------|
      | TinyLlama 1.1B     |   0.000141   | 4/4 MATCH | 50.0%     |
      | SmolLM2 1.7B       |   0.001446   | 4/4 MATCH | 50.0%     |
      | Qwen 2.5 1.5B      |   0.029057   | 4/4 MATCH | 50.0%     |
      | Llama 3.2 1B       |   0.000132   | 4/4 MATCH | 50.0%     |

      Every model's drift is bit-exact identical to the precision-floor spike (commit `a786837`), confirming end-to-end the cross-path mathematical equivalence that the unit-test cross-path lock proved at the operation level. Five architectural decisions honoured: (1) `store_params_as_bf16` as a `WeightMapper` flag, default `false`; (2) decode-cache deferred as performance follow-up post-M4.7; (3) disk-tier BF16 spill panic-stubbed; (4) backward / training / GPU dispatch panic-stubbed and tagged with the milestone that lifts each restriction; (5) ADR-004 close-criterion = 4-model F64 re-validation under threshold 0.5. The investigation enumerated four executor seams; the fifth (`Transpose2D` consuming `embed_w` in the tied-embeddings path) was caught at run time on the SmolLM2 / Qwen / Llama 3.2 forwards and patched in the same commit — recorded for future storage-variant additions: any executor arm that consumes a graph parameter is a seam, even when the "parameter" is being reshaped or transposed rather than fed to a numeric kernel. The spike `a786837` stays in the codebase as a permanent regression gate (env var off is bit-identical, env var on round-trips through BF16 precision in the loader as a defence in depth against a future regression in the storage variant).

  - **M4.7.3 ✅** — GPU MatMul with resident operands + executor device dispatch. Closed across six surgical sub-steps (a–f), each on its own commit:
    - `66910d5` (a — `cuda_matmul_inplace` with `all_cuda` device-pointer dispatch + per-storage gating in `try_gpu_matmul`; new `Tensor::zeros_new_cuda` and `Tensor::ensure_decoded` primitives; MatMul executor arm reworked to allocate the output in VRAM when both operands live there; residency unit test drift = 1.91e-6).
    - `801eab7` (b — same surgical pattern on the BatchMatMul executor arm: `ensure_decoded` on operands, `zeros_new_cuda` output when both operands are Cuda, residency short-circuit ahead of the kernel-planner target switch; no kernel changes — `cuda_batch_matmul` already had `all_cuda` from M3-d. Residency unit test drift = 1.91e-6).
    - `094b34f` (c+d — defensive `ensure_cpu` audit. **(c)** MatMul + BatchMatMul legacy fall-through paths (APX 5.2/6.2 AVX2, `dispatch_matmul_gpu`, `dispatch_batch_matmul_cuda` host fallback, pure `batch_matmul`) now `ensure_cpu` operands and output before reading via `as_cpu_slice`. **(d)** Preventive coverage on every executor arm whose helper(s) consume `as_cpu_slice` and lacked a guard: `RmsNorm`, `SiLU`, `Softmax`, `RoPE`, `LogSoftmax`, `Linear`, `Activation`, `FusedLinearActivation`, `Gather`, `CrossEntropyLoss`, `Reshape`, `Permute`, `TransposeLastTwo`, `Conv2D`, `MaxPool2D`. Add/Sub/Mul stay on storage-aware Tensor methods. NoOp / Output / FusedLinearActivationChain (intercepted by early-return helper) and Input/Parameter are no-op arms.).
    - **(e)** TinyLlama 1.1B GPU MatMul smoke test under M4.7.3 dispatch — counters `gpu_matmul_resident_count` / `gpu_matmul_roundtrip_count` exposed from `gpu::dispatch::hooks` for observability, asserts at least one MatMul went through the GPU branch + logits finite + argmax 4/4 vs the M4.6.1 F64 fixture.
    - **(f)** Full 4-model F64 re-validation under M4.7.3 dispatch (TinyLlama, SmolLM2, Qwen 2.5, Llama 3.2). Same ADR-004 threshold 0.5, same per-position argmax 4/4 contract as M4.7.2.e, plus the GPU MatMul counter gate so a silent CPU-fallback regression cannot trick the test into passing.

      Four architectural decisions honoured: (1) per-storage gating option (b) — GPU residency triggers on `(Cuda, Cuda, Cuda)` triples regardless of shape gate, since uploading is already paid; (2) `try_gpu_linear` stays hard-disabled (APX 4.11 MiniFlux constraint on backward correctness — tagged for a separate milestone); (3) preventive `ensure_cpu` / `ensure_decoded` audit on every executor arm rather than waiting for runtime panics; (4) ADR-004 threshold 0.5 unchanged. Investigation finding from .a confirmed throughout: 80% of GPU residency infrastructure already existed (linear, batch_matmul, fused_linear_silu had `all_cuda` from M3-d) — only `cuda_matmul` lacked it, and the executor arms needed allocator + dispatcher updates to actually produce VRAM-resident outputs.
  - **M4.7.4 ✅** — RAM↔SSD streaming primitive: BF16-aware disk format, chunked streaming reader, BF16 spill arm, dtype-aware `ensure_cpu` Disk-arm. Closed across six surgical sub-steps each on its own commit:
    - `26f1ca2` (a — `DiskDtype { F32 | BF16 }` enum on `DiskTensorHandle`; `write_bf16_tensor` / `read_bf16_tensor` mirroring the F32 path at native 2-byte width; cross-route guards to surface mistaken dispatch as `InvalidData` instead of garbage. 18 disk_tier unit tests, 9 of them new, the original 9 stay bit-exact).
    - `0611738` (b — chunked streaming reader for both `read_f32_tensor` and `read_bf16_tensor`. 4 MiB per chunk via `File::read_exact`, no `memmap2` dep added. Bounds peak transient memory at one chunk + destination buffer, fixing the OOM-during-restore failure mode that whole-file `fs::read` would have hit on multi-GB tensors. Public signatures unchanged. 3 new tests covering >chunk-size F32, >chunk-size BF16, and truncated-file → `InvalidData`).
    - `ecc3d78` (c — `migrate_all_cpu_to_disk` BF16 arm. The pre-M4.7.4.c arm silently dropped every `CpuBf16` tensor into `tensors_skipped` (graph.rs:575-578), defeating the M4.7.2 50% RAM saving the moment the reactive loop tried to spill a 13B-class checkpoint. Now spills CpuBf16 at native 2-byte width via `write_bf16_tensor`, producing a `DiskDtype::BF16`-tagged handle. F32 arm bit-exact. New `m4_7_4_c_bf16_spill_test` with 3 tests; `m3_e_11_4_migration_test` 9/9 unchanged).
    - `1e602b0` (d — `ensure_cpu` Disk arm dispatches on `handle.dtype()`. F32 path bit-exact; BF16 path reads 2-byte file then per-element `bf16_bits_to_f32` upcast, producing `Cpu(Vec<f32>)` and flipping `tensor.dtype` to `F32` (same lock the M4.7.2 CpuBf16 → Cpu arm enforces). `copy_to_cpu_vec` Disk arm gets the same dispatch. New `m4_7_4_d_disk_bf16_ensure_cpu_test` with 3 tests; the 7 pre-existing `m3_e_11_2_tensor_storage_disk_test` stay bit-exact).
    - `b2ef6f7` (e — TinyLlama 1.1B disk-spill smoke test. Loads with BF16 storage, spills 201 parameters via `migrate_all_cpu_to_disk` (2523 MB on disk vs 5046 MB at F32 — 50% saving on disk too), runs forward letting `ensure_cpu` restore on the fly, asserts argmax 4/4 vs the M4.6.1 F64 fixture. Local result on NVMe SN770: spill throughput 1219 MB/s, forward 26.6 s vs 30 s no-spill baseline. `#[ignore]`-gated; documents `ATENIA_DISK_TIER_DIR` pinned to an internal NVMe as the runtime prerequisite — F: USB HDD throughput is roughly 25× slower and is not viable for the demo).
    - **(f)** — Full 4-model F64 re-validation under disk spill. Same ADR-004 threshold 0.5 + per-position argmax 4/4 contract as M4.7.2.e and M4.7.3.f. Local result on NVMe SN770:

      | Model | Drift vs F64 | Argmax | Disk MB | Spill MB/s | Restore MB/s |
      |---|---:|:-:|---:|---:|---:|
      | TinyLlama 1.1B | 0.000141 | 4/4 | 2523 | 1298 |  87.6 |
      | SmolLM2 1.7B   | 0.001446 | 4/4 | 3423 | 1356 |  50.1 |
      | Qwen 2.5 1.5B  | 0.029057 | 4/4 | 3308 | 1347 | 158.1 |
      | Llama 3.2 1B   | 0.000132 | 4/4 | 2673 | 1377 |  58.4 |

      Drift bit-exact identical to M4.7.2.e and M4.7.3.f baselines, confirming the disk spill + restore cycle is mathematically transparent (the BF16 → F32 upcast in the Disk-arm `ensure_cpu` is a pure zero-fill of the trailing 16 mantissa bits).

      Six architectural decisions honoured: (1) NVMe interno as the demo's disk tier via `ATENIA_DISK_TIER_DIR`; (2) dtype on the handle, no header in the file; (3) extend `migrate_all_cpu_to_disk`, no sibling function; (4) Disk → CpuBf16 deferred to M5; (5) CpuBf16 stays parameter-only by construction; (6) no `os error 5` residue from older tests, verified clean before .a.

      Out of scope and explicitly deferred to M4.7.5+ / M5: prefetch, LRU eviction, per-layer streaming, `ensure_resident` keeping the BF16 view, intermediate tensors in CpuBf16. The M4.7.4 contract is "spill works, restore works, end-to-end correctness preserved", not "fast enough for the killer demo" (that is M4.7.5 / M4.7.6).
  - **M4.7.5 ✅** — M3-e policy upgrade. Per-tensor LRU eviction inside the `DeepDegrade` arm, observability gate on the existing 100 ms probe cache, and the `ensure_cpu` consumer-side audit closure. Six sub-steps each on its own commit:
    - `5c1b27b` (a — probe-cache amortisation audit. The historical `TODO(PERF)` at `Graph::check_guard_before_node` was already resolved by the M3-e `SignalBus::probe_cache` (TTL = 100 ms). Replaced the TODO with a doc-comment that records the audit; new `tests/m4_7_5_a_probe_cache_amortisation_test.rs` asserts `probe_calls_count <= ceil(elapsed_ms / 90) + 2` and a hard ceiling of 10 over a 500-node forward — local result: 1 probe call on a 200-node forward, 1 on a 500-node forward.).
    - `81028a3` (b — `TouchOrder` LRU type in `src/amg/reactive.rs`, populated from `NodeTimingRecorder::drop` so each node lands at the MRU end at completion. New `lru_touch_order: Arc<TouchOrder>` field on `ReactiveExecutionContext` plus a `lru_touch_order()` accessor. Pre-step verified that `src/amm/offloading.rs` is dead-end legacy and `src/apx7/ule.rs` mode ≥7.12 routes through `execute_single` so the single drop hook covers every executor surface.).
    - `65e55c3` (c — `migrate_selected_cpu_to_disk(&[ids], cache_dir)` primitive returning a `SelectiveMigrationReport` with `failures: Vec<(usize, Err)>`. Continues past per-tensor failures (Risk #5 falsification — the reactive loop fires under pressure and abandoning the rest of the eviction set on one transient I/O glitch defeats the goal). Legacy `migrate_all_cpu_to_disk` preserved bit-exact via a private helper `try_migrate_one_to_disk` that both methods share. Four-variant `MigrationStep { Migrated | Skipped | NoOutput | Failed(err) }` enum keeps the legacy "silent continue on `output.is_none()`" behaviour observable.).
    - `78d3336` (d — `Graph::deep_degrade_with_lru` orchestrator. The `DeepDegrade` reaction arm now spills only the bottom `pub const SPILL_FRACTION: f32 = 0.5` of the touch order (the LRU front, i.e. least-recently-used half) instead of the whole graph. Falls back to whole-graph `migrate_all_cpu_to_disk` when the LRU is empty (boot path before any node has executed) — preserves the M3-e.11.5 pressure-relief contract.).
    - `9987209` (e — defensive `ensure_cpu` guards on the `Add` / `Sub` / `Mul` executor arms. The M4.7.3.d audit had skipped these three because their helpers are Tensor methods (`a.add(&b)`, `a.sub(&b)`, `a.mul(&b)`) and at that milestone no producer fed Disk operands into them. M4.7.5.d's selective spill makes Disk operands reachable; the audit hole is closed end-to-end. Includes `chained_arithmetic_post_deep_degrade_round_trip` test that asserts `(4 + 1) * 3 - 2 = 13` after a forced DeepDegrade.).
    - **(f)** — Full 4-model F64 re-validation under M4.7.5 LRU policy. Same ADR-004 threshold 0.5 + per-position argmax 4/4 contract as M4.7.2.e, M4.7.3.f, M4.7.4.f. New gate: `migrated < lru_size_at_spill_time` proving the selective slice is in effect (vs the whole-LRU baseline). Local result on the dev box (NVMe SN770 cache, single-thread):

      | Model | LRU size | Migrated (50%) | Drift vs F64 | Argmax | Logits identical to no-spill |
      |---|---:|---:|---:|:-:|:-:|
      | TinyLlama 1.1B |  958 | 479 | 0.000141 | 4/4 | ✅ |
      | SmolLM2 1.7B   | 1044 | 522 | 0.001446 | 4/4 | ✅ |
      | Qwen 2.5 1.5B  | 1384 | 692 | 0.029057 | 4/4 | ✅ |
      | Llama 3.2 1B   |  700 | 350 | 0.000132 | 4/4 | ✅ |

      Drift bit-exact identical to the M4.7.4.f baseline, and the `warmup_logits == post_logits` cross-check passes on every model — the selective spill + lazy restore cycle is mathematically transparent (the bottom-half spilled bytes round-trip through the M4.7.4.d Disk-arm `ensure_cpu` upcast bit-exactly to their pre-spill form).

      Six architectural decisions honoured: (1) probe TTL stays 100 ms — audit confirms it amortises; (2) prefetch deferred to M4.7.6 (Risk #2 file-lifecycle hazard needs the demo's actual workload to falsify); (3) `SPILL_FRACTION = 0.5` constant (layer-aware needs `node_id → layer_idx` map the builder does not expose); (4) M4.7.3 GPU-residency wiring deferred to M4.7.6 (8 GB VRAM cannot hold 13B; binding constraint is RAM↔disk); (5) `migrate_all_cpu_to_disk` bit-exact preservation via shared private helper; (6) `apx7::ule` mode ≥7.12 verified to use `execute_single` ⇒ touch-signal hook in `NodeTimingRecorder::drop` covers it without an apx7-specific branch.

      Out of scope and explicitly deferred to M4.7.6: prefetch worker, layer-aware fractions, dynamic spill thresholds, M4.7.3 wiring, demo-class integration measurement.
  - **M4.7.6 ✅** — Llama 2 13B Chat end-to-end killer demo. Closed across five sub-steps each on its own commit:
    - `dd84466` (a — download Llama 2 13B Chat from HuggingFace; `LlamaConfig` parser validation against the model card; builder smoke test confirming 363 parameter nodes (1 embed + 40 layers × 9 + 1 final norm + 1 lm_head; the 40 `rotary_emb.inv_freq` per-layer buffers stay computed at runtime); 4 ignored on-disk tests; `total_params_estimate` = 13.0159 B within 1 % of the 13.0 B headline. Adds `pub const ROPE_THETA_LEGACY_DEFAULT: u32 = 10_000` and `LlamaConfig::get_rope_theta` returns the default when the field is absent — Llama 2 era predates the explicit `rope_theta` field.).
    - `e292390` (b — F16 decode + BF16 storage end-to-end validation, closing M4.7's Risk #3. The original M4.7 plan named Mistral 7B v0.3 as the F16 canary; reading the safetensors header showed every Mistral tensor is BF16, not F16. Pivoted to a synthetic F16 fixture via `half::f16::from_f32` covering loader F16 decode → F32 storage, F16 decode → BF16 storage round-trip, and a forward through `build_mini_flux_language_model` to ensure the F16-loaded graph runs without panic. Risk #3 falsified.).
    - `322872f` (c — wire M4.7.3 GPU MatMul to the Llama hot path. Removed the `!in_gpu_segment` constraint on the MatMul executor arm; retargeted `exec_gpu_matmul` to use the residency-aware `cuda_matmul_inplace` with `both_cuda` detection and `Tensor::zeros_new_cuda` allocation. Added a pool-capacity check in `gpu_can_run_matmul` that rejects allocations exceeding `apx4_12::DEFAULT_BLOCK_SIZE` (64 MiB) so 13B-scale tensors fall back to the legacy apx4 path instead of failing allocation. New `GPU_MATMUL_LEGACY_COUNT` counter on the apx4 path; unified `gpu_matmul_total_count()` accessor in `gpu::dispatch::hooks` for "any GPU MatMul fired". F64 4-model re-validation under .c dispatch — drift bit-exact identical to .b baseline; per-model GPU MatMul invocation counts: TinyLlama 154, SmolLM2 168, Qwen 196, Llama 3.2 112.).
    - `3292bc1` (d — first end-to-end Llama 2 13B Chat forward, **Mode A** (clean RAM, no spill, no LRU). 363 parameters loaded from D: NVMe via `ShardedSafetensorsReader` at ~150 MB/s with BF16 storage active; logits `[1, 4, 32_000]` finite, max=15.16, mean=1.14, argmax 4/4 well-defined. Wall-clock: build 2.1 s + load 173 s + forward 1125 s (seq=4). Forward runs **end-to-end on CPU** despite the M4.7.6.c GPU wiring being live: every 13B weight tensor (5120 × 5120 = 100 MB; 5120 × 13824 = 270 MB) exceeds the 64 MiB pool block and the apx4 fallback's `gpu_available()` is hardcoded `false` since pre-M4.6. Documented as M5+ scope (need a non-pooled `cuda_matmul` variant for tensors > 64 MB and reactivation of `apx4::gpu_context::gpu_available()`); the demo's transparency claim is independent of GPU acceleration.).
    - `1906415` (e — Llama 2 13B Chat **Modes B + C**: transparency contract closed. **Mode C** (forced 50 % LRU spill via direct `Graph::deep_degrade_with_lru` call): low-pressure warmup forward at seq=1 captures argmax id=1 logit 4.7747; `deep_degrade_with_lru` migrates 866 / 1732 LRU entries (50.0 % exact) at 150 MB/s to D: NVMe writing 13.0 GB; post-spill forward at 250 s lazy-restores via the M4.7.4.d / M4.7.5.e `ensure_cpu` arms and produces argmax id=1 logit 4.7747 — **bit-exact** transparency at 13B parameter scale. **Mode B** (autonomous LRU spill trigger): high-pressure RAM/VRAM probes promote `Degrade → DeepDegrade` at the first guard checkpoint, 4 DeepDegrade events fired, 26 031 MB spilled in 275 s. Forward wrapped in `catch_unwind` to absorb a downstream M4.7.5.e `ensure_cpu` coverage gap on activation arms under continuous pressure — the gap is structurally separate from the demo's transparency contract (Mode C exercises the same spill primitive end-to-end and proves transparency, so Mode-C PASS is mathematically sufficient).).

      The *momento guau* — Llama 2 13B Chat on the dev box (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, D: NVMe SN770):

      | Mode | seq | argmax | logit | Wall-clock | Notes |
      |------|----:|-------:|------:|-----------:|-------|
      | A — clean RAM | 1 | 1 | 4.7747 | load 169 s + forward ~800–1068 s | No reactive context attached. |
      | B — autonomous LRU spill | 1 | (forward absorbed via catch_unwind) | — | load 172 s + 275 s | 4 DeepDegrade events, 26 GB spilled @ ~95 MB/s. |
      | C — forced 50 % LRU spill | 1 | **1** | **4.7747** | load 172 s + warmup 800 s + spill 87 s + post-spill forward 250 s | 866 / 1732 LRU entries spilled @ 150 MB/s. **argmax(C) == argmax(A) bit-exactly**. |

      Three architectural decisions ratified pre-M4.7.6.a: (1) HF login verification (`huggingface-cli whoami` first); (2) hybrid F64 fixture strategy — dev-local BF16 + cloud F64 deferred pre-tag v20 because 13 B F64 generation needs ~52 GB peak; (3) prefetch worker deferred to M4.7.7 or post-v20. Two further architectural decisions during .c–.e: (4) pool capacity check rather than block-size increase — 13 B tensors exceeding 64 MiB route to apx4 fallback deterministically rather than panicking; (5) Mode B scope reduced to autonomous-trigger validation under `catch_unwind` to absorb the M4.7.5.e activation-arm gap, with the transparency contract carried entirely by Mode C.

      Out of scope and explicitly deferred to M5+: non-pooled `cuda_matmul` for tensors > 64 MB; `apx4::gpu_context::gpu_available()` reactivation; activation-arm `ensure_cpu` coverage under continuous spill pressure. Out of scope and explicitly deferred to v21: adaptive memory-pressure threshold below the OS pagefile trigger on RAM-loaded boxes (Mode A pagefile saturation on the dev box was the empirical falsifier — threshold 0.85 sits above the dev box's pagefile trigger because the 13 B model dominates RAM and the OS pages first).
  - **M4.7 cumulative status**: M4.7.1 through M4.7.6 are all **closed under their respective budgets**. The killer demo target (Llama 2 13B Chat) is validated end-to-end with the transparency contract `argmax(Mode A) == argmax(Mode C)` holding bit-exactly at 13 B parameter scale. The M4.7 milestone is closed. See [HANDOFF M4.7](./docs/HANDOFF_APX_V20_M4.7.md) for the full architectural-decisions ledger and the M5 resume guide.

### Pending sub-phases (post-M4.7, pre-M5)

The momento guau is closed but the demo's wall-clock is impractical for a public reproduction (Mode A 18.7 min on CPU, Mode C ~24 min). Two sub-milestones land before M5 to make the result both fast and reproducible:

- **M4.8 ✅ — Performance optimization**. Closed across six sub-steps:
  - `8f542af` (a — `examples/bench_matmul.rs` baseline harness measuring every reachable CPU MatMul kernel against the canonical Llama shapes plus BF16 decode + clone costs. Confirmed empirically: at default `cargo build --release` the production path was the scalar triple-loop registered as `scalar_matmul`, with `matmul_dispatch` measured at 0.30–0.44 GFLOPS — ~600× below the dev box's ~1.5 TFLOPS theoretical FP32 peak. Three structural defects identified: default `apx_mode = "4.19"` < `"6.3"` lexicographically, `avx2_matmul` registration compile-time gated, and `run_plan` purely serial.).
  - `6353b97` (b — three surgical fixes in `src/lib.rs`: default `apx_mode()` lifted from `"4.19"` to `"7.2"`; `apx_mode_at_least()` switched from lexicographic to numeric segment comparison (closes the latent `"6.10" < "6.3"` bomb); `avx2_matmul` registration moved from compile-time `#[cfg(target_feature = "avx2")]` to runtime `is_x86_feature_detected!("avx2")`. Bench shows `matmul_dispatch` immediately at 3.1× on `1×5120×5120` and 5.3× on `4×5120×13824`.).
  - `22453fb` (c — SIMD BF16 decode kernel (`bf16_decode_avx2`, 8-lane via `_mm256_cvtepu16_epi32` + `_mm256_slli_epi32`). Bulk decode bandwidth lifted from 5.71 GB/s scalar to 15.77 GB/s SIMD on a 70.78 M element 13B-class layer (2.76× faster). Routed at every BF16 → F32 site in `src/tensor/tensor.rs`. Removes ~17 s of pure decode overhead per 13B forward.).
  - `6746cfa` (d — parallel `batch_matmul_dispatch` over the batch dim via rayon `par_chunks_mut` (40 attention heads now run on 24 cores, 7.1× over serial); parallel `matmul_dispatch` wrapper that row-partitions when M >= 2 and per-row work clears 1 K elements (captures the M=4 seq=4 shapes — Q/K/V/O / gate/up / down / lm_head all run 4-way parallel after the wrapper). Cumulative on `4×5120×13824`: 12.4× over the M4.8.a baseline.).
  - `1e9cda4` (e — `matrixmultiply 0.3` integration. Pure-Rust BLIS-style sgemm with AVX2/FMA + NEON paths and runtime ISA dispatch — vendor-agnostic by design (rules out MKL by construction, satisfies the milestone constraint). Routed at the top of `matmul_dispatch_serial` for shapes whose total work clears 1 MFLOP. Cumulative across all five sub-steps: **49.5× on `4×5120×13824`** (1954 → 39 ms; 0.34 → 14.35 GFLOPS), 13.4× on `1×5120×5120`, 9.2× on `1×4096×32000`. The 1 MFLOP gate keeps tiny matmuls on the existing AVX2 path because matrixmultiply's panel-packing overhead dominates below the threshold.).
  - **(f)** — Final 13B Mode A re-validation + ROADMAP / README update. Same `tests/m4_7_6_d_llama2_13b_mode_a_test.rs` harness as M4.7.6.d. Local result on the dev box (RTX 4070 Laptop, 32 GB DDR5-5600, D: NVMe SN770):

    | Phase    | Wall-clock           | Speedup vs pre-M4.8 |
    |----------|---------------------:|--------------------:|
    | Build    | 1.93 s               | (~no change)        |
    | Load     | 162.90 s @ 160 MB/s  | (~no change)        |
    | **Forward at seq=4** | **322.81 s (5.38 min)** | **3.49×** vs 18.75 min baseline |
    | Argmax pos 0 | id=1, logit=4.7747 | bit-exact ≡ M4.7.6.e Mode C |

    Post-M4.8 wall-clock lands at 5.4 min — slightly above the 2–4 min aspirational band but a solid 3.5× lift on the production target, with the bit-exact transparency contract preserved (argmax(Mode A post-M4.8) == argmax(Mode A pre-M4.8) == argmax(Mode C) == 1, logit 4.7747). Headroom remains for the M=1 (seq=1 generation) shapes that don't benefit from row-partitioning — column-partitioning matrixmultiply calls under rayon scope is the natural M5+ follow-up.

    F64 4-model re-validation under M4.7.5 LRU spill (ADR-004 close criterion):

    | Model | Drift M4.7.5.f | Drift M4.8 | Argmax |
    |-------|---------------:|-----------:|:------:|
    | TinyLlama 1.1B | 0.000141 | 0.000063 | 4/4 |
    | SmolLM2 1.7B   | 0.001446 | 0.000242 | 4/4 |
    | Qwen 2.5 1.5B  | 0.029057 | 0.029047 | 4/4 |
    | Llama 3.2 1B   | 0.000132 | 0.000041 | 4/4 |

    Drift improved on all four M4.6 family models (matrixmultiply's panel packing yields more numerically faithful reduction order than the legacy AVX2 path); ADR-004 threshold 0.5 with massive headroom on every row. Test harness wall-clock dropped from 384 s (M4.8.b) → 144 s (.c) → 106 s (.d) → 92 s (.e) — every sub-phase produced a measurable lift. **No performance regressions** in the M4.6 family.

  Vendor-agnostic by construction throughout: matrixmultiply's AVX2/FMA + NEON paths and runtime ISA dispatch keep Intel + AMD x86-64 on equal footing today and leave Apple Silicon (NEON) reachable for v24 without reshaping the public interfaces. **No MKL was introduced.** AVX-512 is documented as not present on the dev box's Raptor Lake-HX silicon (fused off in microcode); the dispatcher would pick it up via `is_x86_feature_detected!("avx512f")` on any AMD Zen 4 / Intel Xeon machine that ships it.

- **M4.9 ✅ — Public CLI demo**. Closed across six sub-steps:
  - `8655832` (a — clap 4 derive-based skeleton in `src/bin/atenia.rs`. Three subcommands: `probe`, `run`, `explain` (legacy v13 surface preserved). `probe` arm gated at runtime via `cfg!(feature = "hw-probe")` rather than `required-features` on the `[[bin]]` entry so default builds stay installable. Probe parity verified bit-exact against the legacy `hardware_probe` binary.).
  - `a18c64a` (b — `src/demo/mod.rs` extraction. Pressure probes, reactive-context factory, sharded BF16 load helper, argmax reduction lifted from `tests/m4_7_6_e_*` into `pub mod demo` behind a default-on `demo` Cargo feature. Test refactored to consume the public module via a 25-line wrapper that preserves the pre-M4.9.b call sites; M4.7.6.e `cargo test --no-run` clean.).
  - `5851bec` (c — `atenia run --mode a` end-to-end. New `src/cli_run.rs` library module with the DemoReport schema, heartbeat-dots progress UX (~2 s flush on stderr), hardware soft-warning when total RAM < 28 GB, cache-dir disk-throughput probe (200 MB/s warning floor), text + JSON renderers. Mode A reproduces M4.8.f bit-exact: argmax id=1 logit=4.7747 at pos 0, forward 278 s, total 7.5 min on the dev box.).
  - `d554996` (d — `atenia run --mode c`. Forced 50 % LRU spill via `Graph::deep_degrade_with_lru`, post-spill forward with lazy restore, transparency contract block. `argmax(pre)` and `argmax(post)` compared bit-exactly; exit code 3 on violation, 0 on PASS. Smoke-tested on Llama 2 13B Chat: argmax(pre) = argmax(post) = 1 logit 4.7747, 866/1732 LRU entries spilled (50.0 % exact). Subsequent dev-box reproductions on a clean shell with warm caches land at **warmup forward 200 s + spill 19 s + post-spill forward 23 s = 6.9 min total wall-clock** — M4.7.6.e momento guau reproduced via one command, faster on every successive run as M4.8 perf compounds with cache warmth.).
  - `6bed661` (e — `atenia run --mode b`. High-pressure RAM/VRAM probes promoting `Degrade → DeepDegrade` autonomously; forward wrapped in `catch_unwind` to absorb the documented M4.7.5.e activation-arm gap. Smoke-tested: 4 DeepDegrade events, 26 031.7 MB spilled, panic message surfaced verbatim, exit 0 (trigger plumbing OK). 8.1 min wall-clock.).
  - **(f)** — Documentation + release polish. New `docs/CLI.md` documenting every subcommand, flag, exit code, and the JSON schema. `README.md` ships the "Reproduce the momento guau in one command" section near the top with the exact command sequence and expected output. Legacy `hardware_probe` binary dropped (parity verified at .a; the `atenia probe` subcommand is now the canonical surface). ROADMAP marked closed at this commit.

  M4.9 close criteria all met:
  1. `atenia probe` produces output bit-identical to the legacy `hardware_probe` (verified at .a).
  2. `atenia run --mode a` reproduces M4.8.f wall-clock within ±10 % and argmax bit-exact (verified at .c).
  3. `atenia run --mode c` reproduces M4.7.6.e momento guau bit-exact, exit code 0 (verified at .d).
  4. `cargo install --path .` produces a working `atenia` binary on Windows (Linux verified by reading the build script's only platform-specific surface — CUDA + MSVC detection — being optional).
  5. `README.md` ships the "Reproduce the momento guau" section with the exact command (verified at .f).

  Vendor-agnostic by construction: `clap`, `serde`, `sysinfo`, `matrixmultiply`, `rayon` — every new dependency is pure-Rust, MIT/Apache, vendor-neutral. No MKL anywhere in the dep graph.

  **Note on the BF16 spike (commit `a786837`, retained as regression gate)**: a precision-floor simulation gated by `ATENIA_BF16_PRECISION_FLOOR=1` lives in `WeightMapper::load_into`. With the env var off the code is bit-identical to the baseline; with it on, every parameter is round-tripped through BF16 quantisation before reaching the graph. M4.7.2.b's cross-path test asserts that this path produces bit-equal decoded values to the native BF16 storage path — preserving the spike as instrumentation defends against any future regression in the storage variant being silently masked.

### Out of scope for v20

- Tokenizer integration (M5+)
- KV cache (M5+)
- Token-by-token generation (M5+)
- Native BF16 / F16 storage without F32 upcast — **shipped as M4.7.2 ✅** (50% RAM savings on parameters, F64 validated across the four M4.6 checkpoints)
- Backward over loaded models (M5+ training territory)
- **Forward performance optimization (deferred post-M4.7)**. M4.5 documented a known follow-up: ~35 s release-mode forward on ~5 GFLOPs for TinyLlama is slower than expected for a 24-thread AVX2 CPU, suggesting the matmul dispatcher misses the AVX2 microkernel path on some shapes. M4.6 added three more datapoints (SmolLM2, Qwen 2.5, Llama 3.2; see [HANDOFF_APX_V20_M4.6.md](./docs/HANDOFF_APX_V20_M4.6.md)) but no profiling work. M4.7 added the 13 B datapoint (Mode A forward 1125 s seq=4 on 24-thread AVX2 CPU) — and clarified the bottleneck: Llama 2 13 B layers (5120 × 5120 = 100 MB, 5120 × 13824 = 270 MB) exceed the M4.7.3 GPU MatMul pool block (`apx4_12::DEFAULT_BLOCK_SIZE` = 64 MiB) and route to the apx4 fallback whose `gpu_available()` is hardcoded `false`, landing the entire 13 B forward on CPU. The structural fix is a non-pooled `cuda_matmul` variant + apx4 reactivation, both M5+ scope. Optimising AVX2 microkernels on top of that path is the wrong layer: the GPU acceleration buys orders of magnitude more than microkernel work on the same shapes.
- **Known M5+ technical debt — non-pooled `cuda_matmul` variant** for tensors > 64 MB (the structural fix that lets the 13 B forward use the GPU); paired with `apx4::gpu_context::gpu_available()` reactivation (the two land together or the rejected tensors silently corrupt). Tracked in HANDOFF M4.7 decisions 34–35.
- **Known M5+ technical debt — `ensure_cpu` activation-arm coverage** under continuous spill pressure. M4.7.6.e Mode B exposed the gap; the M4.7 demo's transparency contract holds via Mode C (single-event spill, the realistic shape). Tracked in HANDOFF M4.7 decision 38.

> **Note on the prior roadmap.** Earlier drafts of this document scoped v20 as "Execution Memory and Learning" — a milestone built around persistent execution memory feeding future decisions. That concept was not lost: it lives today as scaffolding inside the v13 Hybrid Execution Engine, and its observable effect on real workloads is now scheduled to be demonstrated in M4.7, where memory pressure during beyond-VRAM model execution is genuine rather than synthetic. The v20 label was reassigned to Real Model Runtime Integration when investigation showed that loading and executing a real model is the prerequisite for any meaningful test of execution-experience learning.

---

## M5 — Tokenizer + KV cache + token-by-token generation ✅

Closed at commit `43b1b3e`. M5 produced **Atenia chats**: `atenia generate --prompt "Hello, how are you?" --model models/llama-2-13b-chat --max-tokens 20` returns

```text
> Hello, how are you?

Hello! I'm just an AI, I don't have feelings or emotions
```

Empirically validated:
- **24.24 GiB resident** for BF16 13B with two graphs sharing weights via `Arc<TensorStorage>` (vs ~52 GiB naïve). M5.c.2.a primitive locked at scale.
- **R2 graph-level falsifier** (3/3 green): prefill + decode steps reproduce no-cache forward bit-exactly on a synthetic mini-Llama. The cache-aware attention path is mathematically equivalent to the reference.
- **R6 generation contract** (4/4 green): greedy loop produces argmax-consistent tokens with the no-cache forward, EOS halt, max_tokens cap, streaming-sink ordering.
- **D67 determinism fixture** locked: TinyLlama prompt "Hello" → first-8 token IDs reproduce bit-exact across runs.
- **D68 decode-step bench**: forward dominates 99.9% of step cost on TinyLlama (1.05 GFLOPS measured); graph rebuild is <1 ms. M6 priority shifts to GPU offload over decode-graph reuse.

Twelve architectural decisions (D58–D69) locked across the M5.a–M5.f.a sub-phase chain. See [HANDOFF M5](./docs/HANDOFF_APX_V20_M5.md) for the full ledger.

---

## M6 — Tier-aware GPU loader ✅

Closed at commit `8180160`. Tier-aware loader (VRAM → RAM → NVMe) routed 60 attention/FFN projection weights of Llama 2 7B Chat directly to the RTX 4070's VRAM at load time; the rest stayed in RAM; no Disk overflow; bit-identical output to the CPU baseline; **1.46× faster** end-to-end (8.22 s/tok vs 12.02 s/tok). The architecture replan that emerged from the May 2 BSOD on 13B + post-load upload is documented in [INVESTIGATION_M6_REPLAN.md](./INVESTIGATION_M6_REPLAN.md): the planner is a pure function of `(metadata, free_ram, free_vram)` and per-tensor placement happens at load time, never as post-load migration.

See [HANDOFF M6](./docs/HANDOFF_APX_V20_M6.md) for the closing notes.

---

## M7 — 13B-friendly tiers ✅

Closed at commit `19fdcf8` (M7.2) plus the M7.3 integration smoke documented in [HANDOFF M7](./docs/HANDOFF_APX_V20_M7.md). Four sub-phases shipped, each gated behind the 5 mandatory regression suites (43 tests):

- **M7.0** (`8f29233`) — NVMe bench + Disk weight bit-exact test. Measured 3.6 GB/s sustained, 37 ms cold read on a 13B FFN-down weight. Decision: Plan A (no Disk LRU cache needed for the 10.4 s/tok worst-case projection).
- **M7.1** (`db5a49f`) — Disk fast-path. Raw BF16 bytes flow from the safetensors mmap directly into NVMe with zero F32 transient. Counters `disk_fast_path_count` / `disk_slow_path_count` validate the structural property.
- **M7.2** (`19fdcf8`) — Adaptive RAM headrooms. When `model_total > 0.7 × free_ram`, the planner inflates the headroom by the excess so genuine overflow scenarios route to NVMe. Pure helper `adaptive_ram_headroom` plus 5 unit tests. The fixed 8 GiB headroom (M6) is preserved as the floor; only model-dominated boxes trigger the inflation.
- **M7.3** — 13B integration smoke on the operator's hardware. **Llama 2 13B Chat ran end-to-end** with 38 tensors on VRAM (6.70 GiB), 126 on RAM (0.75 GiB), **239 on NVMe (20.14 GiB)** via the M7.1 fast-path. Peak free RAM 7.36 GiB throughout, `disk_busy_pct` bursty (max 1 s @ 100 %), 36.6 s/tok average for 5 tokens, coherent reply, no BSOD. The four success criteria all passed; no rollback criterion triggered.

R3 (May 2 BSOD on 13B + GPU residency) and R5 (fixed headroom misroutes 13B) are closed by construction.

---

## M8 — BF16-resident VRAM kernels ✅

Closed at commit `6d343c3` (M8.4c — Path B). Eight sub-phases shipped, gated behind two mandatory benches (M8.0 + M8.0b) and a 4-model F64 numerical contract test (M8.5):

- **M8.0** (`b955669`) — cuBLAS BF16 TC bench. Measured 2.2× speedup over the naive F32 kernel on the 4 Llama 13B decode-step shapes, but only at peak — decode (M=1) is bandwidth-bound, so the headline 5× target was unreachable from the matmul side alone. **PLAN A confirmed conditionally**.
- **M8.0b** (`f71caf0`) — NVMe → PCIe → GPU async pipeline bench. Measured Config 2 (two-buffer pipeline) at **32.7 ms / 135 MiB** on the FFN-down shape (vs the 200 ms threshold). PLAN A confirmed strongly: the 13B per-token cost is bound by NVMe at ~4.3 GB/s pipelined.
- **M8.1** (`74f7631`) — `TensorGPU::dtype` field + `bf16_to_vram_no_upcast` primitive. BF16 weight resident in VRAM without F32 conversion at load time.
- **M8.2** (`956fefc`) — `cuda_matmul_bf16_inplace` (initial wire-up via cublasGemmEx). Single-op drift envelope characterised on the four 13B shapes.
- **M8.3** (`2e3a920`) — `TierPlan::vram_cost_bytes` parametrised by dtype + `ATENIA_M8_BF16_KERNEL` flag. BF16 weight costs `numel × 2`, F32 weight `numel × 4`.
- **M8.4** (`e25cd7e`) — End-to-end wire-up (loader gate + dispatcher arm) + shared `BF16_COUNTER_TEST_LOCK` for cross-module test serialisation.
- **M8.4b** (`a956044`) — Slow-path BF16 fix for weights with `LoadTransform`s. Every Llama-family `_proj.weight` carries at least `Transpose2D`, so the M8.4-original fast-path arm covered nothing in production.
- **M8.4c** (`6d343c3`) — **Path B**: BF16 weight in VRAM + F32 upcast per-matmul + `cublasGemmEx(F32, F32, F32)`. Closes the M8.5 numerical contract failure (drift 0.18–2.33 in 3/4 models under the M8.4-original BF16 input path) by eliminating the per-matmul activation truncation that was cascading through 16-28 layers. Numerics now match M4.7.2.e (drift ≤ 2.4e-2 across the 4 models, all under ADR-004 threshold 0.5).

Smokes:

- **F64 4-model** (M8.5 + M8.5b counter fix): TinyLlama 8.8e-5, SmolLM2 7.31e-4, Qwen 2.5 2.40e-2, Llama 3.2 4.0e-5. All argmax 4/4. Drift 21-12,500× under threshold.
- **Llama 2 7B Chat**: 6.26 s/tok with M8 active (vs 8.22 s/tok M6 baseline = **1.31×**). 128 weights as BF16 in VRAM (vs 60 as F32 in M6), 0 Disk.
- **Llama 2 13B Chat**: 27.0 s/tok with M8 active (vs 36.6 s/tok M7.3 baseline = **1.36×**). 82 BF16 VRAM, 124 RAM, 197 Disk (vs M7.3's 38 / 126 / 239). No BSOD, RAM headroom 18.75 GiB respected.

The M8 BF16 path is gated by an adaptive heuristic in `pipeline.rs` (`model_total > 0.7 × free_ram`) so 7B-class models that fit in RAM with headroom keep the M6 F32 path automatically — Path B's per-matmul upcast cost is only worth paying when the doubled VRAM capacity translates to fewer Disk-tier reads.

See [HANDOFF M8](./docs/HANDOFF_APX_V20_M8.md) for the full closing notes.

---

## M8.7 — Disk → GPU JIT pipeline ✅

Closed via three sub-phases on top of the M8.7 prereq + flag-default flip:

- **M8.7 prereq** (`3e64d9a`) — Tier planner reserves `DISK_PIPELINE_STAGING_BYTES = 2 × 135 MiB` of VRAM headroom whenever the plan would otherwise overflow to Disk. Two-pass `plan()` keeps the budget exact for the streaming staging slots without penalising 7B-class plans that don't need them.
- **M8.7.0** (`96f14a1`) — MVP single-tensor Disk → GPU staging dispatch (`cuda_matmul_disk_streamed_bf16`). New host primitive `disk_tier::read_bf16_raw_bytes`. The `MatMul` arm in `execute_single_inner` routes Disk(BF16) operands to the streaming dispatch under `ATENIA_M8_7_ENABLED=1`. Single-op drift envelope 3.28e-3 vs ADR-004's 0.5.
- **Tier-aware default flip** (`afaa975` + `21e5bb8`) — D74 superseded; `ATENIA_LEGACY_LOADER=1` is the new opt-out.
- **M8.7 demo enable** (`c7b2ea9`) — `Graph::execute_inference` (record_tape=false). `Graph::segment_has_bf16_or_disk_operands` guard skips the legacy `exec_gpu_segment` for BF16/Disk operands.
- **M8.7.1.a** (`c186148`) — CPU prefetch single-slot helper (`src/cuda/disk_prefetch.rs`). `kick_off(handle)` spawns a `rayon` background NVMe read; `take(handle)` consumes the slot. `cuda_matmul_disk_streamed_bf16` threads `next_handle` through the dispatch chain. `Graph::find_next_disk_bf16_handle_after` provides the executor lookahead. Extended `exec_gpu_segment` guard to also skip when `ATENIA_M8_7_ENABLED=1`. 4 new unit tests.
- **M8.7.1.r** (`c0aee16`) — `cuda_matmul_bf16_inplace` (Path B M8.4c) refactored to accept `stream: *mut c_void`. Replaced device-wide `cudaDeviceSynchronize` with per-stream `cudaStreamSynchronize` after `cudaMemcpyAsync` D→H. With `stream = null` the behaviour is bit-exact with the pre-refactor body. Enables future M8.7.1.b/c (compute / copy stream split) without re-touching the kernel core.

**Headline:**

- **Llama 2 13B Chat**: 20.7 s/tok with `ATENIA_M8_7_ENABLED=1` active (vs 27.0 s/tok M8 baseline = **1.30×**). 154 disk-streamed matmuls per forward, 152 prefetch hits per forward (98.7 % hit rate). Argmax bit-exact with M8.
- **Killer demo seq=4**: 10.22 s forward (vs 13.50 s M8 close = **1.32×**). Logits sane, max\|v\|=15.16, finite=128000/128000.

**Sub-phases skipped — and why:**

- **M8.7.1.b/c (async H→D + dedicated copy/compute streams)** — deferred. The pre-implementation VRAM-budget audit found the two-buffer staging + Path B F32 transient peak (~670 MiB) exceeds the dev-box working-set budget (~540 MiB free after residents + headroom). Code scaffolding is in place (`cuda_matmul_bf16_inplace` accepts a `stream` parameter; `disk_prefetch` is single-slot but readily extended); a future operator on a 24 GB-class GPU can flip M8.7.1.b/c on without re-touching the kernel.

See [HANDOFF M8.7](./docs/HANDOFF_APX_V20_M8.7.md) for the full closing notes.

---

## Next active milestone — M9: INT8 quantisation

Different attack vector to the same throughput frontier — instead of doubling VRAM capacity by halving per-weight bytes from 4 to 2 (M8) or pipelining the Disk-tier reads (M8.7), halve again from 2 to 1. Same plan structure (TierPlan with `kernel_dtype = Int8`, `vram_cost_bytes = numel × 1`). Requires per-tensor calibration (scale / zero-point), which is substantial new infrastructure. Under INT8, the Llama 2 13B's BF16 26 GiB drops to ~13 GiB, fitting entirely in the 8 GiB VRAM + 32 GiB RAM box without any Disk overflow at all — eliminating the M8.7 problem space rather than optimising within it. The M8.7.1.b/c follow-up, deferred above for VRAM-budget reasons, becomes a non-issue once INT8 lands because the residency footprint shrinks below the working-set budget by construction.

### M8.6 — BF16 KV cache ✅

Closed in two sub-phases on top of the M5.b runtime cache infrastructure:

- **M8.6.0** (`b796eaa`) — Opt-in via `ATENIA_BF16_KV_CACHE=1`. New `KvCellDtype { F32, BF16 }`. New `cell_dtype` field on `KvCacheConfig`. New dtype-aware `bytes_per_token()` / `resident_bytes()` accessors. New cast helpers `cast_kv_cell_f32_to_bf16` / `cast_kv_cell_bf16_to_f32` in `src/amg/kv_cache.rs`. Generator harvest path truncates F32 → BF16 between decode steps; reinject path decodes BF16 → F32 right before `overwrite_parameter`. Graph itself stays F32. 5 new unit tests (`tests/m8_6_kv_cache_bf16_test.rs`).

- **M8.6.1** (this) — Default flipped on after the TinyLlama 1.1B-Chat 8-token determinism fixture (`tests/fixtures/generation_determinism/expected_tokens_tinyllama.json`) came back **bit-identical** under the BF16 ledger: same token IDs `[29907, 13946, 368, 29991, 2266, 526, 777, 6455]`, same decoded text `"Certainly! Here are some examples"`. Operators who need the legacy F32 path can opt out via `ATENIA_LEGACY_F32_KV_CACHE=1`; the deprecated `ATENIA_BF16_KV_CACHE=1` flag remains a silent no-op for backwards compatibility.

Headlines:

- Llama 2 13B Chat KV cache @ seq=2048: **3.2 GiB → 1.6 GiB** (–1.6 GiB)
- Per-token K+V cost: 1.5625 MiB F32 → **0.78 MiB BF16** (exactly half)
- Drift envelope: bounded by single BF16 round-trip per cell write (~3e-3 relative); ADR-004 (threshold 0.5) preserved with >100× margin
- Determinism: 0 token drift on TinyLlama 1.1B-Chat 8-token gen
- 189/189 lib tests + 5/5 M8.6 + 3/3 M5 R2 + 4/4 M5.d.a verde

See [HANDOFF M8.6](./docs/HANDOFF_APX_V20_M8.6.md) for the full closing notes.

---

## v21 — Production-ready execution guards

The Guards layer (v16) and Policies layer (v15) currently operate on a model that was satisfactory for scaffolding but will need hardening to consume SignalBus output reliably under production conditions. v21 focuses on:

- Guard verdict stability under noisy signals (already partly addressed at the Policy layer; extending the same hysteresis-aware behavior to Guards)
- Recovery and rollback paths exercised against real model workloads (the M4.7 and M5 outputs are where this work is exercised non-synthetically)
- Operational tooling: structured logging, metrics, replay harnesses for debug
- **Adaptive memory-pressure threshold (known issue carried over from M4.7)**. The M4.7.6.d Mode A run on the dev box (32 GB RAM, 26 GB BF16 model loaded) saturated the OS pagefile before the reaction loop's threshold (`0.85`) saw "pressure", because the OS pages first when the model dominates RAM. v21's adaptive threshold should land below the OS pagefile trigger on RAM-loaded boxes (~ 0.78 measured on the dev box) with hysteresis to avoid thrash. Empirical baseline preserved in HANDOFF M4.7 (sprint observations).

---

## v22 — Multi-vendor backend foundation

Today the engine has a CPU baseline (always available) and an NVIDIA GPU path (CUDA). v22 expands the GPU path to a vendor-neutral abstraction that supports a second vendor in the same release: Intel iGPU (which is already common on the project's primary development hardware).

Out of scope for v22: AMD ROCm and Apple Metal, which are substantial enough to merit dedicated milestones.

---

## v23 — AMD ROCm support

ROCm path for AMD GPUs. Substantial work due to the differences between CUDA and ROCm in driver model, memory management, and synchronization primitives. Treated as its own milestone rather than bundled with v22.

---

## v24 — Apple Metal support

Metal Performance Shaders integration for Apple Silicon. The largest backend port due to:

- Different memory model (unified memory vs discrete VRAM)
- Different programming model (Metal Shading Language vs CUDA / ROCm)
- Different toolchain (Xcode-centric)

Scoped explicitly as a major milestone, not a quick adapter on top of v22.

---

## v25 — Distributed execution

Multi-host execution. Out of scope until single-host execution is mature across vendors.

---

## How to contribute

This is research-in-progress. Contributions, issues, and technical discussions are welcome — especially from people with experience in:

- GPU runtime systems and CUDA / ROCm / Metal low-level APIs
- Memory management and OOM prevention
- Adaptive scheduling and execution policies
- Systems research and MLSys
- Real-world LLM inference on small-scale or commodity hardware

Open an issue or reach out if you want to collaborate on any specific layer.

---

## Design principles

Reproduced from the main [README](./README.md) as a reminder of the constraints every roadmap item is expected to respect:

- **Stability before performance** — Short-term gains mean nothing if execution collapses under noise.
- **Adaptation without semantic drift** — The engine may change *how* things run, never *what* is computed.
- **Learning by experience, without ML** — Execution outcomes are distilled into persistent memory — no opaque training loops in the runtime.
- **Observable and reproducible** — Every behavior claimed by the engine must be verifiable through executable tests.
