# Atenia Engine — Roadmap

This document describes the priorities guiding Atenia Engine's development. It is organized by upcoming APX version, from in-progress work to broader horizons.

This roadmap communicates scope and priority, not calendar commitments. Versions are released when ready.

---

## Status overview

Atenia Engine is currently working through APX v20 (Real Model Runtime Integration). Earlier versions (v12 through v19) are complete. The most recently closed sub-milestone is M4.6: Llama-family compatibility expansion, with four production checkpoints (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B) executing end-to-end and validated against PyTorch F64 mathematical ground truth per ADR-004.

Detailed closing notes per milestone live in the `docs/` directory:

- [docs/HANDOFF_APX_V20_M3.md](./docs/HANDOFF_APX_V20_M3.md) — reactive execution context, real GPU storage, M3-e migration loop
- [docs/HANDOFF_APX_V20_M4.md](./docs/HANDOFF_APX_V20_M4.md) — safetensors loader and weight mapping mechanics
- [docs/HANDOFF_APX_V20_M4.5.md](./docs/HANDOFF_APX_V20_M4.5.md) — real model execution end-to-end (`TinyLlama-1.1B`)
- [docs/HANDOFF_APX_V20_M4.6.md](./docs/HANDOFF_APX_V20_M4.6.md) — Llama-family compatibility expansion (four checkpoints, F64 validation methodology)

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

- **M4.6.2** (deferred until after M4.7 — priority, not feasibility) — Phi 3.5 mini Instruct (3.8B). The architectural deltas vs the Llama family are identified and tractable: a `RopeScaling::Longrope` variant with per-dim `short_factor` / `long_factor` vectors and the `attention_factor` post-multiply on cos/sin (a step llama3 does not need); a fused `qkv_proj` and a fused `gate_up_proj` that need to be split at load time; everything else (RmsNorm, SwiGLU, MHA, half-split RoPE) reuses existing primitives. Estimated ~9 calibrated hours of work — slightly above Phase C. Technically viable on the dev hardware (32 GB RAM accommodates the ~15 GB F32 weights + safetensors buffer + activations) but explicitly deferred on the grounds that **M4.7 is strictly higher impact** — the killer demo is the v20 thesis under genuine memory pressure, not a fifth Llama-family checkpoint. Phi 3.5 mini lands after the *momento guau*. See the M4.6.2 investigation notes for the full architectural diff.

- **M4.7** — Beyond-VRAM execution. The killer demo for the v20 thesis: run a 13B-class model in BF16 on the dev hardware end-to-end. Concrete target hardware: **RTX 4070 Laptop with 8 GB VRAM, 32 GB RAM, project root on an external USB SSD (drive F:)**. A 13B model in BF16 weighs roughly 26 GB on disk — does not fit in VRAM (8 GB), does not fit in RAM alone (32 GB minus working set leaves no room for activations + KV cache), but is executable end-to-end with VRAM ↔ RAM ↔ disk offload orchestrated by the M3 reaction loop. This is the first end-to-end exercise of that loop against a workload that genuinely exceeds VRAM, and the first time the project's core differential — *adapt execution to hardware reality, not the other way around* — is exercised on a real model rather than synthetic memory-pressure injection. Mistral 7B falls out of scope for free once M4.7 lands: its architecture is identical to Llama 2, blocked today only on memory tiering.

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
  - **M4.7.4** — RAM↔SSD streaming primitive (mmap, chunked pull). ~30h.
  - **M4.7.5** — M3-e policy upgrade (LRU per-tensor selection, probe cache, prefetch, ensure_cpu safety net on consumers). ~40h.
  - **M4.7.6** — First end-to-end run on Llama 2 13B (or Mistral 7B v0.3 fallback) + F64 validation per ADR-004. ~30h.
  - **M4.7 cumulative status**: M4.7.1, M4.7.2 and M4.7.3 are all closed under their respective budgets. The remaining three sub-phases (M4.7.4 through M4.7.6) carry an aggregate complexity reference of ~100h; the calendar duration for those will be set by the same kind of execution acceleration the first three sub-phases have already demonstrated.

  **Note on the BF16 spike (commit `a786837`, retained as regression gate)**: a precision-floor simulation gated by `ATENIA_BF16_PRECISION_FLOOR=1` lives in `WeightMapper::load_into`. With the env var off the code is bit-identical to the baseline; with it on, every parameter is round-tripped through BF16 quantisation before reaching the graph. M4.7.2.b's cross-path test asserts that this path produces bit-equal decoded values to the native BF16 storage path — preserving the spike as instrumentation defends against any future regression in the storage variant being silently masked.

### Out of scope for v20

- Tokenizer integration (M5+)
- KV cache (M5+)
- Token-by-token generation (M5+)
- Native BF16 / F16 storage without F32 upcast — **shipped as M4.7.2 ✅** (50% RAM savings on parameters, F64 validated across the four M4.6 checkpoints)
- Backward over loaded models (M5+ training territory)
- **Forward performance optimization (deferred post-M4.7)**. M4.5 documented a known follow-up: ~35 s release-mode forward on ~5 GFLOPs for TinyLlama is slower than expected for a 24-thread AVX2 CPU, suggesting the matmul dispatcher misses the AVX2 microkernel path on some shapes. M4.6 added three more datapoints (SmolLM2, Qwen 2.5, Llama 3.2; see [HANDOFF_APX_V20_M4.6.md](./docs/HANDOFF_APX_V20_M4.6.md)) but no profiling work. The performance optimization milestone should be scoped *after* M4.7 numbers are available, not before — beyond-VRAM execution will produce the first empirical data on whether the actual bottleneck is compute, memory bandwidth, or tier transition latency. Optimising on the M4.6 baseline risks chasing the wrong bottleneck before the killer-demo workload exposes what matters. The principle "make it work, make it right, make it fast" applies in order: M4.5 closed *work*; M4.6 closed *right*; *fast* follows M4.7.

> **Note on the prior roadmap.** Earlier drafts of this document scoped v20 as "Execution Memory and Learning" — a milestone built around persistent execution memory feeding future decisions. That concept was not lost: it lives today as scaffolding inside the v13 Hybrid Execution Engine, and its observable effect on real workloads is now scheduled to be demonstrated in M4.7, where memory pressure during beyond-VRAM model execution is genuine rather than synthetic. The v20 label was reassigned to Real Model Runtime Integration when investigation showed that loading and executing a real model is the prerequisite for any meaningful test of execution-experience learning.

---

## v21 — Production-ready execution guards

The Guards layer (v16) and Policies layer (v15) currently operate on a model that was satisfactory for scaffolding but will need hardening to consume SignalBus output reliably under production conditions. v21 focuses on:

- Guard verdict stability under noisy signals (already partly addressed at the Policy layer; extending the same hysteresis-aware behavior to Guards)
- Recovery and rollback paths exercised against real model workloads (the M4.7 and M5 outputs are where this work is exercised non-synthetically)
- Operational tooling: structured logging, metrics, replay harnesses for debug

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
