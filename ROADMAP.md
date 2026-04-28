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

  **Sub-phase plan (calibrated 2.5×):**
  - **M4.7.1** — Sharded safetensors loader (multi-file + `model.safetensors.index.json`, drop-after-decode RAM bound). ~20h.
  - **M4.7.2** — Native BF16 storage layer + decode-on-access. **~40–50h** (recalibrated from the original ~60h estimate after the BF16 precision-floor spike commit `a786837` — see note below).
  - **M4.7.3** — GPU MatMul with resident operands + executor device dispatch. ~40h.
  - **M4.7.4** — RAM↔SSD streaming primitive (mmap, chunked pull). ~30h.
  - **M4.7.5** — M3-e policy upgrade (LRU per-tensor selection, probe cache, prefetch, ensure_cpu safety net on consumers). ~40h.
  - **M4.7.6** — First end-to-end run on Llama 2 13B (or Mistral 7B v0.3 fallback) + F64 validation per ADR-004. ~30h.
  - **Total: ~190–200h calibrated ≈ 5–6 calendar weeks** at sustainable pace. The original roadmap estimate of "5–10 days" was substantially optimistic; this is the realistic scope after end-to-end gap analysis.

  **Note on the BF16 spike (commit `a786837`)**: a precision-floor simulation gated by `ATENIA_BF16_PRECISION_FLOOR=1` is committed to `WeightMapper::load_into`. With the env var off the code is bit-identical to the baseline; with it on, every parameter is round-tripped through BF16 quantisation before reaching the graph, faithfully simulating the precision impact of native BF16 storage without any storage refactor. Empirically on Qwen 2.5 1.5B (the worst-case from the M4.6 family — head_dim=128, scale not power-of-2): max drift vs F64 = 0.029, argmax MATCH 4/4, **17× under the ADR-004 threshold of 0.5** and **53× closer to F64 truth than industry-default PyTorch BF16**. The spike answered the only load-bearing question for M4.7.2 — *does BF16 precision survive a Llama-class forward?* — empirically yes. The remaining work is pure storage plumbing (no precision risk).

### Out of scope for v20

- Tokenizer integration (M5+)
- KV cache (M5+)
- Token-by-token generation (M5+)
- Native BF16 / F16 storage without F32 upcast (separate optimization milestone)
- Multi-file (sharded) safetensors (extension when needed for larger models)
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
