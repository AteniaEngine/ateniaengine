# Atenia Engine — Roadmap

This document describes the priorities guiding Atenia Engine's development. It is organized by upcoming APX version, from in-progress work to broader horizons.

This roadmap communicates scope and priority, not calendar commitments. Versions are released when ready.

---

## Status overview

Atenia Engine is currently working through APX v20 (Real Model Runtime Integration). Earlier versions (v12 through v19) are complete. The most recently closed sub-milestone is M4.5: end-to-end forward execution of `TinyLlama-1.1B-Chat-v1.0` with PyTorch numerical validation.

Detailed closing notes per milestone live in the `docs/` directory:

- [docs/HANDOFF_APX_V20_M3.md](./docs/HANDOFF_APX_V20_M3.md) — reactive execution context, real GPU storage, M3-e migration loop
- [docs/HANDOFF_APX_V20_M4.md](./docs/HANDOFF_APX_V20_M4.md) — safetensors loader and weight mapping mechanics
- [docs/HANDOFF_APX_V20_M4.5.md](./docs/HANDOFF_APX_V20_M4.5.md) — real model execution end-to-end (`TinyLlama-1.1B`)

---

## Current focus: APX v20 — Real Model Runtime Integration

APX v20 connects the completed telemetry and decision infrastructure to real external model execution. The first target was HuggingFace `safetensors` checkpoints. The milestone is structured into sub-phases:

### Completed sub-phases

- **M1** — `Conv2D` and `MaxPool2D` natively in the Adaptive Model Graph, with forward, backward, tape integration, and finite-difference gradient checking.

- **M2** — Reactive execution context attached to the graph; the executor consults guard state before each node and returns typed abort reasons on guard verdicts. Existing APIs preserved as backward-compatible wrappers.

- **M3** — Real GPU allocation for tensors behind a vendor-neutral storage abstraction (`TensorStorage`), real host↔device transfers, and the M3-e reaction loop that moves real VRAM to RAM on guard `Degrade` verdicts.

- **M4** — Model loader mechanics: a `safetensors` reader (header + body, by-name and iterator access), a `WeightMapper` with shape validation and structured `LoadReport` diagnostics, and BF16 / F16 → F32 decode. Validated empirically against a real HuggingFace gpt2 checkpoint.

- **M4.5** — End-to-end real model execution. The engine loads a HuggingFace `TinyLlama-1.1B-Chat-v1.0` checkpoint and runs forward on CPU. Logits match a PyTorch reference within F32-vs-BF16 precision drift over 22 transformer blocks (max absolute diff ≈ 0.73, mean ≈ 0.06, no values diverging by more than 1.0). New graph primitives landed for this: rotary positional embedding, general permute, broadcast multiplication, and rank-4 batched matmul. A complete Llama-2 graph builder consumes the HuggingFace parameter naming convention directly.

### Pending sub-phases

- **M4.6** — Llama-family compatibility expansion: Llama 3.2 (with `rope_scaling: "llama3"`), Qwen 2.5 (with QK-Norm), Phi 3.5 mini, SmolLM3, Mistral 7B (subject to host RAM). Most architectural deltas are small per family; the work is in extending the config parser, the weight mapper, and the builder for each family's specific operations.

- **M4.6.1** — Mathematical ground-truth validation. Extend the per-model PyTorch comparison tests with NumPy F64 reference forwards (full models) and optionally mpmath arbitrary-precision reference (isolated components). Resolves the implicit "PyTorch as ground truth" framing in M4.5-d.1 by measuring drift against mathematical truth directly. See [ADR-002](./docs/decisions/ADR-002-mathematical-ground-truth-validation.md).

- **M4.7** — Beyond-VRAM execution. Run a 13B-class model in BF16 on notebook hardware (8 GB VRAM, 16 GB RAM, SSD). The first end-to-end exercise of the M3 reaction loop against a workload that genuinely exceeds available VRAM. Validates the project's core differential: the ability to execute models on hardware that other engines cannot accommodate.

### Out of scope for v20

- Tokenizer integration (M5+)
- KV cache (M5+)
- Token-by-token generation (M5+)
- Native BF16 / F16 storage without F32 upcast (separate optimization milestone)
- Multi-file (sharded) safetensors (extension when needed for larger models)
- Forward optimization at the matmul dispatcher level (post-M5 follow-up)
- Backward over loaded models (M5+ training territory)

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
