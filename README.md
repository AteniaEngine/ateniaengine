# 🧠 Atenia Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970198.svg)](https://doi.org/10.5281/zenodo.17970198)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**An execution-centric AI runtime system written from scratch in Rust.**

> [!NOTE]
> **Status: Early research in progress.**  
> This project is a working prototype with architectural scaffolding for 
> hardware-adaptive execution intelligence. Several capabilities are now 
> wired to real hardware signals; others remain scaffolding being connected 
> milestone by milestone.  
> See [Current State](#-current-state) for an honest breakdown, and 
> [ROADMAP.md](./ROADMAP.md) for the delivery plan.

---

## 🎯 Vision

Modern AI runtimes assume stable hardware.

**Reality does not.**

GPUs are shared. Memory pressure fluctuates. Schedulers jitter. Execution policies thrash. Most production failures in AI systems are not numerical bugs — they are **decision failures** in the execution layer.

Atenia Engine aims to treat execution as a first-class adaptive system: one that observes runtime signals, reasons about stability and risk, and adapts execution policies without modifying computational semantics.

This repository contains the architectural foundation and the reference implementation under active development.

---

## ⚙️ Execution Is Not Plumbing

In most AI systems, execution is treated as plumbing:

```
launch kernels → move data → hope the hardware behaves
```

Atenia Engine starts from a different premise:

> **Execution makes decisions. Decisions must adapt to reality.**

Execution determines *where*, *when*, and *how* computation runs. Under dynamic conditions, these decisions must be observed, reasoned about, stabilized, and refined over time.

---

## 🧭 Design Principles

These are the principles guiding every design decision in Atenia. Some are fully realized today, others are under active development:

- **🧱 Stability before performance** — Short-term gains mean nothing if execution collapses under noise.
- **🔒 Adaptation without semantic drift** — The engine may change *how* things run, never *what* is computed.
- **🧠 Learning by experience, without ML** — Execution outcomes are distilled into persistent memory — no opaque training loops in the runtime.
- **🔬 Observable and reproducible** — Every behavior claimed by the engine must be verifiable through executable tests.

See [docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md) for the reaction-strategy design that grounds future APX milestones.

---

## 📊 Current State

Atenia Engine is implemented in Rust. The project follows an APX (Adaptive Execution) versioning scheme from v1 to v25, describing the progression from primitives to a fully adaptive runtime.

### ✅ What works today

Real, executable, deterministic:

**Real LLM inference (APX v20 M4.5–M4.7.4)**

- **🤖 Four production LLMs run end-to-end on CPU.** TinyLlama 1.1B Chat, SmolLM2 1.7B Instruct, Qwen 2.5 1.5B Instruct, and Llama 3.2 1B Instruct load from HuggingFace `.safetensors` and produce logits validated against PyTorch F64 ground truth per [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md). Atenia F32 drift sits between **1.32×10⁻⁴ and 1.45×10⁻³** — three-to-four orders of magnitude closer to mathematical truth than industry-default BF16 inference on the same checkpoints. Argmax matches F64 on every position of every model.
- **📦 Sharded safetensors loader.** Multi-file HuggingFace checkpoints (`model-NNNNN-of-NNNNN.safetensors` + `index.json`) load via `ShardedSafetensorsReader` with drop-after-decode. Verified on Mistral 7B v0.3 (14.5 GB across 3 shards); peak RAM stays bounded by the largest single shard, not the sum.
- **💾 Native BF16 parameter storage.** `TensorStorage::CpuBf16(Vec<u16>)` halves the persistent RAM footprint of model parameters (verified at exactly 50.0% on TinyLlama). All four production checkpoints re-validated under BF16 storage active — drift bit-exact identical to the precision-floor spike, all under the ADR-004 threshold.

**Engine foundations**

- **🦀 Tensor engine** — Forward + backward with autograd, CPU + CUDA paths.
- **🧩 AMG (Adaptive Model Graph)** — Graph representation with its own executor, independent of PyTorch/TF.
- **⚡ Fused kernels** — Attention and QKV on CPU and GPU.
- **🖼 Convolutional ops** — `Conv2D` and `MaxPool2D` natively in AMG (forward, backward, tape) (APX v20 M1).
- **🔒 Deterministic execution** — Same input, same output, every time.
- **🏗 Portable build** — `build.rs` auto-detects CUDA Toolkit and MSVC BuildTools; overridable via `CUDA_PATH` and `MSVC_TOOLS_PATH`.

**Adaptive execution scaffolding**

- **📡 Real memory telemetry** — VRAM (via `nvidia-smi`) and RAM (via `sysinfo`), validated against ground-truth readings.
- **🧠 Signal producers** — `FailureCounter` (time-windowed) and `LatencyMonitor` (P50-baseline spike detection) with deterministic purge semantics.
- **🚦 SignalBus** — Assembles real telemetry into `GuardConditions` (v16) and `PolicyEvidenceSnapshot` (v15).
- **⚡ Reactive execution hook** — `ReactiveExecutionContext` wires the SignalBus to the AMG executor; `execute_checked` returns a typed `ExecutionAbortReason` on guard trip (APX v20 M2).
- **📋 Policy registry** — Pure, deterministic, explainable policy layer with evidence-aware evaluation.
- **📜 Execution contracts** — Data structures, validators, replay scaffolding.

### 🟡 Scaffolding still being wired to runtime signals

Architecture in place, full end-to-end wiring in progress:

- **Production guards** — The guard framework (v16) and SignalBus (v19) are real, but built-in `ExecutionGuard` implementations live in tests only. Guards in production code are a v21 deliverable.
- **Execution Policies (v15)** — `DecisionBias` reacts to real SignalBus evidence, but no execution path consumes the resulting bias yet.
- **AMM Forecaster** — Exposes real VRAM and RAM. Integration with memory-manager allocation decisions is pending.
- **Fusion Selector (v6.10/6.11)** — `fused_qkv_us = 0` is a placeholder awaiting paired measurement.
- **`FragmentationWarning`** — Intentionally not emitted (external proxies would be misleading). Deferred until Atenia exposes its own GPU allocator.
- **MNIST pipeline** — Conv2D / MaxPool / Dense run end-to-end on synthetic data; real MNIST dataset + trained weights pending.
- **GPU residency-aware MatMul / BatchMatMul** — shipped as **M4.7.3 ✅**. `cuda_matmul_inplace` / `cuda_batch_matmul` consume `(Cuda, Cuda, Cuda)` triples directly via device pointers; the MatMul and BatchMatMul executor arms allocate VRAM-resident outputs through `Tensor::zeros_new_cuda` when both operands live on VRAM and short-circuit ahead of the kernel-planner target switch. Linear residency stays gated behind the `try_gpu_linear` MiniFlux constraint and is a separate milestone.

### Roadmap (APX v18 → v25)

**Completed:**

- **v12** — Initial learning engine scaffolding (withdrawn paper)
- **v13** — Hybrid Execution Engine (H.E.E.) — adaptive placement scaffolding
- **v14** — Execution timeline + profile infrastructure
- **v15** — Policy layer (5 built-in policies, evidence-driven evaluation)
- **v16** — Execution contracts and guard framework
- **v17** — Kernel normalization and symbolic GPU chain
- **v18** — Memory telemetry foundation:
  - Real VRAM probe via `nvidia-smi`
  - Memory pressure detection (replaces the withdrawn predictive fallback test)
  - System RAM telemetry via `sysinfo`
- **v19** — SignalBus: integrated sensor-to-decision pipeline
  - All 4 `GuardConditions` fields sourced from real telemetry
  - 4 of 5 `PolicySignalKind` variants produced; `FragmentationWarning` deferred
  - `FailureCounter` and `LatencyMonitor` as internal state producers

**In progress — v20 (model runtime integration):**

- **M1–M3 ✅** — Conv2D / MaxPool2D in AMG, reactive executor, real GPU storage with M3-e VRAM→RAM migration. See [HANDOFF M3](./docs/HANDOFF_APX_V20_M3.md).
- **M4 ✅** — Safetensors loader: `SafetensorsReader`, `WeightMapper`, BF16/F16 → F32 decode, validated against gpt2. See [HANDOFF M4](./docs/HANDOFF_APX_V20_M4.md).
- **M4.5 ✅** — TinyLlama 1.1B end-to-end on CPU, PyTorch-bounded numerical drift over 22 layers. See [HANDOFF M4.5](./docs/HANDOFF_APX_V20_M4.5.md).
- **M4.6 ✅** — Llama-family expansion. SmolLM2 1.7B (tied embeddings), Qwen 2.5 1.5B (QKV biases, model_type-aware defaults), Llama 3.2 1B (`rope_scaling: "llama3"` with F64-internal piecewise compute). All four M4.6-scope models F64-validated per ADR-004. See [HANDOFF M4.6](./docs/HANDOFF_APX_V20_M4.6.md).
- **M4.6.1 ✅** — Retroactive F64 validation for TinyLlama (drift 0.000141, ratio 5198× vs BF16).
- **M4.6.2** *(deferred until after M4.7 — priority, not feasibility)* — Phi 3.5 mini. Architectural deltas identified (longrope, fused qkv_proj / gate_up_proj); technically viable but lower impact than M4.7.
- **M4.7** *(in progress)* — Beyond-VRAM execution. Target: RTX 4070 Laptop with **8 GB VRAM, 32 GB RAM, project on USB SSD**. A 13B BF16 model (~26 GB) fits in neither VRAM nor RAM alone but runs end-to-end via VRAM ↔ RAM ↔ disk offload. Six sub-phases:
  - **M4.7.1 ✅** — Sharded safetensors loader (Mistral 7B v0.3, 3 shards, 14.5 GB).
  - **M4.7.2 ✅** — Native BF16 parameter storage. 50% RAM savings, all four M4.6 checkpoints re-validated under BF16 active.
  - **M4.7.3 ✅** — GPU MatMul + BatchMatMul with resident operands and per-storage gating in the executor arms; defensive `ensure_cpu` audit across every CPU-only kernel arm; F64 4-model re-validation under M4.7.3 dispatch (counters added to `gpu::dispatch::hooks` so the validation gates a silent CPU-fallback regression).
  - **M4.7.4 ✅** — RAM ↔ SSD streaming primitive: BF16-aware disk format (`DiskDtype` flag on the handle, files keep the M4.7.2 50 % footprint contract), chunked streaming reader (4 MiB per chunk, no `memmap2` dep), `migrate_all_cpu_to_disk` BF16 arm (was silently skipping every CpuBf16 tensor), `ensure_cpu` Disk arm dispatching on the on-disk dtype. F64 4-model re-validation under disk spill: drift bit-exact identical to the M4.7.3 baseline (TinyLlama 1.41e-4, SmolLM2 1.45e-3, Qwen 2.91e-2, Llama 3.2 1.32e-4), argmax 4/4 on every model. NVMe-backed `ATENIA_DISK_TIER_DIR` documented as the runtime prerequisite for the demo.
  - **M4.7.5** — M3-e policy upgrade (LRU, probe cache, prefetch).
  - **M4.7.6** — First end-to-end run on Llama 2 13B (or Mistral 7B v0.3 fallback) + F64 validation.
- **M5+** — Inference UX: tokenizer, KV cache, token-by-token generation.

> **Forward performance is deferred until after M4.7.** Current release-mode forward times sit between 21 s (Qwen 2.5) and 50 s (SmolLM2) at seq=4 on a 24-thread AVX2 CPU — slower than expected for the GFLOP count, suggesting the matmul dispatcher misses the AVX2 microkernel on some shapes. Optimising before the beyond-VRAM workload runs risks chasing the wrong bottleneck. The principle "make it work, make it right, make it fast" applies in order: M4.5 closed *work*, M4.6 closed *right*, *fast* follows M4.7.

**Later:**

- **v21** — Emergent policy decisions: production guards and policies consume SignalBus output to shape real execution paths
- **v22** — Multi-backend foundation: vendor-neutral abstraction for hardware probes and kernel compilation (NVIDIA + Intel iGPU as first coexistence target, via DXGI on Windows)
- **v23** — ROCm backend (AMD)
- **v24** — Metal backend (Apple Silicon)
- **v25** — Distributed execution, autonomous runtime

Full, current roadmap: [ROADMAP.md](./ROADMAP.md).

---

## 🔬 Running the Code

Atenia Engine compiles with Rust stable (2024 edition or later) and requires no external ML frameworks.

```bash
cargo build --release
cargo test
```

The build script auto-detects CUDA Toolkit and MSVC BuildTools installation paths. If detection fails (CUDA installed in a non-standard location, multiple MSVC versions), override via the `CUDA_PATH` and `MSVC_TOOLS_PATH` environment variables.

### Test coverage

The repository ships **~1100 `#[test]` functions across ~350 test files**, covering:

- Tensor operations and autograd correctness
- Graph construction and execution (CPU + CUDA numerical equivalence where applicable)
- Gradient checking via central finite differences
- F64 numerical validation per [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) on the four production LLMs
- BF16 storage round-trip equivalence with the precision-floor spike (regression gate)
- Sharded safetensors loading on real multi-file checkpoints
- SignalBus producing `GuardConditions` / `PolicyEvidenceSnapshot` from real probes
- AMG executor aborting cleanly on guard verdicts and resuming when conditions are clean

See [docs/TESTS.md](./docs/TESTS.md) and [tests/README.md](./tests/README.md) for methodology and categorisation.

> [!WARNING]
> **Note on test methodology.**  
> Some tests from earlier APX versions use controlled harnesses that 
> inject runtime conditions (memory pressure, policy competition) to 
> exercise the scaffolding. These are being rewritten to derive signals 
> from the engine itself as part of the v18+ roadmap; several were 
> completed in v18 (memory pressure detection) and v19 (SignalBus-driven 
> integration tests).

---

## ❌ What Atenia Engine Is Not

- ❌ Not a machine learning framework
- ❌ Not a compiler or graph optimizer
- ❌ Does not modify model semantics
- ❌ Does not require retraining
- ❌ Does not assume ideal hardware

Atenia is designed to sit **below** ML frameworks and **above** raw hardware execution — addressing a layer they largely ignore: **execution stability**.

---

## 🛠 Implementation

- 🦀 Implemented in **Rust** (2024 edition)
- 🔒 Deterministic execution behavior
- 🧵 Explicit memory and concurrency control
- 🚫 No garbage collection
- 🧩 No opaque runtime adaptation

---

## 📚 Documentation

**Roadmap and architecture**

- [ROADMAP.md](./ROADMAP.md) — Versioned roadmap with milestones and design constraints
- [docs/APX.md](./docs/APX.md) — Per-version APX notes
- [docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md) — Reaction strategies (APX v21+)
- [docs/RESEARCH_INTEL_APIS.md](./docs/RESEARCH_INTEL_APIS.md) — Multi-vendor GPU API research (APX v22+)

**Milestone closing notes (HANDOFFs)**

- [HANDOFF M3](./docs/HANDOFF_APX_V20_M3.md) — Reactive executor + GPU storage + M3-e migration
- [HANDOFF M4](./docs/HANDOFF_APX_V20_M4.md) — Safetensors loader
- [HANDOFF M4.5](./docs/HANDOFF_APX_V20_M4.5.md) — TinyLlama end-to-end
- [HANDOFF M4.6](./docs/HANDOFF_APX_V20_M4.6.md) — Llama-family expansion + F64 validation methodology

**Architectural Decision Records (ADRs)**

- [docs/decisions/](./docs/decisions/) — ADR-001 through ADR-004
- [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) — F64 reference as primary numerical validation methodology

**Testing**

- [docs/TESTS.md](./docs/TESTS.md) — Test methodology and categorisation
- Subversion READMEs: [`src/v13`](./src/v13/README.md), [`src/v14`](./src/v14/README.md), [`src/v15`](./src/v15/README.md), [`src/v16`](./src/v16/README.md), [`src/v17`](./src/v17/README.md)

---

## 🧾 Intellectual Property

- **Patent:** USPTO Provisional Application **63/941,875** (filed December 16, 2025)
- **License:** Apache License 2.0 (with explicit patent grant)
- **Author:** Guillermo Alonso Albella — GAAIA Labs (Independent Research Initiative)

Apache 2.0 allows broad adoption, modification, and commercial use while providing explicit patent protection.

---

## 📄 Research Paper

The initial research preprint has been withdrawn while the implementation matures to fully back its empirical claims.

See [`paper/README.md`](./paper/README.md) for details. A revised version with end-to-end empirical validation will be published once runtime signal integration and real model loading (APX v20+) are complete.

---

## 🤝 Contributing

This is a research-in-progress. Contributions, issues, and technical discussions are welcome — especially from people with experience in:

- GPU runtime systems and CUDA / ROCm low-level APIs
- Memory management and OOM prevention
- Adaptive scheduling and execution policies
- Systems research and MLSys

Open an issue or reach out if you want to collaborate on any specific layer.

---

## 🌐 Links

- 🌍 **Website:** [ateniaengine.com](https://ateniaengine.com)
- 💾 **Repository:** [github.com/AteniaEngine/ateniaengine](https://github.com/AteniaEngine/ateniaengine)
- 🧾 **Zenodo archive:** [10.5281/zenodo.17970198](https://doi.org/10.5281/zenodo.17970198)

---

## 👤 Author

**Guillermo Alonso Albella**  
GAAIA Labs — Independent Research Initiative

---

## About this project's development

Atenia Engine is developed by Guillermo Alonso Albella with significant use of AI code generation tools — primarily Claude and Claude Code from Anthropic. This section exists because I think transparency about how a project is built matters, and because "made with AI assistance" tends to be either hidden or oversold in most projects.

Here's how it actually works:

**What I decide:**
- Architecture and design choices
- Trade-offs between approaches
- What gets implemented and in what order
- Which ideas are worth pursuing and which aren't
- Code review of every change before merge

**What the AI tools execute:**
- Implementation of approved designs
- Routine refactors and test writing
- Research into specific technical options
- Pattern-matching across existing codebases

**What we do together:**
- Reasoning about complex trade-offs
- Debugging specific issues
- Evaluating alternatives when the path isn't clear

Every commit on this repo passes through my review. The AI tools don't merge autonomously. If you see a questionable decision, it's mine, not the AI's.

This workflow lets one person work at the scope this project requires. The architectural thinking, user-facing decisions, and quality standards are human. The volume of boilerplate and research that would otherwise block progress is handled by tools.

If you have thoughts about this approach, open an issue. I'm more interested in honest feedback than in pretending this was written by hand.

---

## 🧠 Final Note

This README does not try to sell.

It states a position — honestly, including what is built and what is still being built.

**And that's what makes it real.**
