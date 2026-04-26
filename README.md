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

- **🦀 Tensor engine** — Forward + backward with autograd, CPU + CUDA paths
- **🧩 AMG (Adaptive Model Graph)** — Graph representation with executor, independent of PyTorch/TF
- **⚡ Fused kernels** — Attention and QKV paths on CPU and GPU
- **🖼 Convolutional ops in AMG** — `Conv2D` and `MaxPool2D` with forward, backward, and autograd tape integration (APX v20 M1)
- **🧮 MNIST pipeline** — Conv2D, MaxPool, Dense on f32 tensors, end-to-end structural
- **🤖 Real LLM execution** — `TinyLlama-1.1B-Chat-v1.0` loads from a HuggingFace `.safetensors` checkpoint and runs forward end-to-end on CPU; logits match a PyTorch reference within F32-vs-BF16 precision drift accumulated over 22 transformer blocks (APX v20 M4.5)
- **🔒 Deterministic execution** — Same input, same output, always
- **📋 Policy registry** — Pure, deterministic, explainable policy layer with evidence-aware evaluation
- **📜 Execution contracts** — Data structures, validators, replay scaffolding
- **📡 Real memory telemetry** — VRAM (via `nvidia-smi` subprocess) and RAM (via `sysinfo`), validated against ground-truth readings
- **🧠 Signal producers** — `FailureCounter` (time-windowed) and `LatencyMonitor` (P50-baseline spike detection) with deterministic purge semantics
- **🚦 SignalBus** — Single integration point that assembles real telemetry into `GuardConditions` (v16) and `PolicyEvidenceSnapshot` (v15)
- **⚡ Reactive execution hook** — `ReactiveExecutionContext` wires the SignalBus to the AMG executor; `execute_checked` returns a typed `ExecutionAbortReason` when a guard triggers mid-run (APX v20 M2)
- **🏗 Portable build** — Auto-detection of CUDA Toolkit and MSVC BuildTools installation paths in `build.rs`; respects `CUDA_PATH` and `MSVC_TOOLS_PATH` overrides

### 🟡 Scaffolding still being wired to runtime signals

Architecture in place, full end-to-end wiring in progress:

- **Production guards** — The guard framework (v16) and the SignalBus (v19) are both real; however, the built-in `ExecutionGuard` implementations currently live in tests only. Guards in production code are a v21 deliverable.
- **Execution Policies (v15)** — `DecisionBias` base weights are hardcoded constants per built-in policy. `evaluate_with_evidence` does react to real signals from the SignalBus, but no execution path consumes the resulting `DecisionBias` yet.
- **AMM Forecaster** — Now exposes real VRAM and RAM readings in addition to the static byte counter. Integration with the memory manager's allocation decisions is pending.
- **Fusion Selector (v6.10/6.11)** — Measures `fused_full_us`; `fused_qkv_us = 0` is a hardcoded placeholder awaiting real paired measurement.
- **`FragmentationWarning`** — Intentionally not emitted. External proxies (reserved memory, per-process attribution) would be semantically misleading. Deferred until the engine exposes its own GPU allocator.
- **MNIST validation** — Structural pipeline runs with a synthetic model. Real MNIST dataset + trained weights pending.
- **v17 model runtime** — Now wired: M4 delivered the safetensors loader (reader, weight mapper, BF16/F16 decode); M4.5 wired the loader to a complete Llama-family graph builder and demonstrated end-to-end execution of TinyLlama-1.1B with PyTorch-bounded numerical drift. Multi-architecture expansion and inference UX (tokenizer, KV cache) are M4.6+ / M5+.

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

**In progress — v20 (model runtime integration, in milestones):**

- **M1 ✅** — `Conv2D` and `MaxPool2D` implemented natively in AMG with forward, backward, tape integration, and gradient checking against finite differences
- **M2 ✅** — `ReactiveExecutionContext` attached to `Graph`; the executor (`run_plan`, `apx7::*` schedulers) consults `check_guard_before_node` before each node; `execute_checked` returns `Result<_, ExecutionAbortReason>` on abort; existing APIs preserved as backward-compatible wrappers
- **M3 ✅** — Real GPU allocation for `Tensor` behind a vendor-neutral storage abstraction (a/c/d closed) plus the M3-e reaction loop that moves real VRAM to RAM on guard `Degrade`. See [docs/HANDOFF_APX_V20_M3.md](./docs/HANDOFF_APX_V20_M3.md).
- **M4 ✅** — Model loader mechanics (safetensors). `SafetensorsReader` (header + body, by-name + iterator access), `WeightMapper` (name → graph parameter mapping with shape validation and `LoadReport` diagnostics), and BF16/F16 → F32 decode. Validated empirically against a real HuggingFace gpt2 checkpoint. See [docs/HANDOFF_APX_V20_M4.md](./docs/HANDOFF_APX_V20_M4.md).
- **M4.5 ✅** — Real model execution. Atenia executes a HuggingFace **TinyLlama-1.1B-Chat-v1.0** checkpoint end-to-end on CPU, with logits that match a PyTorch reference within F32-vs-BF16 precision drift over 22 transformer blocks. New primitives: `NodeType::RoPE` (half-split layout), `NodeType::Permute`, `NodeType::BroadcastMul`; `BatchMatMul` extended to rank 4. New module `src/nn/tinyllama/` with `TinyLlamaConfig` parser, `tinyllama_weight_mapper` (handles the HuggingFace `[out, in]` Linear convention, GQA tile-on-load, attention-scale absorption into K_proj), and `build_tinyllama` (full Llama-2 graph builder). Numerical validation against PyTorch quantified at `max_abs_diff ≈ 0.73`, `mean_abs_diff ≈ 0.06`, with no values diverging by more than 1.0. See [docs/HANDOFF_APX_V20_M4.5.md](./docs/HANDOFF_APX_V20_M4.5.md).
- **M4.6 (proposed)** — Llama-family compatibility expansion: Llama 3.2 (`rope_scaling: "llama3"`), Qwen 2.5 (QK-Norm), Phi 3.5 mini, SmolLM3, Mistral 7B (subject to RAM). Estimated 2–3 days for the architectural deltas, plus regression suite per added family.
- **M4.7 (proposed)** — Beyond-VRAM execution: run a 13B-class model in BF16 on notebook hardware (8 GB VRAM, 16 GB RAM, SSD). The first end-to-end exercise of the M3 reaction loop against a real workload that exceeds available VRAM. 5–10 days estimated.
- **M5+** — Inference UX: tokenizer integration, KV cache, token-by-token generation. The milestone that turns "Atenia produces correct logits" into "Atenia can chat".

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

The repository contains a growing suite (~300 test files as of v20 M2) covering:

- Tensor operations and autograd correctness
- Graph construction and execution
- Deterministic serialization (JSON, CSV)
- Structural integration of APX v13–v20 modules
- CPU + CUDA numerical equivalence where applicable
- Forward equivalence of native AMG conv/pool against the v17 reference
- Gradient checking via central finite differences
- SignalBus producing `GuardConditions` and `PolicyEvidenceSnapshot` from real probes
- AMG executor aborting cleanly on guard verdicts and resuming execution when conditions are clean

See [docs/TESTS.md](./docs/TESTS.md) and [tests/README.md](./tests/README.md) for the test methodology note and category breakdown.

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

- [ROADMAP.md](./ROADMAP.md) — Versioned roadmap with milestones and design constraints per APX version
- [docs/APX.md](./docs/APX.md) — Per-version notes on what each APX milestone introduced and its scope
- [docs/TESTS.md](./docs/TESTS.md) — Test methodology and categorization
- [docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md) — Design principle for reaction strategies (APX v21+)
- [docs/RESEARCH_INTEL_APIS.md](./docs/RESEARCH_INTEL_APIS.md) — Multi-vendor GPU API research (APX v22+)
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
