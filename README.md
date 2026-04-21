# 🧠 Atenia Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970198.svg)](https://doi.org/10.5281/zenodo.17970198)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**An execution-centric AI runtime system written from scratch in Rust.**

> [!NOTE]
> **Status: Early research in progress.**  
> This project is a working prototype with architectural scaffolding for 
> hardware-adaptive execution intelligence. Several claimed capabilities 
> are implemented as deterministic scaffolding and are currently being 
> connected to real hardware signals.  
> See [Current State](#-current-state) for an honest breakdown.

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

---

## 📊 Current State

Atenia Engine is implemented in Rust. The project follows an APX (Adaptive Execution) versioning scheme from v1 to v25, describing the progression from primitives to a fully adaptive runtime.

### ✅ What works today

Real, executable, deterministic:

- **🦀 Tensor engine** — Forward + backward with autograd, CPU + CUDA paths
- **🧩 AMG (Adaptive Model Graph)** — Graph representation with executor, independent of PyTorch/TF
- **⚡ Fused kernels** — Attention and QKV paths on CPU and GPU
- **🧮 MNIST pipeline** — Conv2D, MaxPool, Dense on f32 tensors, end-to-end structural
- **🔒 Deterministic execution** — Same input, same output, always
- **📋 Policy registry** — Pure, deterministic, explainable policy layer
- **📜 Execution contracts** — Data structures, validators, replay scaffolding
- **📡 System load sampling** — Real CPU metrics via `sysinfo`

### 🟡 Scaffolding in place (not yet wired to runtime signals)

Architecture exists. Integration with real hardware signals is in progress:

- **AMM Forecaster** — Currently a tensor byte counter. Needs real VRAM/RAM telemetry.
- **Runtime Guards (v16)** — Pure evaluator of `GuardConditions`. No producer from runtime yet.
- **Execution Policies (v15)** — Return hardcoded `DecisionBias` constants. Waiting on emergent signals.
- **Fusion Selector (v6.10/6.11)** — Measures `fused_full_us`; `fused_qkv_us = 0` is a hardcoded placeholder.
- **Predictive fallback** — Demonstrated in controlled test harnesses. Engine-native trigger pending.
- **MNIST validation** — Structural pipeline runs with a synthetic model. Real MNIST dataset + trained weights pending.

### Roadmap (APX v18 → v25)

**Completed:**

- **v12** — Initial learning engine scaffolding (withdrawn paper)
- **v13** — Hybrid Execution Engine (H.E.E.) — adaptive placement scaffolding
- **v14** — Execution timeline + profile infrastructure
- **v15** — Policy layer (5 built-in policies, evidence-driven evaluation)
- **v16** — Execution contracts and guard framework
- **v17** — Kernel normalization and symbolic GPU chain
- **v18** — Memory telemetry foundation:
  - Real VRAM probe via nvidia-smi
  - Memory pressure detection (replaces the withdrawn predictive fallback test)
  - System RAM telemetry via sysinfo
- **v19** — SignalBus: integrated sensor-to-decision pipeline
  - All 4 GuardConditions fields sourced from real telemetry
  - 4 of 5 PolicySignalKind variants produced; FragmentationWarning deferred
    pending a dedicated GPU allocator
  - FailureCounter and LatencyMonitor as internal producers

**Next:**

- **v20** — Real external model loading (ONNX via ModelLoader)
- **v21** — Emergent policy decisions from telemetry (real guard/policy 
  effects on execution flow)
- **v22** — Multi-backend foundation: abstraction layer for vendor-agnostic 
  hardware probes and kernel compilation (NVIDIA + Intel iGPU as first
  coexistence target)
- **v23** — ROCm backend (AMD)
- **v24** — Metal backend (Apple Silicon)
- **v25** — Distributed execution, autonomous runtime

---

## 🔬 Running the Code

Atenia Engine compiles with Rust stable (2024 edition or later) and requires no external ML frameworks.

```bash
cargo build --release
cargo test
```

### Test coverage

The repository contains 270+ tests covering:

- Tensor operations and autograd correctness
- Graph construction and execution
- Deterministic serialization (JSON, CSV)
- Structural integration of APX v13–v17 modules
- CPU + CUDA numerical equivalence where applicable

> [!WARNING]
> **Note on test methodology.**  
> Some tests from earlier APX versions use controlled harnesses that 
> inject runtime conditions (memory pressure, policy competition) to 
> exercise the scaffolding. These are being rewritten to derive signals 
> from the engine itself as part of the v18+ roadmap.

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

- 🦀 Implemented in **Rust**
- 🔒 Deterministic execution behavior
- 🧵 Explicit memory and concurrency control
- 🚫 No garbage collection
- 🧩 No opaque runtime adaptation

---

## 🧾 Intellectual Property

- **Patent:** USPTO Provisional Application **63/941,875** (filed December 16, 2025)
- **License:** Apache License 2.0 (with explicit patent grant)
- **Author:** Guillermo Alonso Albella — GAAIA Labs (Independent Research Initiative)

Apache 2.0 allows broad adoption, modification, and commercial use while providing explicit patent protection.

---

## 📄 Research Paper

The initial research preprint has been withdrawn while the implementation matures to fully back its empirical claims.

See [`paper/README.md`](paper/README.md) for details. A revised version with end-to-end empirical validation will be published once runtime signal integration (APX v18+) is complete.

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

## 🧠 Final Note

This README does not try to sell.

It states a position — honestly, including what is built and what is still being built.

**And that's what makes it real.**

