# 🧠 Atenia Engine
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970198.svg)](https://doi.org/10.5281/zenodo.17970198)

> **Current Project Status**
> - ✅ **APX v12 completed** — Adaptive execution intelligence fully validated through reproducible tests.
> - 🔄 **APX v13 in progress** — Hybrid Execution Engine (H.E.E.): architectural unification of hybrid execution modes under dynamic runtime constraints.
> - 📄 **Paper**: Preprint — arXiv submission in progress.
> - 🧾 **Patent**: USPTO Provisional Application No. 63/941,875 (Filed Dec 16, 2025).
> - 🌍 **Website**: https://ateniaengine.com


### Execution intelligence for AI systems that operate in the real world

Modern AI runtimes assume stable hardware.

Reality does not.

GPUs are shared. Memory pressure fluctuates. Schedulers jitter. Execution policies thrash.

Failures are rarely numerical bugs.
They are decision failures.

**Atenia Engine** is an execution-centric AI runtime system that treats execution as a dynamic, adaptive control problem — not as a static orchestration layer fixed at compile time.

---

## ⚙️ Execution Is Not Plumbing

In most AI systems, execution is treated as plumbing:

```
launch kernels → move data → hope the hardware behaves
```

Atenia Engine starts from a different premise:

**Execution makes decisions. Decisions must adapt to reality.**

Execution determines *where*, *when*, and *how* computation runs.
Under dynamic conditions, these decisions must be observed, reasoned about, stabilized, and refined over time.

Atenia treats execution as a **first-class system component** — one that reasons, adapts, and learns from experience, while preserving deterministic and reproducible computation.

---

## 🎯 What Atenia Engine Does

Atenia introduces an execution intelligence layer that:

* 🔍 observes execution-relevant runtime signals
* 🧠 reasons about stability, risk, and hardware behavior
* 🔁 selects and stabilizes execution policies
* 🚫 prevents policy oscillation and thrashing
* 🛑 anticipates failures before they occur
* 🔒 adapts execution **without modifying computational semantics**

All adaptation happens at the **execution level only**.

✔ No semantic drift
✔ No hidden learning
✔ No numerical surprises

---

## 🧘 Stability Before Performance

Atenia does not optimize for peak throughput under ideal conditions.

It optimizes for:

* 🧱 stable execution under noise
* 💾 continuity under memory pressure
* 🔮 predictive resilience instead of reactive failure
* 🎚 confidence over aggressive heuristics

Short-term performance gains mean little if execution collapses under real-world conditions.

**Atenia optimizes for execution that survives.**

---

## 🧠 Learning by Execution Experience (Without ML)

Atenia Engine improves execution behavior over time — **without machine learning**.

Execution outcomes are distilled into **persistent execution memory**.

When similar execution contexts reappear, Atenia can:

* ♻️ avoid previously unstable strategies
* 🎯 converge faster to stable policies
* 🧯 reduce unnecessary fallback and defensive behavior

Seeing the same execution twice should never feel like the first time.

---

## 🧪 Virtual Execution Before Real Risk

Exploration is dangerous when done directly on hardware.

Atenia introduces a **Virtual GPU Execution Model** used to evaluate execution policies *before* they reach physical devices.

This enables:

* 🧪 safe autotuning
* 🚨 risk-aware policy filtering
* 🧯 predictive fallback selection
* 🛡 protection against catastrophic failures (e.g. OOM)

Unstable strategies are discarded **before they touch real hardware**.

---

## 🔬 Reproducible Research

Execution intelligence must be observable to be credible.

All experiments described in the paper are implemented as **executable tests**.

```bash
cargo test
```

If the tests pass, the execution engine is alive.

---

## 📁 Test Execution Guide

All execution and research tests in this repository are **fully documented**.

Detailed instructions on how to run tests, enable debug output, interpret results, and reproduce paper experiments are provided here:

📄 **tests/README.md**

This includes:

* standard test execution
* verbose output (`--nocapture`)
* debug execution mode via environment variables
* guidance for reproducing research results

---

## 🧪 Test Coverage

The repository currently includes:

* ✅ 270+ execution and stability tests
* 📄 paper-specific experimental validations
* 🔁 warm vs. cold execution scenarios
* 🧩 end-to-end adaptive execution tests
* 🧠 full validation up to **APX-12**

---

## 📄 Research Context

The technical foundations of Atenia Engine are described in the following paper:

**Atenia Engine: Hardware-Adaptive Execution Intelligence for Stable and Resilient AI Runtime Systems**

📘 Status: **Preprint (publicly available)**  
🧾 Patent: **USPTO Provisional Application No. 63/941,875**  
📅 Filed: December 16, 2025  

The paper is currently hosted in this repository while awaiting arXiv submission approval.

➡️ **Download PDF:**  
`/paper/Atenia_Engine_Execution_Intelligence.pdf`

All experiments described in the paper are fully reproducible via the test suite included in this repository.

The project is released under **Apache License 2.0** and is compatible with this filing.

---

## ❌ What Atenia Engine Is Not

Atenia Engine:

* ❌ is not a machine learning framework
* ❌ is not a compiler or graph optimizer
* ❌ does not modify model semantics
* ❌ does not require retraining
* ❌ does not assume ideal hardware

It complements existing frameworks by addressing a layer they largely ignore:

**execution stability**.

---

## 🛠 Implementation

* 🦀 Implemented in **Rust**
* 🔒 Deterministic execution behavior
* 🧵 Explicit memory and concurrency control
* 🚫 No garbage collection
* 🧩 No opaque runtime adaptation

Designed to sit **below ML frameworks** and **above raw hardware execution**.

---

## 📜 License

📄 **Apache License 2.0**

Allows broad adoption, modification, and commercial use while providing explicit patent protection.

---

## 🌐 Links

* 🌍 Website: [https://ateniaengine.com](https://ateniaengine.com)
* 💾 Repository: [https://github.com/AteniaEngine/ateniaengine](https://github.com/AteniaEngine/ateniaengine)
* 📄 Paper: *(to be added after arXiv submission)*

---

## 👤 Author

**Guillermo Alonso Albella**
Independent Research Initiative — **GAAIA Labs**

---

## 🧠 Final Note

This README does not try to sell.

It states a position.

And that’s what makes it real.
