# Atenia Engine — Roadmap

This document describes the priorities guiding Atenia Engine's development. It is organized by upcoming APX version, from the immediate focus to broader horizons.

This roadmap communicates scope and priority, not calendar commitments. Versions are released when ready.

---

## Current Focus: APX v18 — Runtime Signal Integration

### Context

Atenia Engine has reached APX v17. The work up to this point built a layered architecture — hardware profiling types (v13), execution timelines and memory pressure structures (v14), policy layer (v15), execution contracts and guards (v16), model runtime integration (v17) — almost entirely as scaffolding: pure data structures, validators, and deterministic logic with no live hardware signals feeding them.

The first real sensor landed recently: `src/amm/vram_probe.rs` reads free VRAM from NVIDIA GPUs via `nvidia-smi`, validated end-to-end against ground truth on real hardware. That module is isolated today; it does not yet inform any decision.

APX v18 is the version where scaffolding starts to receive real signals. The existing structures — `GuardConditions`, `PolicySignal`, `AmmForecaster` — were designed to consume runtime telemetry. v18 begins wiring producers to them.

The qualitative shift of v18 is from "the engine would react if told" to "the engine reacts because it measured". It does not yet close the full decision loop (that is v19) and it does not yet cover every signal source (later versions). It establishes the first real path from hardware into an execution-relevant data structure.

### Milestones

1. **Integrate `vram_probe` into `AmmForecaster`** — First real signal flowing from hardware into execution decisions. Replaces the current static byte counter with live VRAM telemetry.

2. **Rewrite `apx_predictive_fallback_test.rs` without hardcoded `panic!()`** — Replace synthetic OOM simulation with real memory pressure detection via `vram_probe`. Closes the most problematic test from the original paper's experimental claims.

3. **Expand to system RAM telemetry** — Add a second real signal (RAM / memory pressure via `sysinfo`). Completes the memory-side signal bus for the AMM layer.

### Out of scope for v18

- Guard integration (v19)
- Policy signal producers (v19)
- Multi-GPU support
- Non-NVIDIA hardware (ROCm, Metal, Vulkan)
- Fragmentation analysis

---

## v19 — Emergent Policy Decisions

v19 connects the signals produced in v18 to the Guards layer (v16) and the Policies layer (v15). `GuardConditions` will be populated from real telemetry rather than test fixtures, and `PolicySignal`s will be emitted by runtime observers so that `evaluate_with_evidence` receives live evidence. This closes the first sensor → decision → action loop: a measured change in runtime state leads to a different execution choice without any synthetic injection in the path.

---

## v20 — Execution Memory and Learning

v20 aims to persist execution outcomes across runs so that similar contexts can inform future decisions. The intent is "learning by experience, without ML" from the main README: no gradient descent, no hidden training — just a durable record of what worked under which conditions, queried deterministically when the runtime sees a similar situation again. This version turns v19's reactive loop into an accumulating one.

---

## v21–v25 — Future Horizons

Directions under consideration, listed without strict ordering:

- Multi-backend support (ROCm, Metal, Vulkan) beyond the current NVIDIA-only path
- Distributed execution across multiple hosts
- Production readiness: reproducible benchmarks, stability validation at scale, operational tooling
- Autonomous runtime: a full adaptation cycle that operates without human intervention

---

## How to Contribute

This is a research-in-progress. Contributions, issues, and technical discussions are welcome — especially from people with experience in:

- GPU runtime systems and CUDA / ROCm low-level APIs
- Memory management and OOM prevention
- Adaptive scheduling and execution policies
- Systems research and MLSys

Open an issue or reach out if you want to collaborate on any specific layer.

---

## Design Principles

Reproduced from the main [README](./README.md) as a reminder of the constraints every roadmap item is expected to respect:

- **Stability before performance** — Short-term gains mean nothing if execution collapses under noise.
- **Adaptation without semantic drift** — The engine may change *how* things run, never *what* is computed.
- **Learning by experience, without ML** — Execution outcomes are distilled into persistent memory — no opaque training loops in the runtime.
- **Observable and reproducible** — Every behavior claimed by the engine must be verifiable through executable tests.
