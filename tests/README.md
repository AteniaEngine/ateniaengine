# Atenia Engine — Test Suite

This directory contains the **full validation and experimentation test suite** for Atenia Engine.

Unlike traditional projects, Atenia Engine relies heavily on **executable tests as proof of behavior**, stability, and execution intelligence.

---

## 📌 Purpose of This Test Suite

The tests in this directory serve multiple roles:

- ✔️ Validate correctness of mathematical operations
- ✔️ Verify execution stability under dynamic conditions
- ✔️ Prove adaptive behavior across executions
- ✔️ Detect regressions during rapid engine evolution
- ✔️ Support claims made in the scientific paper (“Proof, Not Promises”)

This is **not** a minimal test set.  
It is an **engineering laboratory**.

---

## 🧪 Types of Tests You Will Find Here

### 1. Core Correctness Tests
Verify numerical equivalence, gradients, tensor layouts, and operator correctness.

Examples:
- `*_correctness_test.rs`
- `*_equivalence_test.rs`
- `tensor_*_test.rs`

---

### 2. Execution & Stability Tests
Validate that Atenia Engine maintains stable execution decisions across runs.

Examples:
- `apx_runtime_stability_test.rs`
- `apx_learning_effect_test.rs`
- `apx_predictive_fallback_test.rs`
- `apx_safe_autotuning_test.rs`

These tests are central to Atenia’s **Execution Intelligence** claims.

---

### 3. Adaptive & Policy Tests
Ensure that scheduling, planning, and policy layers behave as expected.

Examples:
- `*_scheduler_test.rs`
- `*_selector_test.rs`
- `*_autoplanner_test.rs`
- `*_adaptive_test.rs`

---

### 4. Benchmarks Embedded as Tests
Some benchmarks are implemented as tests for convenience and reproducibility.

These tests:
- print timing and performance data
- assert correctness before reporting speedups
- are **not meant for Cargo Bench**

Examples:
- `*_benchmark_test.rs`
- `*_bench.rs`

---

### 5. End-to-End & Integration Tests
Validate full execution paths across multiple subsystems.

Examples:
- `apx_end_to_end_adaptive_execution_test.rs`
- `gpu_pipeline_test.rs`
- `mini_flux_training_test.rs`

---

## 🧠 Relation to the Scientific Paper

A **subset** of these tests is referenced directly in the paper as empirical evidence.

Those tests:
- are reproducible
- run with no external dependencies
- demonstrate learning effects, stability, and fallback behavior

They intentionally live **alongside the full test suite**, not in isolation, to show that the paper claims are backed by a real system.

---

## How to Run the Tests

### Standard execution
```bash
cargo test
Show test output (println!)
bash
Copy code
cargo test -- --nocapture
 Debug / Introspection Mode
Atenia Engine supports a debug introspection mode enabled via an environment variable.

When enabled, the engine emits detailed internal logs including:

execution decisions

policy selection

fallback triggers

profiling and stability signals

Windows (PowerShell)
powershell
Copy code
$env:ATENIA_DEBUG="1"
cargo test -- --nocapture
Linux / macOS
bash
Copy code
ATENIA_DEBUG=1 cargo test -- --nocapture
This mode does not change execution semantics.
It only increases observability and is intended for research and debugging.

 Notes

The directory contains **284 test files** (current revision). Categories overlap — an APX integration test may also be a benchmark — so counts do not sum to the total.

| Category | Approx. count | What they assert |
|---|---|---|
| Core correctness / numerical / gradient | ~30 | Equivalence against reference CPU or analytical solutions |
| APX phase integration (structural) | ~175 | Module wiring and deterministic behavior across APX 2.x–17.x |
| Adaptive / scheduler / policy / selector | ~35 | Decision logic over synthetic profiles and injected conditions |
| Benchmarks embedded as tests | ~25 | Timing + correctness assertion (not for `cargo bench`) |
| GPU smoke / safety / loader fallback | ~15 | Shape, finiteness, no-panic under `CpuFallback` |
| Training / end-to-end / pipeline | ~20 | Multi-subsystem integration, synthetic inputs |
| Hybrid memory / checkpoint / planner | ~25 | Logical tier movement, serialization determinism |
| Infrastructure (HAL, loader, CLI, etc.) | ~15 | Supporting utilities |

Not all tests are lightweight or fast. Some tests intentionally stress the system to exercise fallback paths. For quick demos, refer to the `examples/` directory instead.

🧭 Philosophy
If a system claims intelligence at runtime,
its behavior must be observable, reproducible, and executable.

This test suite exists to enforce that principle.