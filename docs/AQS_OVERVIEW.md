# AQS — Atenia Quantization Search (Overview)

> **Status: experimental · CPU-only · opt-in · not production certification.**
> AQS is an isolated research subsystem. It is never reached by the
> productive runtime, never enabled by default, and adds no dependency.
> Runtime numeric certification remains governed by ADR-004 / ADR-005
> (see `docs/CERTIFICATION.md`).

This document is the single entry point for understanding AQS. The
per-milestone detail lives in the immutable handoff records
`docs/HANDOFF_AQS_1.md` … `docs/HANDOFF_AQS_10.md`; the original design
audit is `docs/AQS_ARCHITECTURE_AUDIT.md`.

## What AQS is

**Automatic Quantization Search** is an experimental pipeline that, given
a model and a set of candidate quantization policies, measures how much
each policy perturbs the model, classifies the result against the ADR-004
gate, ranks the candidates deterministically, and emits a human report
plus a **draft** manifest. The goal is not a single magic quantization
algorithm — five weight-only techniques were falsified against ADR-004
strict — but an **auditable system that says which quantization is safe
for a given model, and proves it against an F64 reference**.

## Architecture

```text
QuantizationPolicy        (AQS-1)  unified policy trait: BF16 / INT8 / AWQ / Hybrid / GPTQ
        │
        ▼
Tensor drift Evaluator    (AQS-2)  cheap local F32-vs-F32 weight-buffer drift (pre-filter)
        │
        ▼
End-to-End Harness        (AQS-4)  real forward, logits vs F64 fixture (the metric that counts)
        │
        ▼
Certification report      (AQS-6)  classify: Adr004Certified / UsefulLossy / Failed
        │
        ▼
Manifest draft            (AQS-6)  "3.0.0-draft" YAML-ish string (NOT the runtime numcert)
        │
        ▼
Search engine             (AQS-7)  deterministic candidate grid + ranking + selection
        │
        ▼
Runner                    (AQS-8)  callback-based; drives candidates, skips unsupported
        │
        ▼
CLI: atenia search        (AQS-10) renders report + manifest draft from a results file
```

(GPTQ — AQS-3 surrogate then AQS-5 real — plugs in as one policy. AQS-9
wires the runner to the real TinyLlama harness.)

## Milestones

- **AQS-0** — Architecture audit (`docs/AQS_ARCHITECTURE_AUDIT.md`): scoped
  the search space, risks, and roadmap before any code.
- **AQS-1** — `QuantizationPolicy` trait + `CalibrationContext` +
  `PolicyError`; wraps existing BF16 / INT8 / AWQ / Hybrid perturbations
  under one extensible surface.
- **AQS-2** — Per-tensor drift evaluator: clones an F32 weight, applies a
  policy, reports `max_abs_diff` / `mean_abs_diff` / `rmse` / argmax /
  memory. A cheap local signal, explicitly **not** certification.
- **AQS-3** — Experimental GPTQ *surrogate* (diagonal Hessian + error
  diffusion). Wired and unit-tested; later shown insufficient end-to-end.
- **AQS-4** — End-to-end TinyLlama harness: perturbs real weights, runs a
  CPU F32 forward, compares logits to the F64 fixture (the ADR-004 metric).
- **AQS-5** — Real blockwise GPTQ (full `K×K` Hessian, Tikhonov damping,
  Cholesky inverse-Hessian error compensation). Correct, but did not beat
  the plateau on TinyLlama (see Results).
- **AQS-6** — Certification report: deterministic ranking, status
  classification, best-certified / best-useful-lossy selection, and the
  `3.0.0-draft` manifest renderer.
- **AQS-7** — Deterministic search engine: conservative default candidate
  grid + stable policy factory + ranking via the certification report.
- **AQS-8** — Callback-based runner: drives candidate evaluation, declares
  evaluator capabilities, skips unsupported candidates without panicking.
- **AQS-9** — Wires the runner to the real TinyLlama harness from a single
  entry point; GPTQ real skipped by default (opt-in via env).
- **AQS-10** — `atenia search` CLI: renders the report + manifest draft
  from a pre-computed end-to-end results file (no model loading).

## Real TinyLlama results

Measured end-to-end against the F64 fixture (ADR-004 gate
`max_abs_diff < 0.5`). Numbers are from the real AQS-5 / AQS-9 runs:

| policy        | max_abs_diff | argmax | status            |
|---------------|-------------:|:------:|-------------------|
| bf16          | 0.000063     | true   | **adr004_certified** |
| awq (α=0.25)  | 0.889217     | true   | **useful_lossy** (best lossy) |
| hybrid        | 0.831786     | false  | failed            |
| plain_int8    | 1.260771     | false  | failed            |
| gptq (real)   | 1.405399     | false  | failed            |
| gptq (surrogate) | 12.5     | false  | failed            |

Honest reading:

- **bf16 is the only certified policy** (essentially loss-free).
- **AWQ at α=0.25 is the best *useful lossy* option** — argmax-stable,
  ~1.94× compression — but it is **above** the ADR-004 strict gate (0.5),
  so it is **not certified**.
- **GPTQ (both surrogate and real) failed**, and real GPTQ was actually
  *worse* than plain INT8 here. The most likely cause is calibration
  starvation (the Hessian was severely rank-deficient with the small
  calibration set); details in `docs/HANDOFF_AQS_5.md`. Real GPTQ also
  cost ~7.8 h on CPU for one model.
- Five distinct weight-only mechanisms (plain INT8, β outlier, AWQ,
  hybrid, GPTQ) all fail ADR-004 strict on TinyLlama. The weight-only
  **plateau is accepted**; AQS's value is the certification/search layer,
  not a new technique.

## Current status

- **Experimental** — research subsystem; APIs may change.
- **CPU-only** — no CUDA / ROCm / Metal path.
- **Opt-in** — nothing reaches AQS unless explicitly invoked; no policy is
  enabled by default; the productive runtime is untouched.
- **Not production certification** — the `3.0.0-draft` manifest is a draft
  and is never consumed by the runtime. Production numeric certification
  is ADR-004 / ADR-005 (`docs/CERTIFICATION.md`).
- **`atenia search` consumes a results file** produced by the end-to-end
  harness; it does **not** load a model or certify arbitrary models.

## See also

- `docs/AQS_ARCHITECTURE_AUDIT.md` — original design + post-implementation epilogue.
- `docs/HANDOFF_AQS_1.md` … `docs/HANDOFF_AQS_10.md` — per-milestone records.
- `docs/CERTIFICATION.md` — productive ADR-004 / ADR-005 certification.
