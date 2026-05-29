# MoE Certification Substrate (MOE-1)

> **Status: experimental infrastructure only.** This milestone adds **no**
> MoE execution, no router, no top-k, no dispatch, no sparse path, no MoE
> family, no graph operators, and no runtime change. It builds the
> *certification groundwork* so that a future real MoE path has a stable,
> well-defined way to describe a fixture and decide how it can be certified
> against ADR-004. See `src/moe/`.

## Objective

The MOE-0 architecture audit concluded that Atenia's data plane (loaders,
tier planner, disk streaming) is mostly MoE-ready, the compute plane (graph
engine, runtime) is the principal blocker, and — critically —
**certification is gated on having a small MoE fixture**. The ADR-004
contract requires an F64 reference (a PyTorch double-precision forward) per
model; that is infeasible for large MoE checkpoints. MOE-1 encodes the
fixture vocabulary and the certification-strategy decision as code + tests
so the reasoning cannot drift, **before** any MoE code is written.

## How the dense families are certified today

The four ADR-004 fixtures (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B,
Llama 3.2 1B) are certified by:

1. An offline PyTorch script (`tests/fixtures/<model>/generate_f64.py`)
   runs the model in **F64** on a fixed 4-token input and dumps
   `expected_logits_f64.json` (shape `[1, 4, vocab]`).
2. A Rust test runs the Atenia forward and compares logits to that F64
   reference; ADR-004 gate is `max_abs_diff < 0.5`.

The F64 reference is the "practical mathematical truth" (15–17 decimal
digits). Its cost is dominated by holding the **full weights in F64**:
the TinyLlama fixture note records 1.1B params → **~8.8 GB** of F64
weights (8 bytes/param), fitting "comfortably within 16 GB" once
activations/intermediates are added (≈ a 2× peak-over-weights envelope).

**Reusable for MoE, unchanged:** the fixture file format
(`inputs.json` + `expected_logits_f64.json`), the F64-as-truth methodology,
the `max_abs_diff < 0.5` gate, and the offline-script + committed-fixture
flow. **Not reusable as-is:** the *size assumption* — a MoE's total weights
are far larger than its active per-token share.

## What a MoE fixture requires

- A token input + a committed F64 reference of the MoE forward.
- A model whose **full** weights fit in F64 within a realistic host budget
  (the F64 reference must materialise *all* experts, not just the active
  top-k — correctness certification cannot skip the experts a different
  token would route to).
- A stable, license-clear, regenerable artefact that can accompany the
  project for years.

## F64 feasibility — why Mixtral 8x7B is a poor initial fixture

F64 weight bytes = `parameter_count × 8`. Using published sizes:

| Candidate | Total params | F64 weights | Active params | Fits F64 on… |
|---|---:|---:|---:|---|
| **Mixtral 8x7B** | ~46.7B | **~373 GB** | ~12.9B | no commodity host |
| Qwen1.5-MoE-A2.7B | ~14.3B | ~114 GB | ~2.7B | only ≥128 GB, no headroom |
| DeepSeek-V2-Lite | ~15.7B | ~126 GB | ~2.4B | only ≥128 GB, no headroom |
| **tiny synthetic MoE** | ~2M | **~16 MB** | ~1M | any host (trivial) |

Mixtral's ~373 GB F64 footprint is **~42× the TinyLlama fixture** and far
beyond the dev box (RTX 4070 Laptop, 32 GB RAM) or even a 128 GB
workstation. An F64 reference forward also needs activation +
intermediate headroom on top of the weights. So Mixtral cannot be the
*initial* certification fixture: there is no machine in the project's
hardware envelope on which its full-F64 reference can be produced. (These
are arithmetic estimates from `params × 8`; the real peak is higher once
activations are included — they are upper-bounds on feasibility, not
claims of measured runs.)

## Candidate analysis

| Candidate | Size | Experts (top-k) | Certification feasible | Notes |
|---|---|---|---|---|
| Mixtral 8x7B | ~46.7B | 8 (top-2) | **No** (full F64 ~373 GB) | Simplest routing, but un-certifiable at full F64 |
| Qwen1.5-MoE-A2.7B | ~14.3B | 60 +4 shared (top-4) | Partial only | Smallest *real* MoE; F64 ~114 GB → partial-reference strategy |
| DeepSeek-V2-Lite | ~15.7B | 64 +2 shared (top-6) | Partial only | Fine-grained + shared experts; F64 ~126 GB |
| **Tiny synthetic MoE** | ~2M (tunable) | configurable | **Yes (full F64)** | Deterministic, license-free, regenerable, F64-trivial |

## Recommended fixture

**Official Atenia MoE certification fixture: a tiny, synthetic, deterministic
MoE** (e.g. 4 experts, top-2, small hidden dim — a few MB in F64).

Rationale:

- **Full F64 certifiable on any host**, so the ADR-004 methodology applies
  unchanged and strictly — the strongest guarantee.
- **Stable for years**: no external license, no model-version drift, no
  multi-GB download; regenerable from a fixed seed and committed cheaply.
- **Mechanism-focused**: it certifies the *MoE math* (router → top-k →
  expert MLPs → weighted combine) end-to-end, which is exactly what a
  future MoE compute path must reproduce bit-for-bit.

**Secondary (real-model) fixture, when MoE actually ships:**
**Qwen1.5-MoE-A2.7B** under a **partial-reference** strategy — it is the
smallest real MoE and exercises shared experts + finer routing, validating
realism that a synthetic model cannot. It is *not* the initial fixture
because it requires a ≥128 GB host even for a partial reference.

This mirrors the AQS lesson: certify the mechanism on a small deterministic
fixture first; validate realism on a real model second.

## Certification strategy model

`src/moe/fixture.rs` encodes the decision as a pure function:

```rust
pub enum MoECertificationStrategy {
    FullReferenceF64,     // full model fits F64 within host budget
    PartialReferenceF64,  // full model too big; partial / reduced reference
    Unsupported,          // no feasible F64 reference on this host
}

pub fn recommend_strategy(spec: &FixtureMoESpec, host_ram_bytes: u64)
    -> MoECertificationStrategy;
```

Decision (pure, tested):

- F64 weights `< 1 GiB` → `FullReferenceF64` (trivial; the synthetic fixture).
- else fits in `host_ram / 2` (headroom) → `FullReferenceF64`.
- else fits in whole `host_ram` → `PartialReferenceF64`.
- else → `Unsupported`.

`FixtureMoESpec { model_name, family, parameter_count,
active_parameter_count, expert_count, experts_per_token }` is validated for
internal consistency (experts ≥ 1, `experts_per_token ≤ expert_count`,
`active ≤ total`, non-empty names). The evaluated candidates are recorded
as `const` descriptors (`moe::fixture::candidates`) so this analysis cannot
silently diverge from the code.

## Risks

- **No real-model certification yet (HIGH).** The synthetic fixture proves
  the math, not a real MoE's numerics. A partial-reference methodology for
  a real small MoE (Qwen1.5-MoE) is future work and needs a ≥128 GB host.
- **Partial-reference semantics undefined (MEDIUM).** `PartialReferenceF64`
  is a placeholder; what exactly a "partial" reference asserts (layer-wise?
  single-expert? reduced-config?) must be specified before it is trusted.
- **Synthetic ≠ representative (MEDIUM).** A tiny synthetic MoE may not
  surface drift behaviours of production-scale routing; it is a mechanism
  gate, not a realism gate.
- **F64 estimates are weight-only lower bounds (LOW).** Real peak RAM for a
  reference run is higher (activations/intermediates); the feasibility
  table is conservative on "fits", not optimistic.

## Future roadmap (architecture-level; not implemented here)

- **MOE-2** — loader + weight-mapping for expert tensors; fail-loud on
  missing experts.
- **MOE-3** — dense MoE graph (router + compute-all-experts + weighted
  combine), certified F64 on the synthetic fixture.
- **MOE-4** — sparse dispatch (new graph ops + gated execution), numerically
  matched to MOE-3.
- **MOE-5** — routing-aware tier placement (hot experts → VRAM).
- **MOE-6** — real family expansion (Qwen-MoE, DeepSeek-MoE) + partial
  reference; ATKv2 `moe:` DSL extension.

## What MOE-1 did NOT do

No router, no top-k, no dispatch, no sparse MoE, no MoE family, no graph
operators, no runtime / graph / CUDA / tier-planner / adapter-toolkit /
loader / generation / CLI change, no F64 generation, no model loading, no
forward. `src/moe/` is metadata + a pure strategy function + light unit
tests, isolated and experimental.
