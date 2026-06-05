# ADR-007: MoE Certification by Decomposition — the L0–L4 Ladder

## Status

Proposed

## Date

2026-06-03

## Context

ADR-004 (F64 Reference as Default, accepted 2026-04-28) established Atenia's
numeric-correctness contract: a model is **certified** when its F32 forward,
compared against a PyTorch `model.double()` F64 reference on the canonical
4-token sequence `[1,100,200,300]`, satisfies `max_abs_diff < 0.5` **and**
matches the F64 argmax at every position. This contract is strong precisely
because a **dense** forward is *complete*: every weight participates in every
forward, so one F64 comparison certifies the whole model.

The `MOE_CERTIFICATION_AUDIT.md` audit (approved) established that ADR-004's
global-F64 method **cannot be applied to real Mixture-of-Experts checkpoints**,
for two independent reasons:

1. **Scale.** The F64 reference must materialise the model's weights at 8
   bytes/parameter. A MoE's *total* parameters vastly exceed its *active*
   (per-token) parameters: Mixtral-8x7B ~46.7 B total → **~373 GB** F64;
   Qwen1.5-MoE-A2.7B ~14.3 B → **~114 GB**; DeepSeek-V2-Lite ~15.7 B → **~126 GB**
   (arithmetic estimates, `params × 8`, per `src/moe/fixture.rs`). None fits the
   project's hardware envelope (32 GB dev box; optional ≥128 GB workstation).

2. **Sparsity / data-dependent routing.** A MoE forward routes each token to the
   **top-k of N** experts; the other `N−k` experts are never exercised. A single
   F64 forward therefore certifies a **data-dependent subgraph**, *not the
   model*. MoE also introduces a **discrete** correctness axis absent from dense
   models — *did the router select the same top-k experts as the F64 truth?* — a
   near-tie in the gate can flip an expert between F32 and F64 and change the
   entire downstream computation.

The audit's approved conclusion is **certification by decomposition**: split the
MoE forward into units the F64 oracle *can* afford on commodity hardware — each
**expert** (one at a time: ~1.4 GB F64 for a Mixtral expert, ~69 MB for a
Qwen-MoE expert), the **router** (a single small linear, trivial), and the
**attention** block (dense, small) — certify each on the **real weights**,
certify their **assembly** on a full-F64 **topology** fixture, and report the
**lowest level fully passed**. This ADR formalises that conclusion as a
**certification ladder (L0–L4)** over a fixed set of **obligations (C1–C5)**, and
fixes the **reporting discipline** that keeps a partial (e.g. L2) certificate
from being read as the dense-equivalent (L4) certificate.

**This ADR changes no thresholds.** Every numeric gate reuses ADR-004 verbatim
(`max_abs_diff < 0.5` + argmax). No existing numeric criterion is degraded.

## Decision

### 1. The obligations (C1–C5)

A MoE certificate is built from five **obligations**. Each names an *objective*,
the *required evidence*, and an *approval criterion*. Approval criteria reuse the
ADR-004 bar; none is weakened.

#### C1 — Per-expert numeric parity

- **Objective.** Prove that **every** expert MLP, on the **real** checkpoint
  weights, reproduces mathematical truth within F32 precision.
- **Evidence.** For each expert `i` of every MoE layer: `Atenia.expert_i(x)`
  vs a PyTorch F64 reference of that single expert, over a fixed input sweep.
  The F64 reference holds **one expert at a time** in F64 (RAM bounded by the
  largest single expert, not the model).
- **Approval.** `max_abs_diff < 0.5` for **every** expert over the sweep
  (ADR-004 bar). Coverage must be **exhaustive** (all experts), not sampled —
  this is what closes the sparsity blind spot. A single expert exceeding the bar
  fails C1.

#### C2 — Router (routing-decision) parity

- **Objective.** Prove that Atenia selects the **same experts** as truth and
  combines them with the same weights — the MoE-specific axis dense models lack.
- **Evidence.** Per routed position over the input sweep: (a) the top-k **expert
  index set** Atenia selects vs the F64 router's set; (b) router-logit
  `max_abs_diff`; (c) the renormalised / gated combine weights
  (`renormalize_topk`, sigmoid-gated shared expert) vs F64; (d) the **routing
  margin** = the minimum gap between the k-th and (k+1)-th gate logit.
- **Approval.** **Top-k expert-set equality is the hard gate** — any selection
  divergence on the swept inputs fails C2. Router-logit `max_abs_diff < 0.5` and
  combine-weight parity `< 0.5` are required numeric checks; the routing margin
  is **reported as a fragility signal** (a small margin warns the certificate is
  route-fragile near ties) but is not itself a gate.

#### C3 — Attention parity

- **Objective.** Prove the attention block around the MoE FFN matches truth.
- **Evidence.** The attention block (MHA / GQA; MLA is **out of charter /
  deferred**) vs a `model.double()` F64 reference of that block, reusing the
  dense ADR-004 method (attention weights are dense and small).
- **Approval.** `max_abs_diff < 0.5` + argmax where applicable (ADR-004 bar).

#### C4 — Assembly / topology parity

- **Objective.** Prove the engine reproduces the **real routing topology** —
  expert count, top-k, GQA ratio, shared-expert presence/gating — assembled
  end-to-end, at full F64.
- **Evidence.** A reduced-hidden-dimension fixture carrying the **real topology**
  (not the real weights), run **full-F64** end-to-end vs HuggingFace f64. This is
  exactly the existing **scale-topology** certification (`mixtral_scale`
  1.639e-07, `qwen_scale` 1.490e-07; see `HANDOFF_MOE_FULL_15.md`).
- **Approval.** End-to-end `max_abs_diff < 0.5` + argmax 4/4 (ADR-004 bar).
- **Scope note.** C4 certifies the *mechanism* (routing + combine + assembly),
  **not** the trained weights. It transfers to real weights **only in
  conjunction with C1** (which certifies the real weights), because the numeric
  behaviour of a correct implementation is weight-independent *given identical
  dtype and ops* — but that argument is only sound when the real weights are
  separately certified.

#### C5 — Active-path parity (real weights, canonical input)

- **Objective.** Prove the **real model, as actually run**, matches truth on the
  tested inputs — the closest reachable analogue of the dense end-to-end check.
- **Evidence.** The full real-weight forward on the canonical input, referenced
  over the **active subgraph** (only the top-k experts actually routed). Two
  admissible oracles:
  - **F64-active** when the active set fits in F64 RAM; or
  - **F32 cross-framework** (PyTorch F32, per ADR-002 Level 2) when F64-active
    exceeds RAM — explicitly weaker, drift **documented**.
- **Approval.** `max_abs_diff < 0.5` + argmax against the F64-active reference;
  **or**, for the F32 cross-framework fallback, a documented F32-vs-F32 drift
  **with exact argmax / greedy ids** (precedent: DeepSeek scale 7.806e-03 with
  exact argmax, `HANDOFF_MOE_FULL_15.md`). The fallback's weaker guarantee is
  recorded in the manifest, never silently equated to the F64 path.

### 2. The certification ladder (L0–L4)

A MoE family is certified at the **highest level whose every obligation passes**
on a representative real checkpoint. Levels are cumulative and escalating.

| Level | Name | Obligations required | Claim it licenses |
|---|---|---|---|
| **L0** | **Topology-certified** | C4 | "the engine reproduces the real routing/topology vs F64" — the *mechanism*, not the real weights |
| **L1** | **Component-certified** | C1 + C2 + C3 (on **real weights**) | "every expert, the router, and attention match truth on the real weights" — the *pieces*, not their composition |
| **L2** | **Assembly-certified** | L1 + C4 | "the pieces **and** their composition (topology) are certified" |
| **L3** | **Active-path-certified** | L2 + C5 | "the real model, as actually run, matches truth on the tested inputs" |
| **L4** | **Full-certified (dense-equivalent)** | global `model.double()` F64 of the **entire** model (ADR-004 strict, unmodified) | "the model is certified exactly as a dense model is" |

**L4 is reserved and, for any MoE with more than ~3 B active parameters,
currently unreachable** within the project hardware envelope (the global-F64 RAM
wall). L4 is **never fabricated**: a family that has not passed L4 lists it as
`pending` with the **exact blocker** (e.g. "L4: requires ~373 GB F64 RAM").

**L0 is already satisfied** for Mixtral, Qwen-MoE and DeepSeek-MoE (scale-topology
certs, `HANDOFF_MOE_FULL_15.md`). **MOE-CERT-2 + MOE-CERT-2-ext** raised Qwen-MoE
to **L1 (whole model)**: C1 exhaustive over **all 24 layers × 60 routed experts =
1440 experts** on the real trained weights (global worst `max_abs_diff` 4.768e-7,
layer 6 / expert 37, 0 failures); C2 top-k set match on all 24 layers (0 failures,
min routing margin 0.001834, no flip); C3 via the existing attention mechanism
cert. **MOE-CERT-3** then folded in **C4** (the Qwen-MoE scale-topology end-to-end
cert vs HF f64 = 1.490e-7, `tests/moe_scale_cert_test.rs`) to reach L2 = L1 + C4.
**MOE-CERT-4** then added **C5** (active-path): Atenia's controlled `MoeRuntime`
full forward of the real Qwen1.5-MoE-A2.7B on a canonical 4-token input vs a
float64 reference computed **one decoder layer at a time** (HF's own module;
driver validated on the tiny fixture) — end-to-end `max_abs_diff` 1.866e-4 (< 0.5)
+ per-position argmax exact 4/4 — reaching **MoE-certified L3 (active-path-
certified, whole model)** = L2 + C5 (`docs/numcert/qwen1.5-moe-a2.7b.moecert.json`,
`tests/moe_cert4_qwen_active_path_test.rs`). The F64 reference is never the whole
model in F64 (one layer at a time) → **L3 is not L4**; only **L4** (global F64,
~114 GB) remains, reserved/unreachable.

**DeepSeek-V2-Lite (MLA) — also MoE-certified L3.** The same ladder was applied to
the real DeepSeek-V2-Lite (MLA-1): **C1** exhaustive over **all 26 MoE layers × 64
routed experts = 1664 experts** (global worst `max_abs_diff` 1.907e-6, 0 failures);
**C2** top-6 set match on all 26 layers (min routing margin 0.011981); **C4** the
MLA-0 V2-Lite-topology cert; **C5** Atenia's full forward of the real weights —
streamed through the **MLA-2 disk expert-tier** (~4 GB RAM instead of ~58 GB) — vs a
float64 one-layer-at-a-time HF reference = end-to-end `max_abs_diff` **2.587e-5**
(< 0.5) + per-position argmax exact 4/4, deterministic → **MoE-certified L3
(active-path-certified, whole model)** (`docs/numcert/deepseek-v2-lite.moecert.json`,
`tests/moe_mla1_deepseek_c5_active_path_test.rs`). C5 required the **MLA-3** fix
(YaRN `attention_scaling`, not `mscale²` on the softmax scale — see
`docs/HANDOFF_MLA1_C5_ROOT_CAUSE.md`). One layer at a time in F64 → **not L4**; L4
(global F64, ~126 GB) remains reserved/unreachable.

**Mixtral remains at L0** (topology only; real 8x7B weights not provisioned);
raising it up the ladder is later work.

### 3. Reporting discipline (how a partial level is shown honestly)

The central risk is **reading L2 as L4** — a *coverage illusion* in which
"certified" is heard as the dense-equivalent guarantee. The following rules are
binding wherever a MoE certification status is shown.

1. **The level qualifier is mandatory.** The bare word "certified" is
   **forbidden** for MoE. Every claim is written `MoE-certified Ln` (e.g.
   "Qwen-MoE: **MoE-certified L2**"), naming exactly what was exercised.

2. **Headline = lowest fully-passed level.** A family's headline status is the
   **lowest** level all of whose obligations pass. Higher levels are listed
   `pending` with their concrete blocker. (Mirrors the dense numcert discipline
   where CPU-F32 `certified` and GPU-TF32 `pending` coexist in one manifest.)

3. **MoE manifests are a distinct schema variant.** A MoE numcert manifest uses
   `schema_variant: "moe-decomposition"` and carries **per-obligation evidence**
   (C1…C5 blocks with their drift / set-equality / margin numbers) plus a
   `ladder_level` field and a `pending_levels` array (each with `blocker`). It
   does **not** reuse the dense `drift_envelope` shape as if a single global F64
   number existed — because it does not.

4. **`STATUS.md` and `FAMILY_COVERAGE_AUDIT.md` vocabulary.** MoE certification
   is reported with the explicit ladder level and is kept **typographically
   distinct** from dense ADR-004 `CERTIFIED`. The existing
   `FAMILY_COVERAGE_AUDIT.md` status vocabulary (EXPERIMENTAL < SUPPORTED <
   VALIDATED < CERTIFIED) gains a parallel MoE lane: **`MoE-certified Ln`**,
   which is **not** the same token as the dense `CERTIFIED` and must never be
   collapsed into it.

5. **`MODEL_FAMILY_VALIDATION.md` framing.** The MoE section states the ladder
   level, the obligations passed, and — explicitly — that an Ln (`n < 4`)
   certificate is **not** the dense ADR-004 guarantee, with the unreached
   obligations named.

6. **No cross-bar comparison.** A MoE Ln status is never tabulated in the *same
   column* as a dense `CERTIFIED` status without the level qualifier, and the
   two are never summed into a single "N families certified" count without
   splitting dense-certified from MoE-certified-Ln.

## Rationale

- **It reuses, never weakens, ADR-004.** Every numeric gate is the ADR-004 bar
  (`< 0.5` + argmax). The ladder adds *structure and honesty*, not leniency: it
  states which subgraph each level certifies instead of pretending one global
  number exists.
- **It is feasible on commodity hardware.** Decomposition turns an infeasible
  `total × 8` F64 (373 GB) into per-expert F64s bounded by the **largest single
  expert** (~1.4 GB) — and per-expert coverage is the *only* method that also
  certifies the experts a single dense-style forward would skip.
- **It names the MoE-specific failure modes.** C2 makes the discrete routing
  decision a first-class, hard-gated obligation, and the routing-margin metric
  surfaces near-tie fragility that a pure `max_abs_diff` would hide.
- **It refuses to over-claim.** The reporting rules exist so that a real,
  valuable L2 certificate is never mistaken for the (currently unreachable) L4
  guarantee — the same discipline the dense numcerts already apply to CPU-F32 vs
  GPU-TF32.

## Consequences

### Positive

- A precise, auditable definition of "certified" for MoE that scales to real
  checkpoints on a 32 GB host: **Qwen1.5-MoE-A2.7B = L3** and **DeepSeek-V2-Lite
  (MLA) = L3** (active-path, via the one-layer-at-a-time F64 reference + the disk
  expert-tier), **Mixtral = L0** (topology only, real weights not provisioned). L4
  (global F64) reserved/unreachable in this hardware envelope.
- Exhaustive expert coverage (C1) eliminates the sparsity blind spot honestly.
- A reporting contract that structurally prevents the L2-read-as-L4 confusion.
- Defines the `PartialReferenceF64` strategy slot that `src/moe/fixture.rs`
  reserved but left unspecified — without writing code in this milestone.

### Negative

- More obligations and more bookkeeping than a single dense F64 number: a MoE
  certificate is a *composite* artefact (per-obligation evidence), not one line.
- L3 active-path on larger families (Mixtral) needs either a ≥128 GB host or the
  weaker F32 cross-framework oracle — a documented compromise, not a clean F64.
- L4 remains explicitly unreachable for large-active MoE; the contract makes that
  permanent-until-hardware-changes, and visible, rather than papering over it.

### Neutral

- This ADR is **definition-only**. No harness, no manifest, no execution, no
  runtime / loader / `MoeRuntime` / Adapter Toolkit change. Implementing C1–C5
  and raising families up the ladder is the work of MOE-CERT-2/3/4.
- DeepSeek-MoE / MLA stays out of charter (C3 covers MHA/GQA only); its later
  inclusion would extend C3, not change the ladder.

## Alternatives Considered

### Alternative 1 — Lower the ADR-004 threshold so a partial reference "passes"

Rejected. Degrading the numeric criterion to make a feasible-but-incomplete check
pass is exactly the "hide the question" failure ADR-001/ADR-004 forbid. The
ladder keeps the bar and instead states *which subgraph* meets it.

### Alternative 2 — Certify only the synthetic / topology fixture (stop at L0)

Rejected as the *headline*. L0 certifies the mechanism on random weights; it says
nothing about a real checkpoint's trained-weight numerics. It is retained as the
**floor** of the ladder, not the claim.

### Alternative 3 — Sample experts instead of exhaustive C1 coverage

Rejected. Sampling reintroduces the sparsity blind spot the decomposition exists
to close, and per-expert F64 is cheap enough (one small expert at a time) that
exhaustive coverage is affordable. C1 requires **all** experts.

### Alternative 4 — Treat the existing HuggingFace block reference as sufficient

Rejected as a *family* certificate. The HF block reference is a partial oracle
(one layer-0 block, not the assembled real-weight model); it is evidence toward
C1/C2 at block granularity, not a substitute for the ladder.

## Trigger to Revisit

- The hardware envelope changes (e.g. a ≥256 GB host) such that **L4** becomes
  reachable for a target family — then the global-F64 path is preferred and the
  ladder records L4 directly.
- A routing-margin study shows the swept inputs systematically under-cover
  near-tie routing — then C2's coverage (input sweep size / adversarial routing
  inputs) must be strengthened before an Ln claim stands.
- DeepSeek-MoE / MLA enters charter — then C3 extends to MLA and the ladder is
  re-evaluated for that family.
- A cross-framework (F32) C5 result diverges in argmax from greedy truth — then
  the fallback is insufficient and C5 must wait for an F64-active host.

## References

- [ADR-004 — F64 reference as default](./ADR-004-f64-reference-as-default.md) —
  the numeric contract this ADR extends to MoE **without changing its
  thresholds**; the `< 0.5` + argmax bar is reused verbatim.
- [ADR-002 — Mathematical ground-truth validation](./ADR-002-mathematical-ground-truth-validation.md)
  — Level 2 (cross-framework F32) is the admissible C5 fallback oracle.
- [ADR-001 — No decisions without data](./ADR-001-numerical-health-monitor-deferred.md)
  — the "do not hide the question by lowering the gate" principle.
- [MOE_CERTIFICATION_AUDIT.md](../MOE_CERTIFICATION_AUDIT.md) — the approved audit
  this ADR formalises (options A–E, the ladder, the feasibility analysis).
- [MOE_CERTIFICATION_SUBSTRATE.md](../MOE_CERTIFICATION_SUBSTRATE.md) /
  `src/moe/fixture.rs` — the `MoECertificationStrategy` substrate whose
  `PartialReferenceF64` slot this ladder defines.
- [HANDOFF_MOE_FULL_15.md](../HANDOFF_MOE_FULL_15.md) — the scale-topology certs
  that already satisfy **L0** (C4).
- [CERTIFICATION.md](../CERTIFICATION.md) — the dense numcert manifest contract
  the MoE `schema_variant: "moe-decomposition"` parallels.
- [FAMILY_COVERAGE_AUDIT.md](../FAMILY_COVERAGE_AUDIT.md) /
  [MODEL_FAMILY_VALIDATION.md](../MODEL_FAMILY_VALIDATION.md) — the status
  vocabularies that gain the `MoE-certified Ln` lane.
