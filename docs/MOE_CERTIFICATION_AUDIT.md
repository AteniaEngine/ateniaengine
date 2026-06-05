# MoE Certification Methodology — Audit (MOE-CERT-AUDIT, design-only)

> **Status: approved → formalised.** The ladder and obligations proposed below
> are now formalised in
> **[ADR-007 — MoE certification ladder](./decisions/ADR-007-moe-certification-ladder.md)**
> (MOE-CERT-1). This document remains the analysis/rationale record; ADR-007 is
> the authoritative definition of L0–L4 + C1–C5 + the reporting discipline.

**Audit only — no code, no manifests, no commits, no execution, no CI.**
Designs an *honest, scalable* certification methodology for MoE families,
grounded in what already exists. Sources (FASE 1, all read):
`docs/decisions/ADR-004-f64-reference-as-default.md`,
`docs/MOE_PRODUCTION_READINESS_AUDIT.md`, `docs/MOE_ADAPTER_SPEC_AUDIT.md`,
`docs/MOE_OVERVIEW.md`, `docs/MOE_CERTIFICATION_SUBSTRATE.md`,
`docs/HANDOFF_MIXTRAL_CERT_1.md`, `docs/HANDOFF_QWEN_MOE_CERT_1.md`,
`docs/HANDOFF_MOE_FULL_13.md`, `docs/HANDOFF_MOE_FULL_15.md`,
`src/moe/fixture.rs`, and the dense numcert manifests
(`docs/numcert/*.numcert.json`).

> **One-sentence thesis.** A real MoE checkpoint cannot be certified the way a
> dense model is (one global F64 forward), because (a) its full weights do not
> fit in F64 RAM and (b) a single forward only exercises the *top-k routed*
> experts — so the honest path is to **decompose** the MoE forward into
> independently-F64-certifiable units (router, each expert, attention, combine),
> certify each against F64 on the **real weights**, then certify their
> **assembly** on a full-F64 *topology* fixture, and label the result by the
> **lowest level fully passed** — never claiming the unreachable global-F64
> level.

---

## FASE 2 — What does "certified" mean? Dense vs MoE

### Dense certification today (ADR-004, the baseline contract)

| Element | Dense (Llama/Qwen2/Gemma2/Gemma3) |
|---|---|
| **Oracle** | PyTorch `model.double()` — same architecture, **all** weights in F64 |
| **Input** | canonical 4-token sequence `[1,100,200,300]` |
| **Metric** | `max_abs_diff(Atenia F32 logits, F64 logits) < 0.5` **and** argmax match 4/4 |
| **Property proven** | the *entire* computational path (embed → every layer → lm_head) reproduces mathematical truth within F32 precision |
| **Coverage** | **complete by construction**: a dense forward touches **every** weight, so one input certifies the whole graph |
| **Feasibility** | F64 weights = `params × 8`; ~8.8 GB for 1.1 B, fits 32 GB up to ~2–3 B params |

The dense guarantee is strong *precisely because the forward is dense*: there is
no weight the canonical input fails to exercise. Certifying one forward
certifies the model.

### Why MoE breaks every one of those assumptions

| Element | What changes for MoE | Consequence |
|---|---|---|
| **Coverage** | a forward routes each token to **top-k of N** experts; the other `N−k` experts are **never touched** | one F64 forward certifies a **data-dependent subgraph**, *not the model*. "This forward matches truth" ≠ "the model matches truth". |
| **Scale** | total weights ≫ active weights (Mixtral 46.7 B total / 12.9 B active; Qwen-MoE 14.3 B / 2.7 B) | full-model F64 = `total × 8` → **373 / 114 / 126 GB** (`src/moe/fixture.rs` table) — infeasible on any project host |
| **New failure mode: routing** | the router's **discrete top-k selection** has no dense analogue; a near-tie in the gate can flip an expert in F32 vs F64 and change the entire downstream computation | correctness now has a **discrete** axis (same experts chosen?) on top of the numeric axis |
| **New failure mode: combine** | `renormalize_topk` (Mixtral) vs sigmoid-gated shared expert (Qwen) vs ungated — convention-dependent weighted sum | the combine convention must match truth, a piece dense models simply do not have |
| **Oracle** | `model.double()` still *defines* truth, but the RAM wall makes it un-runnable at full scale; today's MoE certs use a **HuggingFace block** F64 reference (a *partial* oracle) | the oracle is available in principle, unaffordable in practice → must certify **pieces** the oracle *can* afford |

**Definition we adopt.** A MoE family is **certified at level L** when, on a
representative checkpoint, the MoE-specific correctness obligations are met to
the ADR-004 bar (`< 0.5` + argmax, **unchanged**) across the units that level
covers:

1. **Numeric parity** of each active unit vs F64 (the dense obligation, applied
   per unit).
2. **Routing parity** — Atenia selects the **same top-k expert set** as F64
   (discrete set-equality, a hard gate) with the renormalized combine weights
   matching.
3. **Coverage** — enough units (ideally **all** experts) and enough inputs that
   the certificate is not just one lucky route.

"Certified" without a level qualifier is **forbidden** for MoE: the level names
exactly what was and was not exercised.

---

## FASE 3 — Alternatives analysed (none assumed correct)

Each row: what it proves, what it misses, and its F64 RAM cost (the binding
constraint). Costs are **arithmetic estimates** (`params × bytes`), consistent
with `src/moe/fixture.rs`; not measured runs.

### A) Global logit parity (full-model F64) — the dense gold standard

- **Proves:** everything at once, identical to dense ADR-004.
- **Misses:** nothing *if it runs* — but a single input still only routes one
  way (coverage), so even here you'd sweep inputs.
- **Cost:** `total × 8` → 373 / 114 / 126 GB. **Infeasible** on real weights;
  feasible only on **tiny synthetic** / **reduced-config** fixtures.
- **Status:** already done at fixture/topology scale (tiny e2e 7.5e-8; topology
  1.5e-7). **Cannot scale to real weights.** Verdict: necessary as the *mechanism*
  gate, insufficient as the *real-weight* gate.

### B) Per-expert parity

- **Idea:** certify each expert MLP independently —
  `Atenia.expert_i(x)` vs `F64.expert_i(x)` over an input sweep, on the **real**
  weights, **one expert in F64 at a time**.
- **Proves:** **all** experts match truth (closes the sparsity-coverage hole);
  each piece is dense-ADR-004-shaped.
- **Misses:** expert *composition*, the router, the combine.
- **Cost:** RAM = **largest single expert** in F64, not the whole model.
  Mixtral expert ≈ `3·d_model·d_ff = 3·4096·14336 ≈ 176 M` params → **~1.4 GB
  F64** (✓ on 32 GB). Qwen-MoE expert ≈ `3·2048·1408 ≈ 8.7 M` → **~69 MB** (✓
  trivially). Count: 8×32 = 256 (Mixtral) / 60+×n (Qwen) cheap runs.
- **Verdict:** **the key unlock** — turns an infeasible global F64 into N
  feasible per-expert F64s, and is the *only* alternative that certifies the
  experts the dense single-forward would skip.

### C) Per-router parity

- **Idea:** certify the gate — router logits + **top-k set** + renormalized
  weights, Atenia vs F64, over the input sweep.
- **Proves:** the MoE-specific *decision* dense models lack; cheap (router is one
  `d_model × N` linear → F64 trivial).
- **Misses:** experts, attention, combine arithmetic beyond the weights.
- **Extra metric it enables:** **routing margin** = min gap between the k-th and
  (k+1)-th gate logit; a small margin warns the route is F32/F64-fragile.
- **Verdict:** **mandatory companion** to (B) — together they cover "right
  experts chosen" + "each expert correct".

### D) Per-layer parity (teacher-forced)

- **Idea:** certify each MoE decoder layer `hidden_in → hidden_out` vs F64,
  **feeding the F64 reference's own intermediate activations** as input.
- **Proves:** per-layer correctness; **localises** drift; bounds error
  accumulation layer by layer.
- **Misses:** end-to-end accumulation (covered separately by a budget) — and it
  needs the oracle to dump intermediate F64 activations.
- **Cost:** one layer's experts in F64 (≈ one layer of (B)'s experts) + dense
  attention → feasible.
- **Verdict:** **useful diagnostic / accumulation bound**, optional for the
  headline cert.

### E) Topological parity (reduced-dim, real topology, full F64)

- **Idea:** a fixture with the **real** structure (N experts, top-k, GQA ratio,
  shared expert, MLA) at a **reduced hidden dim**, full-F64 end-to-end.
- **Proves:** routing + combine + **assembly** at real structural complexity,
  fully F64 — the *engine handles the topology*.
- **Misses:** the **specific trained weights** (synthetic/random weights).
- **Cost:** trivial; already committed (`mixtral_scale` 1.639e-7, `qwen_scale`
  1.490e-7, `deepseek_scale` 7.806e-3).
- **Verdict:** **already have it** (= today's "scale-topology" cert). Certifies
  the *machine*, not the *weights*. The argument that it transfers: a correct
  implementation's numeric behaviour is **weight-independent given identical
  dtype + ops** — so topology-F64 + per-expert-real-weight-F64 *together* cover
  both axes. Neither alone is enough.

### Synthesis

No single alternative is sufficient. **A** is infeasible at real scale; **E**
misses the real weights; **B/C/D** each miss the assembly. The honest
certificate is a **composite ladder**: decompose into units the F64 oracle *can*
afford (**B** experts, **C** router, attention), certify each on the **real
weights**, then certify the **assembly** on a full-F64 **topology** fixture
(**E**), and bound end-to-end accumulation (**D**). This is precisely the
`PartialReferenceF64` slot that `src/moe/fixture.rs` reserved but left
**undefined** — this audit defines it.

---

## FASE 4 — MOE-ADR-DRAFT (proposed decision)

> **Draft only (point-in-time).** This block was the original proposal; it was
> **accepted and formalised** as `decisions/ADR-007-moe-certification-ladder.md`
> (see the status banner at the top). Thresholds **reuse** ADR-004 (`< 0.5` +
> argmax) — **no threshold is lowered or changed**.

### Definition of "certified MoE family"

A MoE family is certified at the **highest level whose every obligation passes**
on a representative real checkpoint. Obligations:

| Code | Obligation | Oracle | Gate (ADR-004 bar, unchanged) |
|---|---|---|---|
| **C1** | **per-expert** numeric parity, **all** experts | one-expert F64 on real weights | `max_abs_diff < 0.5` per expert, over the input sweep |
| **C2** | **router** parity | F64 router on real weights | **top-k set equality (hard)**; router-logit `max_abs_diff < 0.5` (informative); **routing-margin** reported |
| **C3** | **attention** parity (MHA/GQA; MLA deferred) | `model.double()` of the attention block (dense, small) | `max_abs_diff < 0.5` + argmax (reuse dense method) |
| **C4** | **assembly / topology** parity | full-F64 reduced-dim **topology** fixture | end-to-end `max_abs_diff < 0.5` + argmax 4/4 (have it) |
| **C5** | **active-path** parity, real weights, canonical input | F64 over the **active subgraph** *or* F32 **cross-framework** (ADR-002 L2) when active-F64 exceeds RAM | `max_abs_diff < 0.5` (+ argmax); cross-framework drift documented if used |

### Certification ladder (escalating, honestly labelled)

| Level | Obligations | Claim |
|---|---|---|
| **L0 Topology-certified** *(have today)* | C4 | "engine reproduces the real routing/topology vs F64" |
| **L1 Component-certified** | C1 + C2 + C3 on **real weights** | "every expert, the router, and attention match truth on the real weights" |
| **L2 Assembly-certified** | L1 + C4 | "the pieces **and** their composition are certified" |
| **L3 Active-path-certified** | L2 + C5 | "the real model, as actually run, matches truth on the tested inputs" |
| **L4 Full-certified** *(dense-equivalent)* | global `model.double()` of the **whole** model | **reserved / unreachable** for >~3 B-active MoE (RAM); **never fabricated**, always listed as pending-with-blocker |

**Honesty rule.** The manifest headline = the **lowest fully-passed level**.
Higher levels are listed `pending` with the **exact** blocker (e.g. "L4: needs
373 GB F64 RAM"). This mirrors the dense numcert discipline (CPU-F32 certified ≠
GPU-TF32 pending) already in `docs/numcert/*.json`.

### Metrics & thresholds (no ADR-004 change)

- Per-expert / attention / assembly / active-path: **`max_abs_diff < 0.5`** +
  argmax — ADR-004 verbatim.
- Router: **discrete top-k set equality is the gate** (a flip is a hard fail);
  router-logit `max_abs_diff` and the **routing-margin** (min k-th↔(k+1)-th gate
  gap) are reported as *fragility* signals, not gates.
- Cross-framework fallback (C5 when F64 won't fit): document the F32-vs-F32 drift
  + require **exact argmax/greedy** (the precedent: DeepSeek scale 7.806e-3 with
  exact argmax, `HANDOFF_MOE_FULL_15.md`).

### Compute cost (arithmetic estimates, 32 GB dev host)

| Unit | RAM (F64) | Runs | Feasible on 32 GB? |
|---|---|---|---|
| Per-expert (C1), Mixtral | ~1.4 GB / expert | 8×32 = 256 | ✅ |
| Per-expert (C1), Qwen-MoE | ~0.07 GB / expert | 60+×layers | ✅ (trivial) |
| Router (C2) | « 1 GB | per layer | ✅ |
| Attention (C3) | dense block, small | per layer | ✅ |
| Assembly/topology (C4) | reduced-dim, full F64 | 1 | ✅ (have it) |
| Active-path (C5), Qwen-MoE | F64 active 2.7 B → ~22 GB; **F32** ~11 GB | 1 | ⚠️ F64 borderline (no headroom) → **F32 cross-framework ✅** |
| Active-path (C5), Mixtral | F64 active 12.9 B → ~103 GB; F32 ~52 GB | 1 | ❌ on 32 GB → needs ≥128 GB **or** F32-cross-framework on a bigger host |
| Full (L4) | total × 8 = 373/114/126 GB | 1 | ❌ (reserved/unreachable) |

### Risks

- **Coverage illusion (HIGH).** Component certs (L1/L2) prove pieces + assembly,
  **not** the real weights' full-scale interaction on every input. Must be
  labelled L2, never "the model is certified".
- **Routing fragility (HIGH).** A near-tie gate can flip an expert F32↔F64 →
  argmax flip downstream. The routing-margin metric surfaces it; a flip on the
  canonical inputs is a hard fail; **other inputs remain uncovered** → explicit
  caveat.
- **Compositionality assumption (MEDIUM).** L2 assumes correct pieces + correct
  assembly ⇒ correct whole. Exact in infinite precision; F32 accumulation across
  many layers can still drift — bounded by C4 on the topology fixture + a
  per-layer accumulation budget (D). Document the bound; don't assume it away.
- **Synthetic ≠ real weights (MEDIUM).** C4 uses random weights → certifies
  *mechanism*, not trained-weight numerics. The transfer argument
  (weight-independence given identical dtype+ops) is sound **only if** C1
  separately certifies the real weights — hence L2 requires **both**.
- **Oracle pinning (LOW).** HF block reference ≠ `model.double()` of the
  assembled model; pin `transformers`/`torch` versions (already practice in the
  dense numcerts).

---

## FASE 5 — Is certifying Mixtral & Qwen-MoE feasible on reasonable hardware?

Host envelope per the docs: **32 GB dev box (RTX 4070 Laptop)**, optionally a
**≥128 GB workstation**. (All numbers arithmetic, `params × bytes`.)

### Qwen1.5-MoE-A2.7B — **best candidate**

- Experts tiny (~8.7 M each → 69 MB F64) → **C1 trivial for all experts** ✓.
- Router/attention (C2/C3) cheap ✓. Topology C4 already certified (1.490e-7) ✓.
- C5 active-path: F64 active 2.7 B → ~22 GB (fits 32 GB **without headroom** →
  borderline `PartialReferenceF64`); **F32 cross-framework ~11 GB → comfortably
  feasible** ✓.
- Already **runs the real 14.3 B model end-to-end** (disk-tier, `MoeRuntime`).
- **Verdict: L2 fully feasible on 32 GB; L3 feasible via F32 cross-framework on
  32 GB.** Qwen-MoE can reach the **strongest** honest level first.

### Mixtral-8x7B

- Experts ~176 M each → ~1.4 GB F64 → **C1 feasible for all 256** ✓ (sequential).
- C2/C3 cheap ✓; C4 already certified (1.639e-7) ✓.
- C5 active-path: F64 active 12.9 B → ~103 GB ❌ on 32 GB; F32 ~52 GB ❌ on 32 GB
  → needs **≥128 GB** or stays **topology+component** only.
- Plus the **data gap**: no local Mixtral-8x7B weights (≈94 GB download).
- **Verdict: L2 feasible on 32 GB** (per-expert + assembly), **L3 needs a bigger
  host or the F32-cross-framework compromise** + provisioning the real weights.

### DeepSeek-MoE — **out of charter** (deferred); noted for completeness only.

### Acceptable compromises

1. **Headline = L2 (component + assembly)** for Mixtral/Qwen-MoE; L3 active-path
   labelled "cross-framework / behavioural" until a ≥128 GB host exists.
2. **F32 cross-framework** (ADR-002 Level 2) substitutes for F64 at C5 when the
   active set exceeds RAM — explicitly weaker, drift documented, **exact argmax
   required** (DeepSeek 7.8e-3 precedent).
3. **All** experts certified (not sampled) at C1 — cheap, removes the sparsity
   blind spot honestly.
4. **L4 (global F64) stays explicitly unreachable** and is labelled so — never
   fabricated.

---

## FASE 6 — Deliverable

### Executive summary

Dense certification works because a dense forward is **complete** — it touches
every weight, so one `model.double()` F64 comparison (`max_abs_diff < 0.5` +
argmax) certifies the model. **MoE violates both pillars:** its full weights do
not fit in F64 RAM (373 / 114 / 126 GB), and a forward only exercises the
**top-k routed** experts, so a single F64 forward certifies a *data-dependent
subgraph*, not the model — and it introduces a **discrete** correctness axis
(did the router pick the same experts?) that dense models simply lack. The
honest, scalable answer is **certification by decomposition**: split the MoE
forward into units the F64 oracle *can* afford — **each expert** (1.4 GB F64 for
Mixtral, 69 MB for Qwen-MoE, one at a time), the **router** (trivial), and
**attention** (dense) — certify each on the **real weights**, certify their
**assembly** on a full-F64 **topology** fixture (already done), and report the
**lowest level fully passed**. This defines the `PartialReferenceF64` strategy
that `src/moe/fixture.rs` reserved but never specified.

### Options evaluated

| Option | Proves | Misses | Scales to real weights? |
|---|---|---|---|
| A — global F64 | everything | (coverage over inputs) | ❌ RAM |
| B — per-expert | all experts correct | composition, router, combine | ✅ (one expert at a time) |
| C — per-router | routing decision + combine | experts, attention | ✅ (trivial) |
| D — per-layer | per-layer + accumulation bound | end-to-end whole | ✅ (one layer at a time) |
| E — topology | routing+combine+assembly mechanism | the **real trained weights** | ✅ (already committed) |

No single option suffices; **B + C + attention + E** (+ D as a bound) compose
into the ladder.

### Recommendation

1. **Adopt the L0–L4 ladder** and the **C1–C5 obligations** as a future
   `ADR-007`, reusing the ADR-004 bar verbatim (no threshold change).
2. **Certify Qwen-MoE first to L2** (it already runs the real model; experts are
   trivially F64) — per-expert (all) + router + attention + assembly, then **L3
   via F32 cross-framework** on 32 GB.
3. **Certify Mixtral to L2** on 32 GB (per-expert + assembly); defer L3 to a
   ≥128 GB host or the cross-framework compromise + real-weight provisioning.
4. **Emit a MoE numcert manifest** mirroring the dense schema
   (`docs/numcert/*.numcert.json`) but with **per-component evidence** + the
   **level label** + every pending level's **exact blocker** — headline = lowest
   fully-passed level.
5. **Extend `src/moe/fixture.rs`** with the new strategy variants
   (`ComponentReferenceF64`, `CrossFrameworkF32`) so the decision stays
   code-encoded and cannot drift — same discipline as the existing
   `recommend_strategy`.

### Risks (carry-forward)

- Coverage illusion (label L2 honestly, never "certified" bare) — **HIGH**.
- Routing fragility (report routing-margin; canonical-input flip = hard fail;
  other inputs uncovered) — **HIGH**.
- Compositionality (bound accumulation via C4 + D, don't assume) — **MEDIUM**.
- Synthetic ≠ real weights (L2 needs **both** C1-real and C4-topology) — **MEDIUM**.
- Oracle/version pinning — **LOW**.

### Roadmap (no code in this audit)

1. **MOE-CERT-1** — define `ADR-007` (ladder + obligations) + extend
   `src/moe/fixture.rs` strategy enum. *Spec + substrate, S.*
2. **MOE-CERT-2** — implement the **per-expert F64 harness** (C1) + **router
   parity + routing-margin** (C2) + attention reuse (C3); certify **Qwen-MoE to
   L1** on real weights. *M.*
3. **MOE-CERT-3** — fold in C4 (have it) → **Qwen-MoE L2** + the **MoE numcert
   manifest**; then **Mixtral L2** (real-weight provisioning permitting). *M.*
4. **MOE-CERT-4** — C5 active-path: **Qwen-MoE L3** via F32 cross-framework on
   32 GB; Mixtral L3 gated on a ≥128 GB host. *M, host-bound.*
5. *(reserved)* **L4** global-F64 — documented unreachable; revisited only if the
   hardware envelope changes. DeepSeek-MoE / MLA remain **out of charter**.

*Audit only — no source/manifests changed, no commits, no execution, no CI.*
