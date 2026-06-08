# POST-MIXTRAL-L3-ROADMAP-AUDIT (analysis + decision only)

**Audit only — NO code, NO runtime/test change, NO downloads, NO certification runs.**
Goal: consolidate the certification state after MIXTRAL-CERT-3 and pick the next
technical frontier for Atenia. Language: doc in English; final user answer in Spanish.

## FASE 1 — Confirmed current state

Three **real** MoE families are now at **MoE-certified L3 (active-path-certified)** —
real full forward vs an external float64 reference computed **one decoder layer at a
time over the active subgraph** (never the whole model in F64), gated at end-to-end
`max_abs_diff < 0.5` (ADR-004 bar, unchanged) + exact per-position argmax + determinism:

| Family | Level | C5 active-path metric | Milestones | Manifest |
|---|---|---|---|---|
| **Qwen1.5-MoE-A2.7B** | **MoE-certified L3** | active-path (real 14.3B, disk-tier) | MOE-CERT-2/3/4 | `docs/numcert/qwen1.5-moe-a2.7b.moecert.json` |
| **DeepSeek-V2-Lite (MLA)** | **MoE-certified L3** | `2.587e-5` < 0.5, argmax 4/4, det. | MLA-1/2/3 | `docs/numcert/deepseek-v2-lite.moecert.json` |
| **Mixtral-8x7B-v0.1** | **MoE-certified L3** | `3.185e-4` < 0.5, argmax 4/4, det. | MIXTRAL-CERT-1/2/3 | `docs/numcert/mixtral-8x7b-v0.1.moecert.json` |

Invariants held: **none is dense ADR-004 `CERTIFIED`**; **L4 (global F64) reserved /
unreachable** on a 32 GB host; MoE remains opt-in / experimental (not wired into the
productive CLI / Adapter Toolkit). ADR-007 is now validated across **three distinct MoE
shapes**: classic softmax + renorm (Mixtral), sigmoid-gated shared expert (Qwen-MoE),
and MLA + softmax routing (DeepSeek-V2-Lite). MLA, disk-tier residency, and the
one-layer-at-a-time F64 + active-path C5 harnesses are all proven and ~80% reusable.

**Real pending levels:** L4 for all three (reserved/unreachable, never fabricated); no
real MoE family is production-wired.

## FASE 2 — Documentation sync (done in this milestone)

Corrected stale "Mixtral L0/L1/L2 / C5 pending / only-two-families-L3" wording in the
**strategic** docs; historical handoffs/feasibility docs kept, with banners:

- `docs/MOE_OVERVIEW.md` — header now "three families L3" (post-MIXTRAL-L3).
- `docs/MOE_COVERAGE_AUDIT.md` — top banner + coverage matrix rows (renorm top-k, GQA+MoE
  → L3), family table (Mixtral L3 + DeepSeek-V2-Lite L3 row), %-certified (1→3 of ~12).
- `docs/decisions/ADR-007-moe-certification-ladder.md` — status snapshot + consequences
  now read "Mixtral = L3; all three real families L3".
- `docs/STATUS.md` — MIXTRAL-CERT-1/2 bullet marked superseded by MIXTRAL-CERT-3 → L3.
- `docs/MIXTRAL_L3_FEASIBILITY.md`, `docs/MIXTRAL_DATA_READY.md` — "RESOLVED" banners.
- (`docs/STATUS.md`, `MODEL_FAMILY_VALIDATION.md`, `FAMILY_COVERAGE_AUDIT.md`,
  `numcert/mixtral-8x7b-v0.1.moecert.json` were already set to L3 in MIXTRAL-CERT-3.)

## FASE 3 — Remaining frontiers

1. **Modern routing (DeepSeek-V3 family).** sigmoid scoring, **aux-loss-free**
   bias-corrected top-k, **group-/node-limited** routing, `routed_scaling_factor`. The
   router is today **softmax/sigmoid top-k only** — this is the single hardest gap.
2. **DeepSeek-V3 L0 mechanism.** A reduced-dim V3-topology fixture certified vs HF f64
   (the `mixtral_scale` pattern) — depends on (1).
3. **Adapter Toolkit declarative MoE onboarding.** Turn the per-family Rust assembly
   (Mixtral / Qwen-MoE / DeepSeek) into a declarative family spec so DBRX, Grok, OLMoE,
   Llama-4, GLM become config, not code.
4. **C5 multi-input robustness.** Today C5 uses one canonical probe per family; multiple
   diverse probes (incl. longer seq / different routed-expert sets) harden the active-path claim.
5. **Q-LoRA / adapter-on-MoE.** PEFT over the routed experts.
6. **FP8 MoE path.** FP8 expert tier + compute (frontier inference dtype).
7. **GGUF MoE.** Load MoE from GGUF (ergonomics/distribution, not architecture).
8. **Productive wiring (CLI/UX).** Lift MoE out of opt-in/experimental into the
   productive loader + `generate` (currently fail-loud).
9. **Disk-tier persistence / bf16 tier.** Largely **DONE** (proven in MIXTRAL-CERT-3:
   persistent bf16 tier + warm reconstruct). Remaining: polish + bounded-cache default.
10. **Performance.** CPU forward throughput; the bounded-cache disk-tier forward is slow.
11. **Dense family breadth.** More dense families (orthogonal to MoE).
12. **Tokenizer / runtime polish.**

## FASE 4 — Impact evaluation

| Frontier | Strategic impact | Effort | Risk | Reuse of current work | Real evidence on THIS notebook? | Unlocks future families? | Product/user value |
|---|---|---|---|---|---|---|---|
| **Modern routing → V3 L0 (1+2)** | **Very high** (the entire 2025–26 cluster: V3/R1, Kimi-K2, GLM-4.5, Ling) | Med | Med (new routing math; topology-verifiable) | **High** (ADR-007 ladder, scale-cert, disk-tier, active-path C5) | **Yes** — reduced-dim V3 fixture, no huge download | **Yes** — gates the whole frontier | Indirect (capability) |
| Adapter Toolkit declarative (3) | High (long-term multiplier) | Med-High | Low-Med | High (3 reference families exist) | Yes (regenerate existing families) | Yes (onboarding cost ↓) | Indirect |
| C5 multi-input (4) | Medium (hardening) | Low | Low | Very high (same harnesses) | **Yes** (re-run on 3 L3 families) | No | Low (rigor) |
| Q-LoRA on MoE (5) | Medium-High | High | Med | Medium | Partial | No (training-time) | High (fine-tuning) |
| FP8 MoE (6) | Medium | High | High (numerics) | Low-Med | Partial | Partial | Med |
| GGUF MoE (7) | Medium | Med | Low | Med | Yes | No (ergonomics) | High (distribution) |
| Productive CLI wiring (8) | High (turns research → product) | Med-High | Med | High | Yes | No | **Very high** |
| Disk-tier polish (9) | Low (mostly done) | Low | Low | — | Yes | No | Low |
| Performance (10) | Medium | Med-High | Med | Med | Yes | No | Med |
| Dense breadth (11) | Low-Med | Low-Med | Low | Med | Yes | No (dense) | Med |
| Tokenizer/runtime polish (12) | Low | Low | Low | — | Yes | No | Low |

## FASE 5 — Primary recommendation

**Recommendation: (B+C) Modern routing → DeepSeek-V3 mechanism L0** — implement the V3
routing primitives (sigmoid scoring, aux-loss-free bias-corrected top-k, group-/node-limited
routing, `routed_scaling_factor`) as a routing variant and certify them on a reduced-dim
V3-topology fixture vs HF float64 (the proven `mixtral_scale` L0 pattern).

Why this over the others, now:

- **It is the one gap that blocks the whole 2025–2026 frontier.** Atenia's router is
  softmax/sigmoid-top-k only; every current frontier MoE (DeepSeek-V3/R1, Kimi-K2,
  GLM-4.5, Ling, and most new releases) shares the aux-loss-free + group-limited template.
  Coverage of that cluster is ~0% today. Nothing else unlocks more families per unit effort.
- **Maximum reuse, low provisioning cost.** The ADR-007 ladder, the scale-topology cert,
  the disk-tier, and the active-path C5 harness all transfer; a reduced-dim fixture means
  **real, reproducible evidence on this 32 GB notebook** with **no 600 GB+ download** (real
  V3 is impractical locally — L0 mechanism is the honest, feasible first rung, exactly as
  Mixtral/DeepSeek started).
- **It is a dependency, not a parallel option.** DeepSeek-V3 L0 (B) *is* the modern-routing
  work (C); doing them together is the natural unit.
- **Versus the alternatives:** Adapter Toolkit (A) is a strong fast-follow multiplier but
  adds **no new architectural coverage** — the routing gap still blocks the frontier, so it
  is better sequenced *after* the routing primitive exists (then V3 becomes declarative).
  C5 multi-input (D) and disk-tier polish (G) are valuable hardening but incremental.
  GGUF MoE (E) is ergonomics. Productive CLI wiring (F) has the highest *direct user value*
  and is the strongest "second track", but wiring an experimental, single-probe,
  CPU-only MoE into product is premature before the frontier routing exists — recommend it
  as the milestone **after** routing+toolkit, when there is a product-worthy MoE surface.

## FASE 6 — Next 3 milestones

### M1 — MOE-V3-ROUTE-1: DeepSeek-V3 modern-routing mechanism → L0
- **Objective:** add the V3 routing primitives and certify the mechanism at L0 (topology)
  vs HF float64.
- **Scope:** sigmoid expert scoring; aux-loss-free **bias-corrected** top-k selection
  (selection uses score+bias, combine uses the raw score); **group-limited / node-limited**
  routing; `routed_scaling_factor`. A reduced-dim DeepSeek-V3-topology fixture
  (mixtral_scale-style) + an ADR-007 C4/L0 scale-cert test.
- **No-hacer:** no real V3 weights (impractical locally); no MTP head; no FP8; no
  productive-loader wiring; do not touch Qwen/DeepSeek-V2/Mixtral certs or the Adapter
  Toolkit; do not lower the ADR-004 gate.
- **Validation:** topology `max_abs_diff` vs HF f64 under the scale-cert bound; top-k **set
  equality** vs the reference router; argmax exact; deterministic; `cargo test --lib`.
- **Expected result:** DeepSeek-V3 routing primitives exist and are **MoE-certified L0
  (mechanism)**; the V3/R1/Kimi/GLM cluster becomes mechanically reachable. Not dense
  CERTIFIED; not L1–L4.

### M2 — MOE-ATK-DECL-1: declarative Adapter Toolkit MoE onboarding
- **Objective:** express the certified MoE families (Mixtral, Qwen-MoE, DeepSeek-V2,
  +V3-routing from M1) as a **declarative family spec** so new families are config, not Rust.
- **Scope:** an ATK MoE schema (expert layout, router type, shared-expert, GQA, routing
  variant, tier policy) + a generator that reproduces the existing per-family assembly.
- **No-hacer:** no new architectures certified here; no numeric-threshold change; no
  download; keep the existing hand-written paths until the declarative output is proven
  equivalent.
- **Validation:** regenerate each existing family from the spec and assert **byte/numerically
  identical** assembly + unchanged C1/C2 results; `cargo test --lib`.
- **Expected result:** adding DBRX/Grok/OLMoE/Llama-4/GLM becomes a declarative entry; the
  per-family Rust cost drops to ~0.

### M3 — MOE-C5-MULTIINPUT-1: C5 multi-input robustness across the three L3 families
- **Objective:** strengthen the active-path certificate from one canonical probe to a small
  **diverse probe set** (different lengths, different routed-expert coverage) on Qwen-MoE,
  DeepSeek-V2-Lite, and Mixtral.
- **Scope:** extend the F64 reference generator + C5 harness to N probes; record worst-case
  over the set; reuse the disk-tier + bounded-cache pattern proven in MIXTRAL-CERT-3.
- **No-hacer:** no new families; no L4; no threshold change; no runtime change (harness +
  reference only); keep within the 32 GB envelope (bounded expert cache).
- **Validation:** for every probe, `max_abs_diff < 0.5` + argmax exact + determinism; the
  manifests record `c5_inputs: N` with the worst-case metric.
- **Expected result:** the three L3 certificates are hardened (active-path proven on a probe
  set, not a single input) — still L3, still not L4, but materially more robust.

## FASE 7 — This document

`docs/POST_MIXTRAL_L3_ROADMAP_AUDIT.md` (this file).

## FASE 8 — Commit / push / CI

Docs-only change → single commit + push. CI may run (repo CI builds lib + runs lib tests);
no `src/`/test change, so a docs-only commit is expected to pass trivially or be skipped by
path filters.
