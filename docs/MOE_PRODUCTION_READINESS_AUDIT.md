# MoE Production Readiness Audit

> **Point-in-time audit (post-MLA-3 note).** Snapshot at MOE-PROD-8 /
> RUNTIME-MOE-2. Since then the ADR-007 certification track advanced:
> **Qwen1.5-MoE-A2.7B = MoE-certified L3** and **DeepSeek-V2-Lite (MLA) =
> MoE-certified L3** (real-weight C1+C2+C4+**C5 active-path**, via the MLA-2 disk
> expert-tier; MLA-1/2/3) — so "DeepSeek real run ❌" below is now **✅**. **Mixtral
> = L0** (real weights still not provisioned). `MoE-certified Ln`, not the dense
> ADR-004 `CERTIFIED`; **L4** reserved/unreachable. Productive wiring
> (CLI/Adapter Toolkit) is still pending — the production-distance assessment below
> stands. See `docs/HANDOFF_MLA_3.md`, `docs/MOE_COVERAGE_AUDIT.md`.

**Audit only — no code, no manifests, no commits, no execution.** An honest
distance-to-production assessment for MoE, reconciling the experimental close
(MOE-0..19), the full-path build (MOE-FULL-1..15), the family certs
(QWEN-MOE-CERT-1, MIXTRAL-CERT-1), and the **engine production track
(MOE-PROD-1..8 + RUNTIME-MOE-2-reopened)** that the older `MOE_OVERVIEW.md`
predates. Sources: `docs/MOE_OVERVIEW.md`, `docs/MOE_FULL_PATH_AUDIT.md`, the
`HANDOFF_MOE_FULL_*` / `HANDOFF_MOE_PROD_*` series, `docs/STATUS.md`,
`docs/MODEL_FAMILY_VALIDATION.md`, `docs/FAMILY_COVERAGE_AUDIT.md`.

## ⚠️ Headline correction (read first)

The premise "we're far from productive MoE" is **out of date**. The reality:

- **Qwen-MoE already runs a real, full, 14.3 B model end-to-end.** Qwen1.5-MoE-
  A2.7B (27 GB, 8 shards) loads via `atenia moe-generate` with
  `ATENIA_MOE_EXPERT_TIER=disk` (experts streamed to NVMe, RAM ~3–4 GB) and
  generates **coherent real text** ("The capital of France…"), routing (top-4 of
  60 + shared), manifest-gated, fail-loud opt-in (RUNTIME-MOE-2-reopened;
  MOE-PROD-1..8). It is **slow** (~min/token, disk-tier, I/O-bound) and not
  full-f64-certified at 14 B scale, but it is **real and working**.
- The MoE **block math, full transformer path, generation + KV cache, GQA, MLA,
  residency/tiering, sharded loading** are all **done and certified** (~1e-7…1e-10
  vs HuggingFace f64 on tiny + topology fixtures).
- What is *not* done is **productive integration into the normal `atenia
  generate` runtime** (it lives behind a separate `atenia moe-generate` command +
  opt-in), the **Adapter Toolkit MoE spec**, and a **full-model certification
  methodology** (ADR-004 f64 does not scale to MoE).

So the remaining work is **mostly integration wiring + one certification-
methodology research item**, *not* new core engineering.

## FASE 2 — Capability table (evidence-backed)

Status: ✅ Done · ⚠️ Partial · ❌ Pending. "Done (experimental/controlled)" = works
behind the opt-in MoE runtime, not the dense path.

| Capability | Status | Evidence |
|---|---|---|
| **Detection** | ✅ Done | `moe/detect.rs`, MOE-2; loader fail-loud guard |
| **Metadata plane** | ✅ Done | `moe/data_plane.rs` `MoeWeightMap`, MOE-9 |
| **Tensor binding** (classic + packed) | ✅ Done | `moe/binding.rs`, MOE-10/15; `moe_packed_experts_test` |
| **Real layer** (router+experts+shared) | ✅ Done | `moe/layer.rs` `RealMoeLayer::forward_auto`, MOE-11/17/18 |
| **Stack** (multi-layer) | ✅ Done | `moe/stack.rs`, MOE-12 |
| **Graph integration** | ✅ Done | MOE-FULL-4 `NodeType::MoeRealLayerReference` (wraps the certified block) |
| **Decoder layer** (norm+attn+resid+MoE) | ✅ Done | MOE-FULL-5 `moe/decoder_layer.rs` |
| **Full forward** | ✅ Done | MOE-FULL-6 `moe/full_forward.rs` — **7.451e-08** vs HF f64 |
| **Logits parity** | ✅ Done | MOE-FULL-6/15 — Mixtral 1.6e-7, Qwen 1.5e-7, DeepSeek 7.8e-3 (argmax exact) |
| **Generation** (prefill+decode) | ✅ Done | MOE-FULL-7 `moe/generate.rs` — greedy ids exact, **4.470e-08** per-step |
| **KV cache** | ✅ Done | MOE-FULL-7 (per-layer harvest + re-inject); MLA latent-KV is the exception |
| **Productive loader** (sharded + residency) | ✅ Done (controlled) | MOE-PROD-1 `MoeWeightSource`; MOE-PROD-2 disk tier (57 GB→~3 GB); bit-exact |
| **GQA / attention bias / MLA** | ✅ Done | MOE-FULL-9 (GQA 5.96e-08), -11 (Qwen QKV bias), -12 (DeepSeek MLA 9.99e-06) |
| **Real full-model run** | ⚠️ Partial | **Qwen-MoE ✅ real 14.3 B e2e** (RUNTIME-MOE-2-reopened); **DeepSeek-V2-Lite ✅ real e2e** (MLA-2 disk tier, C5 active-path L3 — post-audit); **Mixtral-8x7B ❌** (no local weights) |
| **Adapter Toolkit** | ❌ Pending | no MoE family/tensor spec in `src/adapter_toolkit/` (BLOCKER) |
| **Runtime integration** (normal `atenia generate`) | ⚠️ Partial | `MoeRuntime` works, but only via the separate controlled path; dense runtime is MoE-unaware |
| **CLI integration** | ⚠️ Partial | `atenia moe-generate` exists (separate, token-id + text); not `atenia generate` |
| **Fail-loud removal** | ⚠️ Partial (by design) | lifted **only** inside the controlled opt-in path; the dense loader still refuses MoE intentionally |
| **Certification (real full models)** | ❌ Pending | block + scale-**topology** certified; full-weight f64 infeasible (ADR-004 doesn't scale) — needs the MOE-1 partial-reference methodology |

## FASE 3 — Answers

**1. What % of the MoE work is really done?**
Split by layer:
- **Core MoE engineering (block math, routing, conventions, packed/classic,
  GQA, MLA, KV cache, residency/tiering, sharded load): ~90% done.** This is the
  hard part and it is largely finished + certified.
- **Real-model execution: ~60%** — Qwen-MoE proven on a real 14.3 B model;
  Mixtral/DeepSeek engine-ready but no real-weight run.
- **Productive integration (dense-runtime routing, ATK MoE spec, fail-loud lift,
  config wiring): ~30%** — a controlled path exists, but it is a separate CLI.
- **Full-model certification methodology: ~20%** — topology-certified only;
  the scale-f64 problem is open (possibly permanently, per ADR-004).
- **Weighted overall: ~70% done**, concentrated remaining work in integration +
  a cert-methodology research item, not new capability.

**2. The most important blocker.**
**The full-model certification methodology** (ADR-004 f64 is infeasible at
14B–47B). Everything else remaining is engineering-known wiring; *this* is the
one open *research* question, and it gates any honest "production-supported,
certified" claim. (The nearest *integration* blocker is the **Adapter Toolkit MoE
spec** + the dense-runtime routing/fail-loud-lift, but those are known work.)

**3. What's missing to run Mixtral from the normal runtime?**
- (a) **Real Mixtral-8x7B weights provisioned** + a real end-to-end run (the
  engine path is ready — sharded loader, disk residency, all three layouts
  certified; this is a **data/execution gap, not code**).
- (b) **Dense-runtime integration**: an Adapter Toolkit Mixtral-MoE spec, the
  dense loader's fail-loud lifted behind the opt-in, and `atenia generate`
  routing MoE checkpoints to `MoeRuntime` (today only `atenia moe-generate`).
- (c) A **scale-cert datapoint** on the real weights (topology already certified).

**4. What's missing to run Qwen-MoE from the normal runtime?**
**Less than Mixtral — it already runs the real full model**, just via the
separate `atenia moe-generate` + opt-in. To reach the *normal* runtime:
- (a) **Adapter Toolkit Qwen-MoE spec** (experts, top-k, shared-expert sigmoid
  gate, `norm_topk_prob`, router name).
- (b) **Dense loader fail-loud lift + routing** so `atenia generate` dispatches
  detected MoE to `MoeRuntime`.
- (c) **Performance** is the real UX gap: ~min/token on disk-tier (I/O-bound; the
  measured lever is AV-exclusion + fewer tier bytes, not compute) — works but
  **not interactive**.
- (d) Optional: full-scale certification (methodology open).

**5. Which parts are real engineering vs only wiring?**
- **Real engineering — DONE:** MoE block math, sparse top-k routing, packed/
  classic binding, HF conventions, GQA, MLA + interleaved RoPE, KV cache,
  expert residency/tiering (RAM/NVMe), expert cache, sharded loading, bf16/int8
  expert tiers.
- **Real engineering — REMAINING:** the **full-model certification methodology**
  (partial/sub-reference F64 at scale) — the only genuinely *new* intellectual
  work left.
- **Only wiring (known, incremental):** Adapter Toolkit MoE family/tensor spec;
  dense-loader fail-loud lift behind the opt-in; `atenia generate` → `MoeRuntime`
  routing; folding `moe_config` into the productive config path.

## FASE 4 — Gap analysis, risks, complexity, roadmap

### Current state
The MoE substrate is **complete and certified**; one family (Qwen-MoE) runs a
**real full model** end-to-end through a controlled, opt-in path; the dense
runtime remains intentionally MoE-unaware.

### Gap analysis (to "productive from the normal runtime")
| Gap | Type | Complexity | Confidence |
|---|---|---|---|
| Adapter Toolkit MoE spec | wiring | **M** | high (data exists in `moe_config`/`family.rs`) |
| Dense fail-loud lift + `generate`→`MoeRuntime` routing | wiring | **M** | high (runtime exists) |
| Productive config parsing (MoE fields) | wiring | **S** | high (`moe_config` sidecar exists) |
| Real Mixtral-8x7B run | data/exec | **S-code / L-infra** | medium (needs ~94 GB weights + slow disk-tier) |
| Full-model certification methodology | **research** | **L–XL** | **low** (ADR-004 doesn't scale) |
| Performance (interactive speed) | engineering | **L** | low (measured I/O-bound; AV + fewer bytes are the levers; still slow) |

### Risks
- **Certification honesty.** Without a scale-cert methodology, productive MoE can
  only be claimed *behaviourally validated*, not *certified* — must not be
  over-claimed (same discipline as Phi/Gemma).
- **Fail-loud regression.** Lifting the dense guard risks silently accepting MoE
  on the default path — must stay strictly behind the opt-in + tested.
- **Performance perception.** Disk-tier min/token is not interactive; shipping it
  on the normal runtime invites "MoE is broken/slow" reports unless gated + documented.
- **Scope creep to DeepSeek-V3** (FP8 + MLA + huge) — explicitly out of charter.

### Recommended roadmap (if/when MoE is prioritised)
1. **MOE-INTEGRATE-1** — Adapter Toolkit MoE spec + productive config wiring
   (Qwen-MoE first; the family that already runs). *Wiring, M.*
2. **MOE-INTEGRATE-2** — dense `generate` → `MoeRuntime` routing behind the
   opt-in (fail-loud lifted only on the validated path). *Wiring, M.*
3. **MOE-REAL-MIXTRAL** — provision + run real Mixtral-8x7B (execution/data).
4. **MOE-CERT-SCALE** — the partial/sub-reference F64 methodology (research, L–XL)
   — the gate for any "certified" claim.
5. *(separate, ongoing)* MoE performance (AV-exclusion + fewer tier bytes; already
   scoped) — needed before MoE is *interactive*, not before it is *correct*.

## FASE 5 — CERTIFY-BREADTH-2 vs MoE productionisation

| Dimension | **A) CERTIFY-BREADTH-2** | **B) MoE productionisation** |
|---|---|---|
| Effort | ~2–3 h (Gemma-3-1B now, Gemma-2-2B after freeing RAM) | weeks (integration) + open research (scale cert) |
| New code | none (run committed harness) | ATK MoE spec + routing + fail-loud lift + cert methodology |
| Risk | low (cheap to find a >0.5 drift; honest finding either way) | medium-high (cert methodology unsolved; perf non-interactive; fail-loud regression surface) |
| Value delivered | **+2 certified dense families** (Gemma-2, Gemma-3) — raises *trustworthy* coverage of already-supported, fast, interactive families | a whole model **class**, but **slow** (min/token), **uncertified at scale**, opt-in only |
| Certainty of payoff | high | medium (correctness high; "productive + certified + fast" uncertain) |
| Strategic fit | closes the audit's #1 gap (functional ≫ certified) with existing infra | high ceiling, but better as a dedicated multi-milestone series |

**Verdict: A) CERTIFY-BREADTH-2 delivers more strategic value *today*.** It is
cheap, certain, needs no new code, and directly fixes the headline weakness
(only 2 of 7 dense families certified) on families that are **fast and
interactive**. MoE has a higher ceiling but its remaining cost is dominated by an
**unsolved scale-certification methodology** and **non-interactive performance** —
so it is a deliberate, sequenced *bet*, not a quick win. The right order is:
**finish dense certification (CB-2) first, then open a dedicated MoE-INTEGRATE
series** (the engine is ready; the work is wiring + a cert-methodology research
item), keeping MoE honestly labelled *experimental/behavioural* until the scale
cert exists.

## Executive summary

MoE in Atenia is **far more complete than "MOE-FULL-1..6" implies**: the block
math, full transformer path, generation, GQA/MLA, KV cache, residency/tiering and
sharded loading are **done and certified**, and **a real 14.3 B Qwen-MoE already
generates coherent text end-to-end** behind a controlled opt-in path (slow,
disk-tiered, not scale-certified). The remaining distance to "productive from the
normal runtime" is **~30% — mostly integration wiring** (Adapter Toolkit MoE
spec, dense-loader fail-loud lift, `generate`→`MoeRuntime` routing) **plus one
open research item** (full-model certification methodology, since ADR-004 f64
does not scale) **and a performance gap** (min/token, I/O-bound — works but not
interactive). Mixtral is engine-ready but unrun (no local weights); Qwen-MoE is
the closest to productive.

- **MoE completeness: ~70% overall** (core engineering ~90%; productive
  integration ~30%; scale certification ~20%).
- **Top blocker: the full-model certification methodology** (research; ADR-004
  doesn't scale). Nearest integration blocker: the Adapter Toolkit MoE spec +
  dense-runtime routing.
- **Recommendation: do CERTIFY-BREADTH-2 first** (cheap, certain, +2 certified
  interactive dense families), **then** open a dedicated **MOE-INTEGRATE** series
  (engine is ready; remaining work is wiring + the cert-methodology research),
  keeping MoE labelled experimental/behavioural until scale cert exists.

*Audit only — no source/manifests changed, no commits, no execution.*
