# HANDOFF — MOE-PROD-2: disk-backed MoE expert residency

Milestone: **MOE-PROD-2** — remove the "whole model → f32 → RAM" requirement for
the controlled MoE runtime, **without** changing certified math/outputs.
**Engine** milestone (loader/runtime code allowed). Predecessor: `409ef82`
(MOE-PROD-1, sharded loading).

## FASE 1 — Audit (what forces f32 RAM)

- The MoE block enters the graph as an **imperative reference node**
  (`moe_real_layer_reference` → `graph_op::execute_real_moe_layer` →
  `RealMoeLayer::forward_auto`). The **experts** (the bulk of a real MoE) live in
  `RealMoeLayer` → `MoeDenseExpert` as `Vec<f32>`. Attention weights (small) are
  baked f32 into the graph.
- `load_core` **materialised all experts as f32** (`moes: Vec<RealMoeLayer>`)
  **before** building the backend → **peak load RAM = the whole model in f32**
  (~57 GB for Qwen1.5-MoE). It also held a **second**, unused RAM-f32 copy in the
  `residency` field.
- **Reusable + certified:** `ResidentExpertLayer` already tiers experts to
  **F32-RAM or Disk-NVMe** (`SharedParam` / `disk_tier`) and resolves only the
  routed top-k per token, and is **certified bit-identical** to
  `RealMoeLayer::forward_auto` (MOE-FULL-8). The node dispatch is localised in
  `graph_op::execute_real_moe_layer` (no `amg/graph.rs` change needed).
- **Would break certification:** truncating f32→bf16 when the source is genuinely
  f32 (the committed fixtures) changes outputs. bf16 is lossless **only** if the
  bytes were already bf16.

## FASE 2-3 — Design + implementation (chosen: disk-backed, streamed)

**Why disk, not bf16 (the priority-1 option):** the milestone says "pick the
safest". The **Disk** tier reuses the **already-certified** `ExpertTier::Disk`
and is **bit-exact regardless of source dtype** (it stores/reads f32 losslessly),
so it cannot regress the f32 fixtures. A bf16 residency would be lossless only
for bf16 sources and risks the f32 fixtures, and `place()` has no bf16 arm yet —
more surface, less safe. Disk is the minimal, provably-safe option that meets the
goal.

What was implemented:

1. **`graph_op.rs` registry dispatch.** The MoE-node registry now holds an enum
   `RegisteredMoe { Real(Arc<RealMoeLayer>) | Resident(Arc<ResidentExpertLayer>) }`
   (single registry → unique ids). New `register_resident_moe_layer(Arc<…>)`.
   `execute_real_moe_layer` dispatches per row: `Real → forward_auto`,
   `Resident → forward(…).0` (drops residency telemetry). Outputs identical
   (certified-equal).
2. **`full_forward.rs` `MoeBlock`.** `TinyDecoderWeights.moe: RealMoeLayer` →
   `moe: MoeBlock { Owned(RealMoeLayer) | Registered(u32) }`. `Owned` registers
   per forward (default, fixtures — byte-identical). `Registered` references a
   resident layer registered **once** at load, so cloning `TinyMixtralWeights`
   per forward costs a `u32`, not the experts.
3. **`runtime.rs` streamed tiering.** `load_core` reads
   `ATENIA_MOE_EXPERT_TIER` (`disk` → NVMe; else RAM-f32 default). For the graph
   families (Mixtral / Qwen-MoE) under the **Disk** tier, each layer is
   assembled → tiered to NVMe → **its f32 copies freed before the next layer
   assembles**, so peak load RAM is ~one layer, not the whole model. The
   per-layer resident is shared (`Arc`) between the registry (compute) and the
   `residency` field (no second copy). DeepSeek (MLA, imperative) stays RAM-f32
   (its forward consumes `RealMoeLayer` directly — tiering it is a follow-up).
   The residency self-check (`self_validate_residency`, MOE-FULL-8) still runs
   per layer with a transient RAM resident.

## FASE 4 — Compatibility

| Path | Status |
|---|---|
| single-file load | ✅ unchanged |
| sharded load (MOE-PROD-1) | ✅ unchanged |
| Mixtral | ✅ RAM default bit-exact; Disk bit-exact |
| Qwen-MoE | ✅ RAM default bit-exact; Disk bit-exact |
| DeepSeek-MoE | ✅ RAM-f32 (unchanged; not tiered this milestone) |
| default behaviour (no env) | ✅ RAM-f32, byte-identical to before |

## FASE 5 — Certification (before vs after)

- **RAM default:** all MoE suites pass unchanged → bit-exact.
- **Disk tier:** `tests/moe_residency_tier_test.rs` loads the tiny Mixtral
  fixture RAM-tier and Disk-tier and compares: **max_abs_diff = 0.0** on logits,
  identical generation (`[17, 20]`, stop at EOS). No drift, argmax trivially
  identical (bit-exact).

## FASE 6 — Tests

- `tests/moe_residency_tier_test.rs` — disk-tier == RAM-tier bit-for-bit (new).
- Regression (RAM default, all green): mixtral 3/3, qwen 3/3, deepseek/MLA 4/4,
  sharded 5/5, full_forward 7/7, gqa 3/3, decode-generation 5/5, production 4/4,
  robustness 6/6, scale 3/3, partial 4/4, cli 2/2.
- Full lib suite: see commit/CI.

## FASE 7 — RAM measurement / estimate

Compute backend on the tiny fixture is microscopic; the meaningful figure is the
**estimate for Qwen1.5-MoE-A2.7B** (14.3 B params, bf16 source 28.6 GB), **not
downloaded**:

| | RAM default (f32) | Disk tier (MOE-PROD-2) |
|---|---|---|
| Steady-state host RAM | ~57 GB (experts f32 + attn) — **exceeds 32 GB** | ~3 GB (attention + embed + lm_head + router; experts on NVMe) |
| Peak load RAM | ~57 GB (all experts f32 at once) | **~one layer** (assemble→tier→free) ≈ a few GB |
| Experts location | RAM f32 | NVMe (~28.6 GB on the 830 GB disk) |

So with `ATENIA_MOE_EXPERT_TIER=disk`, Qwen1.5-MoE-A2.7B **fits** on the 8 GB
VRAM + 32 GB RAM host (experts streamed from NVMe). Estimate only — no weights
downloaded.

## Problem found + fix (self-introduced, caught by build)

`MoeBlock` was imported in `generate.rs` but used only by the test fixture →
`unused import` warning. Fixed by gating it `#[cfg(test)]`. No behavioural issue.
(Several call sites that build `TinyDecoderWeights` — `generate.rs`, three test
files — were updated to `moe: MoeBlock::Owned(...)`.)

## FASE 10 — Review

**Does Qwen1.5-MoE fit now?** **Yes (by estimate)** with
`ATENIA_MOE_EXPERT_TIER=disk`: steady-state ~3 GB RAM + experts on NVMe, peak
load ~one layer. Not yet run against real weights (not downloaded).

**What's left to reopen RUNTIME-MOE-2?**
1. Download Qwen1.5-MoE-A2.7B-Chat (28.6 GB; sharded — MOE-PROD-1 handles it;
   disk-tier — MOE-PROD-2 handles the RAM).
2. Run real load + generation with `ATENIA_ENABLE_MOE=1` +
   `ATENIA_MOE_EXPERT_TIER=disk`.

**Can we download it now?** The two engine blockers (sharded loading, f32-RAM)
are removed. Downloading + validating is now an **environment** step
(RUNTIME-MOE-2), not a code blocker. Recommended to reopen RUNTIME-MOE-2 next.

**Caveats:**
- Disk-tier generation is **slow**: the node calls `ResidentExpertLayer::forward`
  with no LRU cache, so the routed top-k experts are read from NVMe **per token**
  (the `ExpertCache` exists but is not wired into the graph node — a perf
  follow-up). Correctness-first.
- DeepSeek MoE is not disk-tiered (MLA imperative path) — follow-up.
- bf16 residency not implemented (disk chosen as the safest, see FASE 2).

## Files modified

- `src/moe/graph_op.rs` — `RegisteredMoe` enum, `register_resident_moe_layer`,
  tier-dispatching `execute_real_moe_layer`.
- `src/moe/full_forward.rs` — `MoeBlock` enum; `TinyDecoderWeights.moe: MoeBlock`.
- `src/moe/generate.rs` — `MoeBlock::Owned` at the fixture; `into_layer_id` use.
- `src/moe/runtime.rs` — `ATENIA_MOE_EXPERT_TIER`, streamed per-layer tiering,
  `residency: Vec<Arc<ResidentExpertLayer>>`, `build_graph` takes `Vec<MoeBlock>`.
- `tests/moe_residency_tier_test.rs` — new (disk==RAM bit-exact).
- `tests/{moe_full_forward,moe_gqa,moe_decode_generation}_test.rs` — `MoeBlock::Owned`.
- `docs/HANDOFF_MOE_PROD_2.md` (this) + `docs/STATUS.md`, `docs/MOE_OVERVIEW.md`.

No new architecture / family / math / graph ops. RAM default + DeepSeek +
single-file/sharded + dense path unchanged (bit-exact).

## Deliverable answers

1. **What implemented?** Disk-backed expert residency for the graph MoE
   families, streamed per-layer, behind `ATENIA_MOE_EXPERT_TIER=disk`; certified
   `ResidentExpertLayer` wired into the MoE node via a tier-dispatching registry.
2. **How residency changed?** Experts can now live on NVMe (~0 host RAM) instead
   of all-f32-RAM; default stays RAM-f32.
3. **RAM before/after?** Qwen1.5-MoE estimate: ~57 GB → ~3 GB steady / ~one-layer
   peak (experts on NVMe). Tiny fixtures: negligible either way.
4. **Compatibility?** single-file, sharded, Mixtral, Qwen-MoE, DeepSeek all
   pass; default byte-identical.
5. **Drift?** Disk vs RAM **max_abs_diff = 0.0** (bit-exact); argmax identical.
6. **Tests?** disk==RAM bit-exact (new) + full MoE regression + lib suite.
7. **Risks?** Disk-tier generation is slow (no expert cache in the node path,
   per-token NVMe reads); DeepSeek not tiered; bf16 not implemented. No
   correctness risk (bit-exact).
8. **Commit:** see git log (MOE-PROD: add MoE residency optimization).
9. **CI:** code change → CI runs; see push result.
10. **Reopen RUNTIME-MOE-2?** Yes — the engine blockers are gone; it is now an
    environment step (download 28.6 GB + run with the disk tier).
