# MLA-2 — Disk/bf16 expert-tier for the experimental MLA (DeepSeek) forward — AUDIT

> **UPDATE (MLA-2 implemented).** Option A was implemented as designed below.
> `DeepseekFfn::Moe` now holds `Arc<ResidentExpertLayer>` and `layer_step` calls
> the uncached `forward`; `runtime.rs` honours the env tier for DeepSeek (default
> `Ram`, `disk` via `ATENIA_MOE_EXPERT_TIER=disk`). Disk tier proven **bit-identical**
> to RAM on the MLA-0 fixture (both 9.072e-5 vs HF). RAM ~58.5 GB → ~4 GB. The
> minimal path uses the **ephemeral f32** disk tier (~59 GB NVMe); bf16-on-disk is a
> later persistent-tier optimisation. **C5 not run; DeepSeek-V2-Lite stays L2.** See
> `docs/HANDOFF_MLA_2.md`.

**Status: AUDIT ONLY. No code changed, no manifest changed, no certification run,
no commit.** This document designs, with evidence, how to let the experimental
DeepSeek MLA forward stream its experts from a disk/bf16 residency tier (as Qwen
already does) so that the C5 active-path forward of the **real DeepSeek-V2-Lite**
fits this host — the blocker found in MLA-1 / C5 FASE 1.

Goal of MLA-2 (a *later* milestone): unblock C5 → L3 **without lowering thresholds,
without numeric loss, and without touching the productive runtime / Qwen / Mixtral
paths.** This audit does not implement it.

---

## 1. Confirmed root cause (FASE 1)

C5 requires running Atenia's **real full forward** on DeepSeek-V2-Lite and gating it
against a one-layer-at-a-time F64 reference (the Qwen C5 pattern,
`tests/moe_cert4_qwen_active_path_test.rs`). The Python reference side is feasible
(one F64 decoder layer at a time). **The Atenia side does not fit this host.**

Measured / config facts:

| Fact | Value | Source |
|---|---|---|
| DeepSeek-V2-Lite on disk (bf16, 4 shards) | **29.26 GB** | `Get-ChildItem` |
| One f32 copy of the whole model in RAM | **~58.5 GB** | disk × 2 |
| Host RAM total / free | **31.7 GB / 17.9 GB** | `Win32_OperatingSystem` |
| `hidden` / `vocab` | 2048 / 102400 | `config.json` |
| routed experts / top-k / shared | 64 / 6 / 2 | `config.json` |
| `moe_intermediate_size` / dense `intermediate_size` | 1408 / 10944 | `config.json` |
| layers (dense-first) | 27 (`first_k_dense_replace=1`) | `config.json` |
| `tie_word_embeddings` | false (separate `lm_head`) | `config.json` |

Code path that forces this (exact locations):

1. **`src/moe/runtime.rs:874-878`** — DeepSeek is hard-pinned to `ExpertTier::Ram`:
   ```rust
   let expert_tier = if family == MoeFamily::DeepSeekMoe {
       ExpertTier::Ram          // <-- forces RAM-f32 for DeepSeek
   } else {
       expert_tier_from_env()   // Qwen/Mixtral honour ATENIA_MOE_EXPERT_TIER=disk
   };
   ```
   The in-code comment says it outright: *"DeepSeek's MLA forward consumes the
   `RealMoeLayer` imperatively, so it stays RAM-f32 (tier change is a follow-up)."*

2. **`src/moe/runtime.rs:931-933`** — for DeepSeek the assembly loop keeps **both** a
   `ResidentExpertLayer` (pushed into `residency`, then never used by the forward)
   **and** the f32 `RealMoeLayer` (pushed into `deepseek_moes`):
   ```rust
   if family == MoeFamily::DeepSeekMoe {
       residency.push(Arc::new(resident_ram)); // built, but unused at forward time
       deepseek_moes.push(moe);                // the f32 layer the forward actually uses
   }
   ```

3. **`src/moe/runtime.rs:971-980`** — `build_deepseek(..., deepseek_moes)` moves every
   f32 `RealMoeLayer` into `DeepseekWeights`, all 26 MoE layers held simultaneously.

4. **`src/moe/mla.rs:129-132`** — the FFN variant holds the f32 layer by value:
   ```rust
   pub enum DeepseekFfn { Moe(RealMoeLayer), Dense(DenseFfn) }
   ```

5. **`src/moe/mla.rs:410-417`** — the forward calls `RealMoeLayer::forward_with`
   directly, i.e. it reads the in-RAM f32 experts of every layer:
   ```rust
   DeepseekFfn::Moe(m) => m.forward_with(&h2, cfg.moe_convention()).expect(...),
   ```

**Why Qwen's disk tier does not apply today:** Qwen/Mixtral run through
`Backend::Graph` + `MoeBlock`, which already carries the residency tier
(`ResidentExpertLayer` on NVMe, resolved per-token, LRU-cached). DeepSeek runs
through `Backend::Mla`, whose imperative `layer_step` consumes a `RealMoeLayer`
that never touches the tier. The tier machinery is **fully built and certified**
(`src/moe/residency.rs`); it is simply **not wired into the MLA branch**.

### The key reuse asset (already certified)

`ResidentExpertLayer` (`src/moe/residency.rs`) already provides everything needed:

- `from_real_layer(layer, ExpertTier::Disk)` → experts on NVMe, **zero RAM**
  (`resident_ram_bytes()` collapses to ~router size).
- `forward(x)` and `forward_cached(cache, x)` perform the **exact** certified
  operations of `RealMoeLayer::forward_with` (same `route`, same
  `top_k_routing_with`, same SwiGLU, same combine, same shared-expert convention,
  convention-aware: `Atenia` renorm vs `HuggingFaceQwen` + shared gate).
- **bf16-auto on-disk format** (`TierFmt::Bf16Auto`, `bf16_truncate_lossless`): for
  any weight decoded from a bf16 source (every DeepSeek-V2-Lite weight is bf16 on
  disk → low-16-mantissa-bits zero) the bf16 tier is **bit-identical** to f32.
- Bit-identity is locked by existing tests
  (`disk_tier_matches_real_layer_bit_for_bit`, `forward_is_deterministic_across_tiers`)
  **and** by the runtime's load-time `self_validate_residency` (`runtime.rs:928`),
  which already runs for DeepSeek today.

So the integration is "route the MLA FFN through the certified resident layer
instead of the raw `RealMoeLayer`," not "build a new tier."

---

## 2. Alternatives (FASE 2)

Per-expert sizes (f32 bytes): routed expert = `3 · 1408 · 2048 · 4` ≈ **34.6 MB**;
combined shared (width `2·1408`) ≈ **69 MB**/layer. Backend (tier-independent,
always RAM-f32):

- embed + lm_head (untied): `2 · 102400 · 2048 · 4` ≈ **1.68 GB**
- MLA attention ×27 (`w_q`+`w_kv_a`+`kv_a_ln`+`w_kv_b`+`w_o`+norms ≈ 13.8M params/layer)
  ≈ **1.49 GB**
- dense layer-0 FFN (`3 · 10944 · 2048 · 4`) ≈ **0.27 GB**
- **backend total ≈ 3.4 GB RAM**

Routed+shared expert params total ≈ **14.85B** → **~59 GB f32 / ~29.7 GB bf16**.

### Option A — disk-tier DeepSeek experts (bf16-auto on NVMe), MLA attention in RAM  ★ recommended

Wire the DeepSeek MLA FFN to `ResidentExpertLayer` at the env-selected tier
(`ATENIA_MOE_EXPERT_TIER=disk`), exactly like Qwen. Experts live on NVMe (bf16-auto,
lossless); top-k (+shared) resolved on demand and dropped.

- **RAM:** backend ~3.4 GB + one layer's active set (top-6 routed + 2 shared,
  resolved to f32 transiently) ~0.3 GB + optional LRU cache budget (configurable) →
  **≈ 4 GB** (vs **58.5 GB** today). Fits 17.9 GB free with wide margin.
- **NVMe:** experts bf16 ≈ **29.7 GB** on the tier dir. `F:` has **726 GB** free;
  point `ATENIA_DISK_TIER_DIR=F:\...`.
- **Numeric risk: zero.** bf16-auto is bit-identical for bf16-source weights; any
  non-representable value auto-falls back to f32 (`bf16_truncate_lossless → None`).
  No threshold lowered, no metric invented.
- **Changes:** isolated to the DeepSeek branch (see §3). Qwen/Mixtral untouched.
- **Reuse:** ~100% of `residency.rs` + `disk_tier.rs` + `ExpertCache`. The runtime
  already *builds* a resident layer per DeepSeek layer (currently RAM, unused).
- **Complexity: low-moderate.** Only wrinkle = threading a mutable `ExpertCache`
  through the imperative `layer_step`; sidesteppable for C5 (see §3, uncached path).

### Option B — bf16 resident experts in RAM (on-the-fly per-expert convert)

Keep experts in RAM but as bf16 (`SharedParam::Bf16`), upcast per expert at forward.

- **RAM:** experts bf16 ≈ **29.7 GB** + backend 3.4 GB = **~33.1 GB → exceeds 31.7 GB
  total.** Infeasible (and leaves no headroom for the OS/transients). **Rejected.**
- Numeric risk zero (bf16 lossless) but it does not solve the blocker.

### Option C — bespoke per-layer/token active-expert streaming (new code)

Write a fresh streaming path that reads each routed expert from the source shards
per token and drops it.

- **RAM:** comparable to A (~4 GB).
- **But:** this is essentially what `ResidentExpertLayer::forward` already does
  (resolve top-k on demand → execute → drop), minus the certified tier, LRU cache,
  bf16-auto, and the equality locks. Re-implementing it duplicates certified code
  and re-opens convention/numeric risk for **no benefit over A**. **Rejected** in
  favour of reusing A's certified machinery.

### Option D — freeze DeepSeek-V2-Lite at L2 (do nothing)

Honest and zero-risk; documents the C5 blocker as a hardware/integration limit.
Acceptable fallback, but L3 stays blocked by an **integration gap**, not a real
numeric or modelling limitation — so A is preferable if L3 is wanted.

| Option | RAM | NVMe | Numeric risk | Reuse | Complexity | Feasible here |
|---|---|---|---|---|---|---|
| **A disk-tier** | **~4 GB** | ~29.7 GB | **none (bf16 lossless)** | ~100% | low-mod | **yes** |
| B bf16-RAM | ~33 GB | 0 | none | high | low | **no (OOM)** |
| C bespoke stream | ~4 GB | shards | re-opened | low | high | yes (worse than A) |
| D freeze at L2 | n/a | n/a | none | n/a | none | yes (no L3) |

---

## 3. Recommendation (FASE 3) — Option A

Wire the experimental MLA FFN through the already-certified `ResidentExpertLayer`
disk/bf16 tier. It maximises correctness (bit-identical, bf16-lossless), minimises
risk, reuses the Qwen path almost entirely, and is fully isolated to the
experimental `Backend::Mla` branch with **zero** impact on the productive runtime,
Qwen, or Mixtral.

### Proposed change surface (for the future MLA-2 implementation — NOT done here)

1. **`src/moe/mla.rs`** — change the FFN variant to hold the resident layer:
   ```rust
   pub enum DeepseekFfn { Moe(Arc<ResidentExpertLayer>), Dense(DenseFfn) }
   ```
   Call site (`layer_step`, ~line 413):
   ```rust
   DeepseekFfn::Moe(m) => m.forward(&h2).map(|(y, _)| y).expect("deepseek moe forward"),
   ```
   `ResidentExpertLayer` already carries the convention (`resolve_convention()`),
   so `cfg.moe_convention()` is no longer passed in — the resident layer reproduces
   the same `HuggingFaceQwen`/`Atenia` behaviour that `forward_with` did.

   - **C5-minimal:** use **uncached** `forward` → no cache threading, no change to
     `forward_prefill`/`forward_decode` signatures, fully deterministic and
     bit-identical. C5 is a 4-token probe, so re-reading the tier per token is fine.
   - **Productive (later, optional):** thread a per-layer `ExpertCache` (interior
     mutability, e.g. `Mutex<ExpertCache>` in the FFN, or a cache vector owned by
     the forward) to amortise NVMe reads during generation. Not needed for C5.

2. **`src/moe/runtime.rs:874-878`** — stop forcing `ExpertTier::Ram` for DeepSeek;
   honour `expert_tier_from_env()` like the other families.

3. **`src/moe/runtime.rs:931-933`** — for DeepSeek, build the resident layer at the
   chosen tier and pass it (Arc) into `build_deepseek`; **drop** the raw
   `RealMoeLayer` after the resident is built (mirrors the Qwen disk arm at
   `runtime.rs:940-964`), so no f32 expert copy is retained.

4. **`build_deepseek` (`runtime.rs:1281`, `mla.rs` `DeepseekFfn`)** — accept
   `Vec<Arc<ResidentExpertLayer>>` instead of `Vec<RealMoeLayer>`.

Keep the persistent-tier warm-reconstruct (MOE-PROD-4/5) **disabled** for DeepSeek
initially (`runtime.rs:889` already excludes it) — use the simplest ephemeral disk
tier (UUID-named, deleted on drop) to minimise the change. Persistent tier is a
later optimisation, out of MLA-2 scope.

**Out of scope (NO HACER, unchanged):** latent compressed KV cache, Q-LoRA,
DeepSeek-V3, productive loader changes, performance optimisation beyond the tier
wiring, any numerics change, Adapter Toolkit.

---

## 4. Validation plan (FASE 4) — prove parity BEFORE any real C5

Each step is bit-exact equality against the existing RAM-f32 path (no new metric,
no threshold change). Ordered cheapest-first; stop and report on any mismatch.

1. **Tiny fixture parity (primary cheap gate).** Run the existing MLA-0 test
   `mla0_v2lite_full_forward_matches_hf` with `ATENIA_MOE_EXPERT_TIER=disk`
   (bf16-auto). Expectation: **identical `9.072e-5`** vs HF f64 and exact argmax —
   i.e. byte-identical to the current RAM run. This alone proves the tier wiring is
   correct end-to-end on the exact V2-Lite conventions (YaRN + dense-first +
   no-renorm + MLA + shared).
2. **Layer-level parity (new unit test).** For a tiny DeepSeek MoE layer, assert
   `DeepseekFfn::Moe` disk-tier `forward(x) == RealMoeLayer::forward_with(x, conv)`
   bit-for-bit (mirrors `residency.rs::disk_tier_matches_real_layer_bit_for_bit`).
3. **Per-token parity.** MLA-0 `mla0_v2lite_greedy_generates_to_eos` under
   `disk` tier must yield the **identical** greedy/EOS sequence as RAM, and be
   deterministic across two runs.
4. **Comparison against current RAM-f32.** The MLA-0 fixture *is* the RAM-f32
   reference; steps 1–3 passing == no regression vs today's certified path.
5. **Regressions.** Full MoE suite green, especially the Qwen/Mixtral runtime and
   residency tests (they never enter the DeepSeek branch) + `residency.rs` unit
   tests + `moe_mla1_deepseek_decomposition_test` (C1/C2 unaffected).
6. **Only after 1–5 pass:** generate the C5 F64 reference (python, one layer at a
   time, `gc` between layers — peak ~one layer in F64, never the whole model → not
   L4), then run the real C5 harness under `disk` tier with
   `ATENIA_DISK_TIER_DIR=F:\...`. Gate: end-to-end `max_abs_diff < 0.5` (ADR-004,
   unchanged) + exact per-position argmax + deterministic. Pass → L3; fail → do NOT
   certify, document, STOP.

Expected resources for the real C5 run: **~4 GB RAM**, **~29.7 GB NVMe (bf16)** on
`F:` (726 GB free). Reference build: peak ~one F64 layer (~few GB).

---

## 5. Risks

- **Cache mutability (complexity).** Threading `ExpertCache` through the imperative
  MLA forward needs interior mutability. **Mitigation:** C5 uses the uncached
  `forward` (no threading); caching is a later, optional perf step.
- **NVMe write cost.** First load writes ~29.7 GB to the tier. One-time per run
  (ephemeral tier deleted on drop). Acceptable on `F:` (726 GB, fast NVMe).
- **bf16 representability.** Only a concern if a weight is not bf16-representable →
  auto-fallback to f32 (already handled, no silent loss). DeepSeek-V2-Lite is bf16
  on disk → all lossless.
- **Convention drift.** `ResidentExpertLayer` already encodes the
  `HuggingFaceQwen`/`Atenia` conventions and is load-time self-validated against the
  `RealMoeLayer`; the MLA-0 parity gate (step 1) catches any mismatch before C5.
- **Scope creep.** The tier touches `runtime.rs`; risk of perturbing Qwen/Mixtral.
  **Mitigation:** all edits are inside `if family == DeepSeekMoe` arms; the Qwen
  disk path is the template and stays byte-identical.
- **Not L4.** This streams experts; it never holds the whole model in F64. L4
  remains reserved/unreachable. C5/L3 ≠ L4.

---

## 6. Bottom line

- **Root cause:** not numerics / model / YaRN / dense-first — purely **expert
  residency**: the MLA forward consumes a RAM-f32 `RealMoeLayer`; the certified
  disk/bf16 tier exists but is not wired into the `Backend::Mla` branch.
- **Recommended:** **Option A** — route the MLA FFN through the existing certified
  `ResidentExpertLayer` disk tier (bf16-auto). RAM **~4 GB** (from 58.5 GB), NVMe
  **~29.7 GB** on `F:`. **Zero numeric risk** (bf16 lossless, bit-identical),
  isolated to the experimental DeepSeek path, ~100% reuse of the Qwen mechanism.
- **Worth implementing?** **Yes, if L3 is the goal** — it is the only path to a
  feasible, honest C5 on this host, at low risk. If L3 is not a priority, **freezing
  at L2 (Option D)** is a fully honest fallback. The decision is value/priority, not
  technical feasibility — A is feasible and safe.
