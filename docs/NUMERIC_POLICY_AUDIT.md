# NUMERIC POLICY — Architecture Audit (documentation only)

Audit of the numeric-precision architecture built across NUMERIC-POLICY-1/2 and
MOE-PERF-1/2/3 (HEAD `8e7b61e`, CI green). **No code was changed.** Scope: the
MoE compute-precision policy, the quantized tier, certification, fallbacks, and
where the remaining ROI is. Hardware ref: RTX 4070 Laptop (8 GB VRAM), 24-core
CPU, 32 GB RAM, NVMe, Windows 11; real model Qwen1.5-MoE-A2.7B-Chat.

---

## FASE 1 — Numeric Inventory

### Compute policies (`src/moe/numeric_policy.rs`)
`NumericPolicy { Certified, Strict, Fast }`, selected by
`ATENIA_NUMERIC_POLICY={certified|strict|fast}` (cached) or in-process override.
**Default + universal fallback: `Certified`.**

| Policy | Expert-FFN matmul | Reference relation | Status |
|---|---|---|---|
| **Certified** | f64 accumulation (CPU) | the reference itself | ✅ done, default |
| **Strict** | f32 accumulation (CPU, 24-core) | tolerance-certified | ✅ done, fast default |
| **Fast** | f32 on GPU (`cuda_matmul`, CPU-f32 fallback) | tolerance-certified | ⚠️ works, **slower**, not recommended |

Scope note: only the **expert FFN** (gate/up/down) honours the policy. The
**router** stays f64 on every policy (so top-k routing is identical). Attention /
lm_head run through the GraphBuilder; softmax/SiLU are f64.

### Tier storage dtypes (`src/tensor/disk_tier.rs`, `src/moe/residency.rs::TierFmt`)
Orthogonal to the compute policy — this is **on-disk format**, dequantised to f32
before the forward.

| Dtype | Bytes | Lossy? | Knob | Status |
|---|---|---|---|---|
| **F32** | `numel*4` | no | (fallback) | ✅ |
| **BF16** | `numel*2` | no (lossless for bf16-source) | `ATENIA_MOE_TIER_BF16` (default on) | ✅ |
| **QInt8** | `rows*4 + numel` | yes (per-row int8) | `ATENIA_MOE_TIER_QUANT=int8` (default off) | ✅ certified |

Backend tensors (embed/lm_head/attention/router/gate) are F32/BF16 only; **only
routed + shared experts** may be QInt8.

### Certification (`PolicyCertificate`)
Metrics (f64): `max_abs_diff`, `mean_abs_diff`, `rmse`, `argmax_match_rate`,
`tokens_match`. `passes(τ)` = every-row argmax agrees **and** tokens identical
**and** `max_abs_diff ≤ τ`. `STRICT_LOGIT_TOLERANCE = 0.5`. Plus a **cheap
in-memory sim** (`ATENIA_MOE_QUANT_SIM=int8`) used to certify a candidate format
without a cold rebuild.

### Fallbacks (what is actually wired)
- **Compute:** `Certified` is default; unknown/unset env → `Certified`.
- **GPU:** `cuda_matmul` auto-falls-back to the exact CPU f32 matmul on any CUDA
  error / no-GPU / CUDA-less build (kernel level).
- **Tier load:** `try_warm_reconstruct` → `None` (→ certified shard path) on
  manifest version/model_id mismatch, missing file, or a size matching none of
  f32/bf16/qint8.

### Manifest / dtypes
Manifest **v5**: per-entry `dtype` (`f32`/`bf16`/`qint8`) + explicit on-disk
`bytes`. Version history: v2 (identity), v3 (bf16 experts), v4 (bf16 backend),
v5 (qint8 + bytes). A bump invalidates older tiers (safe one-time rebuild).

### Measured results (real model, prompt `22,25,29`, max-new 2, warm)
| | tier | resolve | wall | tokens |
|---|---|---|---|---|
| bf16 (NP-1) | 26.7 GiB | ~90 s | ~180 s | `16,15` |
| **int8 (NP-2)** | **14.3 GiB** | **45 s** | **142 s** | `16,15` (certified) |

---

## FASE 2 — Gap Analysis

**The scaffolding is complete and working; the *governance* of correctness is
not yet production-grade.** Gaps, by severity:

1. **Certification is offline/manual, not enforced or persisted.** `PolicyCertificate`
   exists and is used in unit tests + the ad-hoc sim run, but the **runtime does
   not** run a candidate-vs-Certified check and auto-fall-back; it **trusts** the
   env choice. And there is **no persisted per-checkpoint/per-tier certificate
   artifact** (unlike the dense `numcert`). So "if it doesn't certify → fallback"
   is a *design principle honoured manually*, not an automated guarantee. **This
   is the #1 gap for production.**
2. **The certificate sample is tiny.** Certification used 2–8 tokens on **one**
   prompt. A trustworthy certificate needs a **validation prompt set** + longer
   sequences + a perplexity / top-k-agreement metric, not just argmax on 8 tokens.
3. **Policy/tier coherence.** The original proposal wanted the quantized tier
   *expressed as a policy* (`Fast`/`QuantizedFast`). It shipped as an **orthogonal
   storage env** (`ATENIA_MOE_TIER_QUANT`) instead — a defensible, documented
   choice, but it means "what numeric mode am I in" is split across two knobs
   (compute policy + tier dtype) with no single source of truth.
4. **`Fast` is dead weight on this target.** It works and certifies but is
   **slower** (MOE-PERF-2: per-token GPU GEMVs over transient weights lose to the
   CPU). No current use case. Its docstring is also **stale** ("currently ==
   Strict" — it actually dispatches to CUDA).
5. **"Certified = f64" is not uniformly true.** Certified is the *reproducible
   CPU path*; the expert FFN is f64, but attention/lm_head go through the
   GraphBuilder (f32 on CPU). The reference is deterministic and is what every
   tier change is bit-checked against, but it is **not pure-f64 end-to-end** —
   worth stating explicitly so the "reference" claim isn't over-read.
6. **Integrity = size-only.** Tier validation checks file **size**, not content.
   A corrupt-but-right-size qint8/bf16/f32 file would dequantise garbage. (Same
   pre-existing risk for all dtypes; a checksum would close it.)
7. **Scope: graph families only.** The policy + quant tier apply to Mixtral /
   Qwen-MoE. **DeepSeek/MLA stays RAM-f32** (no tier, no policy). Documented, not
   a defect, but a coverage hole.

**Technical debt (minor):** the stale `Fast` docstring; `DiskDtype::bytes_per_element()`
returns 1 for QInt8 (a leaky abstraction — the real size needs `qint8_disk_bytes`);
the int8 **sim** path (`qdq_per_row_i8` in `resolve`) is now partly redundant
with the real tier and could be retired once the cert harness subsumes it.

---

## FASE 3 — ROI Ranking (future levers, highest first)

| Lever | Expected ROI | Risk | Notes |
|---|---|---|---|
| **AV exclusion (operational)** | **Very high** (~rest of the 45 s resolve → ~NVMe speed) | none (config) | Not code — a one-line operator action; the CLI already detects + recommends it. The single biggest remaining win. |
| **Certificate infrastructure** (persisted artifact + validation prompt set + auto-fallback) | High (correctness, unlocks trusting non-Certified defaults) | low | Makes the non-bit-exact paths *safe to ship by default*. Prerequisite for "production". |
| **int4 / group-wise int4 tier** | Medium-high (tier → ~7 GiB, resolve → ~25 s) **iff it certifies** | medium (likely token flips) | **Sim-certify first** (cheap, reuses the int8 machinery). Per-row int4 may fail; group-wise (64) is the fallback. |
| **IO improvements** (tier caching across runs is done; consolidation) | **Negative / harmful** | — | MOE-IO-1 measured: consolidation makes the AV scan whole layer blobs — *worse*. Eliminated. |
| **CUDA improvements** (resident shared, batched prefill, lm_head GPU) | **Low** | high | MOE-PERF-3 measured: expert matmul < 1 % of the wall; GPU can't move an I/O-bound wall. Only lm_head (a real GEMM) might help marginally. |
| **Residency / streaming changes** | Low | high | Bottleneck is bytes/AV, not residency mechanics; already on-demand + cached. |

---

## FASE 4 — Architecture Review

- **Does Numeric Policy meet the original design?** **Mostly.** The tiered
  Certified/Strict/Fast model, the tolerance certificate, and the safe-fallback-
  to-Certified principle are all present and working. The **deviations**: (a) the
  quantized tier is a storage env, not a policy variant (documented); (b)
  certification is offline/manual rather than an enforced runtime guard; (c) no
  persisted certificate artifact yet. The *spirit* (correctness > performance,
  Certified is the reference + fallback, every lossy path is tolerance-checked)
  is upheld.
- **Inconsistencies:** the `Fast` docstring is stale; "Certified = f64" is not
  uniform (FFN f64, attention/lm_head f32-CPU); the numeric mode is split across
  two knobs.
- **Possible simplifications:** unify "what numeric mode" behind a single
  resolved descriptor (compute policy × tier dtype) for logging/telemetry; retire
  the int8 *sim* once the cert harness covers it; collapse `Fast` to an explicit
  opt-in experiment (it has no default use case).

---

## FASE 5 — Roadmap (NOW / NEXT / LATER / NEVER)

**NOW (highest value, low/zero risk):**
- Document + surface the **AV-exclusion** operational win (already detected by the
  CLI) — it dwarfs any remaining code lever.
- **Certificate infrastructure**: a persisted per-checkpoint/per-tier certificate
  (argmax + tokens + drift over a small **validation prompt set**), so a
  non-Certified tier/policy is *provably* safe and can become a trusted default.

**NEXT (real code lever, evidence-gated):**
- **int4 sim-certification** (no implementation until it certifies). If per-row
  int4 fails, try **group-wise int4**. Big additional byte/resolve win *iff* it
  passes.
- Minor cleanups: fix the `Fast` docstring; single numeric-mode descriptor in the
  stats line.

**LATER (only with new evidence):**
- DeepSeek/MLA tier + policy coverage.
- Content checksums in the manifest (integrity beyond size).
- lm_head GPU offload (the one GEMM big enough to maybe pay off) — measure first.

**NEVER (measured dead-ends — do not pursue without contradicting evidence):**
- Optimising the **expert matmul** further (CPU or GPU) — it is < 1 % of the wall.
- **Tier file consolidation** — harmful under real-time AV (scans whole blobs).
- **Per-token GPU offload of routed experts** — transfer-bound, strictly loses.

---

## FASE 6 — Deliverable answers

1. **Real state of Numeric Policy:** functionally **complete and working** —
   three compute tiers (Certified/Strict/Fast) + three tier dtypes
   (f32/bf16/qint8), a tolerance certificate, and safe fallbacks; bit-exact
   default; one certified, measurable win shipped (int8: −46 % tier, −21 % warm,
   identical tokens). CI green.
2. **Done:** Certified (f64 reference), Strict (f32 CPU, fast default), the bf16
   + int8 tiers, the `PolicyCertificate` metrics, the cheap sim-certification, the
   load-time + GPU fallbacks, manifest v5.
3. **Incomplete:** **enforced/persisted certification** (it's offline/manual);
   a **validation prompt set** (sample is 8 tokens / 1 prompt); a single numeric-
   mode source of truth; DeepSeek coverage; content-integrity checks.
4. **Highest ROI:** (1) the **AV exclusion** (operational, huge, zero code);
   (2) **certificate infrastructure** (unlocks shipping fast modes by default);
   (3) **int4** *if* it sim-certifies.
5. **Not worth it:** more expert-matmul optimisation (CPU/GPU), tier
   consolidation, per-token routed-expert GPU offload, streaming/residency
   rework — all measured dead-ends.
6. **Does int4 deserve to exist?** **Maybe — gated, not yet.** It is the only
   remaining high-ROI *code* lever (≈ halve the tier again), but per-row int4 is
   far lossier and may flip tokens. **Decision: run the cheap int4 sim-cert
   first; implement only if it certifies; else try group-wise int4; else stop.**
   Do **not** build it on intuition.
7. **Does CUDA deserve priority?** **No.** Measured: the GPU does not help the MoE
   path (expert matmul < 1 % of the wall; per-token GEMVs are transfer-bound).
   `Fast` exists, certifies, but is slower. Lowest priority; revisit only for
   lm_head, and only with a measurement.
8. **Is the architecture production-ready?** **The compute is; the governance is
   not yet.** The numbers are real, certified, reproducible, and CI-green, and the
   default is bit-exact + safe. But shipping a *non-Certified* path as a trusted
   default needs the **enforced/persisted certificate + a validation set** (gap
   #1/#2). Until then, non-Certified modes are **opt-in and operator-verified**,
   which is correct but not turnkey.
9. **Risks:** (a) a future model/quant could silently drift if certification
   stays manual — mitigated today by default-Certified + opt-in; (b) size-only
   integrity; (c) the numeric mode split across two knobs invites operator
   confusion; (d) `Fast`'s stale docs could mislead.
10. **Final recommendation:** **Freeze the precision *kernels* — the architecture
    is functionally complete and the expert-compute frontier is exhausted.**
    Spend the next effort on **certification governance** (persisted certificate
    + validation prompt set + optional runtime auto-fallback), which converts the
    already-built fast modes from "opt-in, manually verified" into "trusted,
    default-safe." Treat **AV exclusion** as the immediate operational win and
    **int4** as an evidence-gated experiment. Keep CUDA and tier-layout work off
    the roadmap unless new evidence appears.

---

*No code, tests, commits, or CI were touched by this audit. Documentation only.*
