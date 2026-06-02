# HANDOFF — MOE-PROD series (1 → 8): closing audit

The **MOE-PROD** series is **closed** at commit `8610cb6` (CI green). It took the
controlled MoE runtime from "loads but is impractically slow / RAM-bound" to a
**bit-exact, tier-reusing, bf16, multi-core warm path** on a real 14.3 B
(2.7 B-active) MoE on an 8 GB-VRAM / 32 GB-RAM laptop. This document audits the
whole series, states the cumulative result, and hands off to the next series
(**NUMERIC-POLICY / MOE-PERF**, designed in
[PROPOSAL_NUMERIC_POLICY_MOE_PERF.md](./PROPOSAL_NUMERIC_POLICY_MOE_PERF.md)).

Hardware reference for every number: RTX 4070 Laptop (8 GB VRAM), 24-core CPU,
32 GB DDR5, NVMe SN770, Windows 11. Real model: Qwen1.5-MoE-A2.7B-Chat
(24 layers, 60 experts, top-4, shared expert, hidden 2048).

## 1. Per-milestone audit

| # | Title | What it did | Real result |
|---|---|---|---|
| **1** | Sharded MoE loader | Load multi-shard HF checkpoints (`model.safetensors.index.json`); single open shard cached; bit-identical to single-file. | Unblocked real MoE load (peak RAM ~1 shard, not all). |
| **2** | Disk-backed expert residency | `ExpertTier{Ram,Disk}`; experts live on NVMe (`SharedParam::Disk`), only top-k resolved per token → **zero RAM** for non-resident experts. | Fit a 50 GB-f32 expert set on a 32 GB box. |
| **3** | Expert cache integration | Bounded LRU (`forward_cached`) so re-routed experts skip the NVMe re-read; honest measurement. | **22.7 % hit ratio, ~0 % wall speedup** — proved **load**, not routed re-reads, dominated. |
| **4** | Persistent tier (scope A) | Deterministic tier names + manifest; **skip the 50 GB write** when a valid tier exists. | cold 3757 s → **warm 2445 s (~35 %)**; bit-exact. |
| **5** | Warm reconstruction (scope C) | Rebuild the **whole backend from the tier** — no shard read, no expert assembly; safe fallback. | warm 2445 → **701 s (~71 %)** (same-prompt baseline later = **350 s**); bit-exact. |
| **6** | BF16 tier + shared cache | Experts persist **bf16** (auto-lossless); pinned **shared-expert** cache (read once/layer, not every token). | same-prompt warm 350 → **204 s (~42 %)**; tier 53.3 → 28.6 GiB; cold 2587 → 1942 s; bit-exact. |
| **7** | BF16 backend + bulk reader + profiling | bf16 the backend too; memcpy-speed reader; **per-phase warm profiling**. | **Key finding:** reconstruct is only **34.6 s of 204 s (~17 %)** — load was **not** the bottleneck. tier 28.6 → 26.7 GiB; cold 1942 → 1755 s. Warm wall unchanged (gen-bound). |
| **8** | Generation compute | Parallel `matvec` (24 cores, bit-exact f64-per-row); single-copy expert resolve; gen profiling. | warm 204 → **184 s (~10 %)**; bit-exact. Confirmed **generation compute (~94 s)** is dominant. |

## 2. Cumulative metrics (real, same-prompt where comparable)

Rigorous same-prompt chain (`22,25,29`, max-new 2, ids `16,15` — bit-exact):

```
MOE-PROD-5 (f32 tier, no shared cache, = pre-6 behaviour)   350 s
MOE-PROD-6 (bf16 experts + shared cache)                    204 s
MOE-PROD-7 (bf16 backend + bulk reader)                     204 s   (load 34.6s; gen-bound)
MOE-PROD-8 (parallel matvec + single-copy resolve)          184 s
```

- **Warm wall: 350 → 184 s (~47 %)** across PROD-5→8, **bit-exact** end-to-end.
- Across the whole series the warm path went from **~2445 s (PROD-4 warm) to
  184 s** (different prompts early on; ~13× on the headline numbers).
- **Tier on disk: ~53 → 26.7 GiB** (bf16 experts + backend).
- **Cold (writes tier): 3757 → 1755 s.**
- **Warm reconstruction (load only): now ~34 s (~18 % of wall).**
- **RAM:** experts never materialised all at once (disk tier); shared expert
  pinned (~one per layer); reconstruction reads the backend f32.
- **Bit-exactness:** preserved at every step; proven by deleting the shards and
  regenerating identically, and by disk==RAM / cached==uncached equality tests.

## 3. Bottlenecks found (in discovery order)

1. **Experts don't fit in RAM** (50 GB f32) → PROD-1/2.
2. **Routed re-reads** suspected → measured **not** dominant (PROD-3).
3. **Tier write (50 GB) every run** → PROD-4.
4. **Shard read + f32 expert assembly on warm** → PROD-5.
5. **Per-token expert NVMe volume + shared expert read every token** → PROD-6.
6. **Backend f32 read + element-by-element decode on warm** → PROD-7.
7. **(measured) Generation graph execution (~94 s), CPU f64 matmul** → PROD-8
   (partially: parallelised the matvec) → **now the dominant, and bit-exact-CPU
   levers are exhausted**.

## 4. Bottlenecks resolved

- RAM ceiling (disk tier), redundant tier write (persist), shard read + assembly
  on warm (reconstruct), per-token + shared I/O (bf16 + shared cache), backend
  read + slow decode (bf16 backend + bulk reader), single-thread expert matmul
  (rayon), redundant resolve copy (single-copy). **All bit-exact, all with safe
  fallbacks, default path unchanged.**

## 5. Current state

- Controlled MoE runtime: sharded load → disk-tier residency → **persistent bf16
  tier** → **warm reconstruction from tier** (no shards) → bounded expert cache +
  **pinned shared cache** → **multi-core bit-exact generation**.
- Opt-in (`ATENIA_ENABLE_MOE` / `--experimental-moe`,
  `ATENIA_MOE_EXPERT_TIER=disk`, `ATENIA_MOE_TIER_PERSIST=1`); escape hatches
  `ATENIA_MOE_TIER_BF16=0`, `ATENIA_MOE_SHARED_CACHE=0`, `ATENIA_MOE_EXPERT_CACHE`.
- Manifest **v4**; profiling via `ATENIA_MOE_CACHE_STATS=1`.
- CI green; bit-exact (`16,15`) on the real model.

## 6. Lessons learned

- **Measure, don't guess.** PROD-3 (cache gave ~0 % despite 22.7 % hits) and
  PROD-7 (reconstruct only 17 % of wall) both **overturned** plausible
  hypotheses. The profiling instrumentation was the highest-value deliverable —
  it redirected the whole block away from load and onto generation compute.
- **Honest baselines beat headline numbers.** The "701 s → 204 s" comparison was
  confounded by prompt length (8 vs 4 MoE rows); re-measuring a same-prompt
  baseline (350 s) gave the truthful gain.
- **Bit-exactness is cheap when you respect data, not math.** Every win (bf16
  lossless-by-construction, tier reuse, parallel-rows, single-copy) changed
  *where/how bytes move*, never the arithmetic — so bit-equality held for free.
- **ROI is front-loaded.** Load optimisation (PROD-4/5/6/7) returned huge gains;
  once load fell below ~20 % of the wall, further load work had near-zero ROI.
- **The last mile is the math.** The residual bottleneck is pure CPU f64
  compute, which cannot shrink further **without changing the numerics** — a
  policy decision, not an engineering one.

## 7. Limitations of the certified f64 path

- The MoE reference (`dense.rs` / `mla.rs` `matvec`, router softmax, SwiGLU)
  accumulates in **f64** for reference-grade, **bit-reproducible** correctness.
  This is the source of the certificate — and the speed ceiling.
- f64 scalar accumulation does not vectorise well (per-element `f32→f64`),
  is ~2–4× slower than f32, and ignores the GPU and Tensor Cores entirely.
- Parallelising across rows (PROD-8) is the **only** speedup that keeps
  bit-exactness; it is now applied. **There is no further bit-exact CPU lever of
  meaningful ROI.**
- The model's active compute (2.7 B params/token) on CPU f64 is inherently
  ~tens of seconds per token at hidden 2048 — interactive latency is not
  reachable on this path.

## 8. Handoff

The next series — **NUMERIC-POLICY / MOE-PERF** — is **designed (not
implemented)** in
[PROPOSAL_NUMERIC_POLICY_MOE_PERF.md](./PROPOSAL_NUMERIC_POLICY_MOE_PERF.md):
a tiered numeric policy (certified-f64 vs tolerance-certified fast paths) and the
evaluation of f32, bf16, GPU offload, and Tensor Cores against a tolerance
certificate. No code is to be written until that proposal is reviewed.
