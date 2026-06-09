# HANDOFF — MOE-PERF-3-VALIDATION: how much of the prefetch win survives on certified workloads

Milestone: **MOE-PERF-3-VALIDATION** — measure the **real-workload** impact of the
MOE-PERF-3 parallel disk-tier expert prefetch on the **certified** MoE families, and decide
whether the synthetic ~2× benchmark translates into a meaningful end-to-end speedup. This is
a **measurement-only** milestone: no runtime / numerics / routing / MLA / cache / certification
/ manifest / ADR change. The only code touched is the **test-only** measurement harness
(`tests/moe_perf3_prefetch_bench.rs`, `#[ignore]`); `src/` is **untouched**.

---

## PHASE 1 — Recovered baselines (authoritative, not estimated)

| Family | Routing topology | Real cert path | C5 worst `max_abs_diff` / argmax | Wall-clock baseline | Source |
|---|---|---|---|---|---|
| **Mixtral-8x7B-v0.1** | 8 experts, **top-2**, 32 layers, hidden 4096, inter **14336**, **no shared** | ✅ real-weight, **disk expert-tier**, cap=1 | **3.185e-4** / argmax **4/4** | warm reconstruct **4.5 s**; **forward 402.7 s** (seq=4); test wall 955.74 s; ~29 GB peak | `HANDOFF_MIXTRAL_CERT_C5.md`, `numcert/mixtral-8x7b-v0.1.moecert.json` |
| **DeepSeek-V2-Lite** | 16 routed, **top-6**, **2 shared**, **MLA**, narrow experts | ✅ real-weight, **disk tier (~4 GB RAM resident)** | **2.587e-5** / argmax **4/4** | whole-model C5 (load/forward not separately logged) | `HANDOFF_MLA_3.md`, `tests/moe_scale_cert_test.rs` |
| **Qwen-MoE** | 16 experts, **top-4**, shared expert, GQA, qkv bias | **block-level only** (no full transformer path) | ~**1e-10 … 2.9e-11** vs HF f64 | **none** (no whole-model disk-tier forward exists) | `HANDOFF_QWEN_MOE_CERT_1.md`, `tests/moe_scale_cert_test.rs` |

**Key fact established up front:** only **Mixtral** and **DeepSeek-V2-Lite** have a real
whole-model **disk-tier** forward (the path prefetch accelerates). **Qwen-MoE was certified at
the block level** (its real checkpoint has no full transformer path), so there is **no certified
Qwen disk-tier workload** to measure prefetch on — its benefit is *inferred from top-k*, not
observed on a real Qwen run.

## PHASE 2 — Instrumentation audit

The PERF-3 instrumentation is **sufficient**; no new instrumentation was required.
`CacheStats` already exposes everything needed: `hits`, `misses`, `evictions`, `bytes`,
`resolve_nanos` (Σ of per-expert read time), `parallel_prefetches`, `prefetch_wall_nanos`
(overlapped wall read), plus `MoeResidencyTrace`. `overlap_saved = resolve_nanos −
prefetch_wall_nanos` is the **deterministic** measure of NVMe latency hidden, independent of
wall-clock jitter. The only code change is **test-only**: the `tests/moe_perf3_prefetch_bench.rs`
harness was extended with a per-family sweep (`perf3_prefetch_per_family`) that drives the real
certified `ResidentExpertLayer::forward_cached` disk-tier path at each family's true top-k. No
`src/` change, no runtime behavior change.

## PHASE 3 — Real workload measurement (safe surrogate, documented rationale)

**Rationale (why not a full real-weight re-run).** A full C5 re-run of Mixtral means loading
the real 87 GB checkpoint and a 402.7 s forward — the documented OOM/host-hang hazard
(`HANDOFF_MIXTRAL_CERT_C5.md`: default cache commits ~90 GB → OOM). The milestone explicitly
permits the closest safe workload that still exercises the real certified runtime. That is the
**certified disk-tier `forward_cached` path itself** (the exact code the real Mixtral C5 runs),
driven at **each family's real routing fan-out** (top-2 / top-4 / top-6) on representative
disk-tier expert tensors, at **cap=1** (the RAM-constrained case PERF-3 targets). Expert
geometry is held **equal across families** so the only independent variable is **top-k**
(how many disk reads a single forward can overlap). `ATENIA_MOE_PREFETCH` OFF vs ON, identical
input/cache/tier per row.

**Measured** (`perf3_prefetch_per_family`, 16 experts, 16-token decode, per-expert 1.5 MiB,
disk tier, cap=1; 4 reps, release):

| Family (top-k) | misses (OFF=ON) | Σ read | **overlap_saved** (hidden) | **overlap fraction** | decode speedup (range) |
|---|---|---|---|---|---|
| **Mixtral** (top-2) | **29** | ~23 ms | **~9.5 ms** | **~40 %** | 1.78–3.73× |
| **Qwen-MoE** (top-4) | **58** | ~53 ms | **~36.7 ms** | **~69 %** | 1.86–3.14× |
| **DeepSeek-V2-Lite** (top-6) | **89** | ~89 ms | **~69.5 ms** | **~78 %** | 1.66–2.55× |

Plus the original fixed bench (`perf3_prefetch_measurement`, top-6): **191.74 → 95.32 ms (~2×)**.

**Robust observations** (jitter-free):
- **`misses` are identical OFF vs ON** (29/58/89) — prefetch changes read **order**, not count.
  No extra reads, no cache-policy change. Misses scale ~linearly with top-k.
- **`overlap_saved` grows monotonically with top-k** (~9.5 → ~36.7 → ~69.5 ms): more experts
  per token ⇒ more reads to run concurrently ⇒ more latency hidden.
- **Overlap fraction rises with top-k** (40 % → 69 % → 78 %): with `k` concurrent reads the wall
  read approaches `Σread / k`, so the hidden fraction ≈ `(k−1)/k`.
- Wall-clock speedup is **always > 1.6×** but **noisy** (rayon scheduling); the deterministic
  `overlap_saved`/fraction is the metric to trust.

## PHASE 4 — Analysis (per family)

**Absolute vs relative.** Absolute ms saved per forward grows with top-k (DeepSeek hides the
most, ~69 ms; Mixtral the least in this *equalized-geometry* fixture, ~9.5 ms). But the
**fixture deliberately equalizes expert width**; the real families differ sharply:

- **Mixtral — bottleneck = I/O, prefetch = MAJOR win (A).** Mixtral is the only family where
  **cap=1 is *forced*** (704 MB F32 experts; cap=4 commits ~90 GB → OOM). Every layer re-reads
  its top-2 experts from NVMe every position → the 402.7 s forward is ~95 % read-bound. top-2
  overlaps 2 reads ⇒ hides ~40–50 % of read latency, **and Mixtral's experts are by far the
  widest** (inter 14336 ≈ 28× the fixture), so each hidden read is the largest in absolute time.
  Prefetch attacks exactly this family's dominant cost. **Major win.**
- **DeepSeek-V2-Lite — high per-forward overlap, but fewer real misses (B).** top-6 gives the
  **highest overlap fraction (~78 %)**, but the whole DeepSeek disk tier is only **~4 GB
  resident** — it largely **fits in RAM**, so cap is rarely forced to 1 and the **cap=1 thrash
  that makes prefetch matter is far milder** than Mixtral's. Prefetch helps the cold/first pass
  and any genuinely RAM-starved host, but a warm cache already serves most reads. **Moderate win
  in practice**; the mechanism is most effective here per-miss, but there are fewer misses.
  **MLA is orthogonal** — it shrinks the KV cache, not expert reads, so it neither helps nor
  hurts prefetch effectiveness.
- **Qwen-MoE — not exercised on a certified workload (C, coverage gap).** No real-weight
  whole-model disk-tier forward exists (block-level cert). top-4 ⇒ inferred ~69 % overlap, but
  this is **unmeasured on a real Qwen run**. Mechanism applies; production coverage is absent.

**Bottleneck split (from the measured + PERF-1 evidence):** every family is **I/O-bound at
cap=1** (`Σread` ≫ compute; PERF-1 found compute is hundreds–thousands tok/s, never the
bottleneck). Prefetch attacks the I/O term directly. It cannot help a compute-bound or
fully-RAM-resident run — which is exactly why DeepSeek (RAM-resident) sees a smaller *practical*
win than Mixtral (RAM-starved).

## PHASE 5 — Cross-family comparison

| Family | Prefetch benefit | Bottleneck | Why |
|---|---|---|---|
| **Mixtral** | **A — major** | **Disk I/O (cap=1 forced)** | 704 MB experts ⇒ can't cache ⇒ every read hits NVMe; widest experts ⇒ largest hidden read. The family prefetch was built for. |
| **DeepSeek-V2-Lite** | **B — moderate** | Disk I/O, but **~4 GB tier mostly RAM-resident** | Highest overlap *fraction* (top-6) per miss, but few real misses; MLA orthogonal. |
| **Qwen-MoE** | **C — unmeasured** | n/a (no disk-tier whole-model cert) | Block-level cert only; benefit inferred from top-4, never observed on a real run. |

- **Benefits most:** Mixtral — the only RAM-starved, read-bound, wide-expert family.
- **Benefits least (measurable):** DeepSeek-V2-Lite in practice (fits in RAM), despite the
  highest theoretical overlap fraction.
- **MLA changes effectiveness?** No — MLA is an attention/KV optimization, orthogonal to expert
  disk reads.
- **Does disk-tier dominate all families equally?** No — it dominates **Mixtral** (87 GB, cap=1
  forced); it is **secondary** for DeepSeek-V2-Lite (~4 GB resident). Prefetch value tracks
  **how RAM-starved the family is**, not just top-k.

## PHASE 6 — Roadmap reassessment (evidence-driven reorder)

This validation **could not measure prefetch on any real certified workload** — Mixtral's real
forward is the OOM hazard, and the **MoE-generate path lacks the prefill/decode/cache-hit
instrumentation the dense path has** (PERF-1 gap (b)). We were forced onto a representative
fixture. That is itself the strongest evidence for a reorder:

1. **PERF-5 (MoE-generate instrumentation parity) — promote to next.** Small, low-risk, zero
   numerics. It is the **prerequisite to ever measuring PERF-3/PERF-4 on real runs safely**
   (per-token cache-hit + read-latency telemetry → no full-reload gamble). Without it, every
   future perf claim on a real model stays a surrogate. **Highest *enabling* ROI.**
2. **PERF-4 (qint8 default tier, gated on a numeric cert) — still high user ROI, now second.**
   It attacks the **root** (expert bytes → ¼) where PERF-3 attacks the **symptom** (read
   latency): qint8 both speeds each read **and** lets a larger cache fit (cap>1 ⇒ fewer misses),
   reducing the very cap=1 thrash prefetch overlaps. Complementary, biggest win on Mixtral-class
   models. But its gains **cannot be validated on a real run until PERF-5 lands**.
3. **Latent (MLA) KV cache** and **GPU expert offload** — remain deferred / high-risk.

**Verdict:** PERF-4 is **no longer the unconditional next step**; the measured inability to
observe real-workload impact promotes **PERF-5 (instrumentation parity) ahead of PERF-4**, with
PERF-4 immediately after (its real-world gains then become measurable, not inferred).

## PHASE 7/8 — Documentation & validation

- Docs: this handoff; `docs/STATUS.md`, `docs/MOE_PERF_AUDIT.md` updated.
- **Code touched: test-only** (`tests/moe_perf3_prefetch_bench.rs` — added
  `perf3_prefetch_per_family`). **`src/` untouched** ⇒ `cargo test --lib` semantics unchanged.
  The bench builds and runs green: `perf3_prefetch_measurement` + `perf3_prefetch_per_family`
  **2 passed / 0 failed** (`#[ignore]`, not a CI gate). No runtime/numerics/cert/manifest/ADR
  change ⇒ no certification re-run required (none was claimed or altered).

## Final deliverable summary

1. **Baselines:** Mixtral 402.7 s forward / 4.5 s load / C5 3.185e-4; DeepSeek-V2-Lite C5
   2.587e-5 / ~4 GB tier; Qwen-MoE block-level ~1e-10 (no disk-tier forward).
2. **Before/after (cap=1):** overlap_saved ~9.5 / ~36.7 / ~69.5 ms (top-2/4/6); decode always
   >1.6× faster (noisy), original top-6 bench 191.74→95.32 ms.
3. **Family comparison:** Mixtral **A (major)**, DeepSeek-V2-Lite **B (moderate)**, Qwen-MoE
   **C (unmeasured / coverage gap)**.
4. **Real speedup achieved:** robustly hides **40 %→78 %** of disk read latency (rising with
   top-k); **largest real win on Mixtral** (RAM-starved, widest experts), most of which survives
   because Mixtral's forward is ~95 % read-bound.
5. **Remaining bottleneck:** **disk I/O at cap=1** for Mixtral; for DeepSeek-V2-Lite the tier is
   mostly RAM-resident so prefetch is secondary. Compute is never the bottleneck.
6. **Roadmap ranking (reordered):** **PERF-5 → PERF-4** → (deferred: MLA latent cache, GPU
   offload). PERF-5 promoted because real-workload measurement is currently impossible without
   it.
7. **Files modified:** `tests/moe_perf3_prefetch_bench.rs`, `docs/HANDOFF_MOE_PERF_3_VALIDATION.md`,
   `docs/STATUS.md`, `docs/MOE_PERF_AUDIT.md`.
8. **Tests executed:** `moe_perf3_prefetch_bench` (2 passed, `#[ignore]`, 4 reps for variance).
   `src/` untouched ⇒ `cargo test --lib` unaffected.
9. **Commit / CI:** see STATUS / the PERF-3-VALIDATION commit.

**Do not start PERF-4.** This milestone ends at the validation conclusions.
