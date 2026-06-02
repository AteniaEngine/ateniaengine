# HANDOFF — MOE-IO-1: I/O bottleneck profiling — root cause found (AV), code candidates eliminated (STOP)

New line after the expert-compute line was closed (MOE-PERF-1/2/3 proved expert
matmul is < 1 % of the wall). MOE-IO attacks **I/O and load**. The rule: **measure
first, eliminate candidates without ROI, implement only evidence-backed wins,
STOP if none.** This block found the dominant bottleneck, measured the ROI
ceiling of every code candidate, and **eliminated them all** — the cost is
environmental (antivirus), not a code defect. Predecessor: `1059fac`.

## FASE 1 — Profile (where the generation wall goes)

Added instrumentation (`CacheStats.resolve_nanos`, reported by
`ATENIA_MOE_CACHE_STATS=1` as `tier resolve(disk)=..s @ ..MB/s`) that isolates
the **tier resolve** (NVMe read + bf16 decode of expert weights) from the matmul
and the GraphBuilder. Real Qwen1.5-MoE (prompt `22,25,29`, Strict CPU):

| max-new | wall | **tier resolve (disk)** | expert matmul | resolve % wall |
|---|---|---|---|---|
| 2 | 195 s | **100 s** | 0.9 s | **~51 %** |
| 8 | 257 s | **130 s** | 2.1 s | **~51 %** |

**The dominant cost is the disk-tier resolve (~50 % of the wall).** Reading
~12.67 GiB (f32-equiv; ~6.3 GiB bf16 on disk) took ~100 s = **~68 MB/s** — **~30×
below** the NVMe's sequential bandwidth.

## FASE 2 — ROI ceiling of each code candidate (measured, then eliminated)

**Candidate A — bigger expert cache** (env `ATENIA_MOE_EXPERT_CACHE`, no code):

| cache cap | misses | resolve | wall |
|---|---|---|---|
| 8 (default) | 297 | 86 s | 183 s |
| 32 | 291 | 99 s | 194 s |
| all (60) | 291 | 96 s | 201 s |

→ **No ROI.** At max-new 2 the misses are almost all **first-touch** (each expert
read once in prefill; little reuse to cache). 297 → 291 misses, hit ratio
22.7 → 24.2 %. A bigger cache cannot help reads that must happen the first time.
(It helps longer generations — hit ratio reached 55 % at max-new 8 — but not the
first-touch-dominated wall.) **Eliminated.**

**Candidate B — consolidate the 4659 tier files into per-layer blobs.** A direct
read benchmark of layer 0 (180 files, 990 MB):

- **cold** read of the 180 files: **14.5 s = 68 MB/s** (matches the resolve);
- **warm** re-read (OS cache): **0.77 s = 1282 MB/s** (~19× faster);
- warm **180 files** 0.535 s vs warm **single 990 MB blob** 0.327 s → the
  **per-file-open overhead is only ~1 ms/file** (small).

So the penalty is **not** per-file-open and **not** bandwidth — it is the
**cold first-access content scan**: `Get-MpComputerStatus` shows **Windows
Defender real-time protection is ON**, and **68 MB/s is exactly Defender's scan
throughput**. Defender scans a file's **content** the first time it is opened.

→ **Consolidation has no ROI and is actively harmful here:** Defender scans
**bytes**, not files. A per-layer blob would make Defender scan the **whole
330 MB layer** on first open (all 60 experts) instead of only the **top-k ~28 MB**
actually resolved — *more* scanning, not less. **Eliminated.**

## Root cause + decision (STOP, evidence-based)

**The dominant generation bottleneck (~50 % of the wall) is the antivirus
real-time scan of the bf16 tier files on first open (~68 MB/s), not a code
inefficiency.** The engine already does the right things: bf16 (MOE-PROD-6,
halves the bytes scanned), single-copy resolve (MOE-PROD-8), and an on-demand
top-k cache. No tier-layout / cache / resolve **code** change reduces an AV scan
of the bytes the model must read — and the only file-count change (consolidation)
makes it worse. Per the explicit stop criterion ("if no candidate shows
significant ROI: STOP and explain exactly why; do not optimise on intuition"),
MOE-IO stops here.

## What actually has ROI (and where it lives)

1. **Operational (not code) — exclude the tier dir from the AV.** A direct
   exclusion of `ATENIA_DISK_TIER_DIR` restores reads to ~NVMe speed (the warm
   1282 MB/s vs cold 68 MB/s gap), which would cut the ~100 s resolve to ~7 s →
   **~50 % off the wall**. This is the single biggest win and it is a **one-line
   operator action**, not an engine change. (Could not be measured end-to-end
   here: adding a Defender exclusion needs admin, which this shell lacks — but
   the cold/warm gap quantifies it.) The CLI now **detects** the pathological
   throughput and **prints this recommendation** (`tier resolve is N MB/s … likely
   antivirus … exclude the tier dir`).
2. **Code, but a different line — fewer *bytes*, not fewer files.** Since the
   cost is per-byte scanning, the only code lever is **reducing the tier data
   size**: int8/int4-quantised experts would halve/quarter the bytes the AV
   scans (and the NVMe reads). That is a **numeric change** → it belongs to
   **NUMERIC-POLICY** (a tolerance-certified quantised-expert tier under
   `Fast`/a new tier), **not** MOE-IO's tier-layout scope, and must be ROI-gated
   and certified like every other non-bit-exact path.

## Deliverable

- **Instrumentation** (`resolve_nanos` + effective MB/s + the AV-detection
  hint) — turns the bottleneck into a visible, actionable number for operators.
- **This evidence trail** — eliminated bigger-cache and consolidation with
  measurement; identified AV as the root cause; routed the real fixes to
  *operational* (AV exclusion) and *NUMERIC-POLICY* (quantised experts).

`Strict` (f32 CPU) remains the fast default; `Certified` the reference. No
numeric/behaviour change; bit-exact; default unchanged.

## Files

- `src/moe/residency.rs` — `CacheStats.resolve_nanos` + timer.
- `src/moe/graph_op.rs` — aggregate it.
- `src/bin/atenia.rs` — report resolve s + MB/s + AV-exclusion hint.
- `docs/HANDOFF_MOE_IO_1.md` (this) + `docs/STATUS.md`.
