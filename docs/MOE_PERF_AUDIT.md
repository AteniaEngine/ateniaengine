# MOE-PERF-1 — MoE Performance Audit & Optimization Roadmap (measurement only)

**Audit only — no runtime / numerics / certification / manifest / ADR change.** This
milestone measures where MoE time goes and produces a prioritized roadmap. The only code
added is a **test-only** `#[ignore]` timing bench (`tests/moe_perf_scale_bench.rs`) that
*calls* the existing runtime; nothing in `src/` changed.

Terminology unchanged: `MoE-certified L3 = active-path-certified`, **not** the dense
ADR-004 `CERTIFIED`; **L4** (global F64) reserved/unreachable.

---

## PHASE 1 — Existing performance instrumentation (inventory)

| Area | What exists | Where |
|---|---|---|
| **Tier load timing** | `[ATENIA] tier warm reconstruct: total / validate / layers / embed+lm_head` (s); cold build `reused / written / tensors / GiB` | `moe/runtime.rs` (`ATENIA_MOE_CACHE_STATS=1`) |
| **Expert cache stats** | `CacheStats { hits, misses, evictions, bytes (read), shared_hits/misses }` per layer cache; `ExpertCache::stats()` | `moe/residency.rs` |
| **Per-forward residency** | `MoeResidencyTrace { materialized_bytes }`; `resident_ram_bytes()`, `full_materialization_bytes()` | `moe/residency.rs` |
| **Per-op compute timing** | node-level timing report (top-N by total time) | `amg/graph.rs` (`ATENIA_NODE_TIMING=1`) |
| **MatMul trace** | per-matmul time report (top-N) | `amg/graph.rs` (`ATENIA_MATMUL_TRACE=1`) |
| **Loader / matmul counters** | `CounterSnapshot` deltas (vram/disk fast/slow paths, bf16/int8 matmul, disk-streamed) for load + generation | `cli_generate.rs` |
| **Generate wall / tok-s** | load secs, total secs, tok/s, prefill heartbeat (first-token visible) — **dense path** | `cli_generate.rs` |
| **Disk tier I/O** | typed read/write (`read_*_to_f32`, `write_*_tensor_named`), bf16/qint8 on-disk dtypes, antivirus-aware sizing notes | `tensor/disk_tier.rs` |

**Already measurable:** load (tier reconstruct + cold build), per-layer expert cache
hit/miss + bytes, per-forward materialized bytes, resident vs full RAM, per-op compute, per-
matmul time, loader path counts. **Gaps (not blocking this audit):** (a) no single
aggregated profile that sums load/prefill/decode/router/attention/expert in one report; (b)
the **MoE generate** path (`run_moe_text → controlled_moe_generate → rt.generate`) does not
emit prefill-vs-decode split / first-token latency / tok-s like the dense path does; (c) no
global disk-tier read-latency / bytes-read counter (per-layer `CacheStats.bytes` covers it
indirectly). None blocks measurement; (b) is the one worth closing in PERF-5 (instrumentation
parity), zero numerical impact.

---

## PHASE 2 — Instrumentation used / minimal addition

No new runtime instrumentation was needed: the existing env-gated timers + `CacheStats` +
`MoeResidencyTrace` already cover load / expert-I/O / per-op compute with **zero overhead
when disabled** (all behind `ATENIA_*` env checks or returned structs). The audit therefore
adds **only** a reproducible measurement harness:

- `tests/moe_perf_scale_bench.rs` (`#[ignore]`, test-only): times `load_from_files` /
  `forward_logits` (prefill) / `generate` (decode) + tok/s for the three families'
  scale fixtures. It only *calls* `MoeRuntime`; no `src/` change, not run in CI (timing is
  not a gate). Run: `cargo test --release --test moe_perf_scale_bench -- --ignored --nocapture`.

---

## PHASE 3 — Real measurements

### 3a. Fresh compute measurement (this audit) — scale fixtures, reduced dim

Topology-representative fixtures (real expert count / top-k / GQA / MLA / shared, reduced
hidden dim), RAM backend (no disk tier), release build, 24-thread APX, this 32 GB host:

| Fixture | Family | Load | Prefill (seq=5) | Decode tok/s |
|---|---|---|---|---|
| `mixtral_scale` | Mixtral | **13.50 ms** | **3.07 ms** | **219.7 tok/s** |
| `qwen_scale` | Qwen-MoE | **8.14 ms** | **2.69 ms** | **223.6 tok/s** |
| `deepseek_scale` | DeepSeek-MoE (MLA) | **18.71 ms** | **0.37 ms** | **3397 tok/s** |

**Finding:** at reduced dim the **routing + attention + expert + MLA compute is trivial**
(sub-ms to a few ms; hundreds–thousands tok/s). The MoE *mechanism* is not the bottleneck.
The real-world cost is **weight volume → disk/RAM I/O at real scale**, isolated out here.

### 3b. Real-weight measurements (captured from the certification runs)

Real numbers from this project's actual L3 certification runs (logs / handoffs — not
estimates; full real-weight re-runs were **not** repeated here: an 87 GB Mixtral load is a
~1 h heavy operation that previously caused host memory pressure, and re-running it only to
re-time it is out of scope for an audit and against the "no unexpected heavy run" guard):

| Model | Real measurement | Source |
|---|---|---|
| **Mixtral-8x7B** (real, 87 GB) | **cold tier build ≈ 88 GB written** (one pass, minutes of NVMe writes); **warm tier reconstruct (load) = 4.5 s**; **forward (prefill seq=4) = 402.7 s** with bounded expert cache (`ATENIA_MOE_EXPERT_CACHE=1`); test wall 955.74 s | MIXTRAL-CERT-3 / `HANDOFF_MIXTRAL_CERT_C5.md` |
| **Mixtral** default cache | per-layer cache of `2·top_k = 4` reconstructed-**F32** experts × 32 layers ≈ **90 GB commit → OOM** | MIXTRAL-CERT-3 |
| **Mixtral on HDD** | ~87 GB read ≈ **~1 h** (HDD + antivirus); mitigated by copying to NVMe SSD | MIXTRAL-DATA / CERT runs |
| **DeepSeek-V2-Lite** (real, 15.7 B) | MLA-2 **disk tier ≈ 4 GB RAM** resident (vs ~58 GB as f32); C5 whole-model `2.587e-5` | MLA-2/3 / `HANDOFF_MLA_3.md` |

**Finding:** real-scale time is **dominated by weight I/O**, not compute:
- **Startup:** cold tier build (read shards + write ~88 GB tier) is the largest one-time
  cost; warm reconstruct is then **4.5 s** (≈ 90× cheaper) — the persistent tier already
  converts most startup cost into a one-time build.
- **Prefill/decode:** the **402.7 s** Mixtral forward (seq=4) is ~100 s/position, dominated
  by **expert re-materialization** — `cache=1` re-reads each layer's experts from NVMe every
  position (it cannot keep them: `cache=4` would fit in compute but commits ~90 GB → OOM).
  The expert FFN is wide (`intermediate_size 14336`), so each expert is ~176 M params =
  **704 MB F32** (decoded from a 352 MB bf16 tier file) — the F32 decode doubles bytes.

---

## PHASE 4 — Bottleneck ranking (evidence-backed)

1. **Cold tier build / first-ever load I/O** — read 87 GB shards + write ~88 GB tier; the
   biggest one-time cost (minutes on NVMe, ~1 h on HDD). *Evidence: 88 GB tier, HDD ~1 h.*
2. **Expert re-materialization during forward (RAM-bounded cache thrash)** — the dominant
   *recurring* cost: Mixtral forward 402.7 s (seq=4) with `cache=1` re-reading experts every
   layer. The cache-capacity ↔ RAM tradeoff is the core tension (`cache=4` → 90 GB OOM).
   *Evidence: 402.7 s forward; cache=1 ↔ cache=4 OOM.*
3. **F32 expert decode (2× byte amplification)** — experts stored bf16 (352 MB) but decoded
   to F32 (704 MB) for compute; doubles RAM pressure and the bytes the cache holds.
   *Evidence: 704 MB/expert F32; 90 GB at cache=4.*
4. **Warm reconstruct backend reads** — 4.5 s (validate + per-layer attn/router + embed/lm_head).
   Small relative to (1)/(2) but the irreducible warm-load floor. *Evidence: 4.5 s, broken
   down by CACHE_STATS.*
5. **Antivirus scanning of tier files (Windows)** — per-file scans inflate read latency;
   qint8/bf16 tiers reduce bytes scanned. *Evidence: `disk_tier.rs` notes; HDD ~1 h.*
6. **Compute (routing / attention / MLA / expert FFN matmul)** — **NOT a bottleneck** at any
   measured scale (sub-ms reduced dim; APX/rayon-parallel at scale, I/O-bound). *Evidence:
   PHASE 3a sub-ms; 219–3397 tok/s.*

---

## PHASE 5 — Optimization candidates

For each: expected gain · complexity · risk · families affected.

### Low risk / High ROI
- **A. bf16-resident expert cache (skip F32 decode in cache)** — keep cached experts in
  **bf16** (352 MB) and decode per-use, or compute in bf16 where the policy allows; **halves**
  cache RAM → a larger cache fits without OOM → fewer re-reads. *Gain: large (attacks #2+#3);
  Complexity: M; Risk: Low–Med (numerics: must stay within the certified gate — bf16 compute
  is a policy already present); Families: all (esp. Mixtral wide FFN).* **No numerics change
  if it only changes cache storage, not the compute dtype.**
- **B. Default `ATENIA_MOE_EXPERT_CACHE` auto-sized to free RAM** — pick the largest cache
  that fits a RAM budget instead of the fixed `2·top_k` (which OOMs at real scale). *Gain:
  large (avoids the cache=4 OOM, enables a bigger safe cache); Complexity: S; Risk: Low
  (operational default; numerically identical); Families: all.*
- **C. qint8 expert tier as the default for huge models** — `disk_tier` already supports
  qint8 (¼ bytes) → smaller tier, less I/O, less antivirus scan. *Gain: Med–large on startup
  + reads; Complexity: S–M (already implemented, gate it on/cert it); Risk: Med (qint8 needs a
  numeric certificate — NUMERIC-POLICY-3 already gates this); Families: all.* **Certification
  required before default — out of a pure perf milestone.**

### Medium risk
- **D. Expert prefetch / async tier reads** — overlap NVMe reads of the next layer's selected
  experts with current-layer compute. *Gain: Med (hides read latency under compute);
  Complexity: M–L; Risk: Med (concurrency); Families: all.*
- **E. Parallel cold tier build** — write the ~88 GB tier with parallel per-shard workers
  (today serial-ish to bound RAM). *Gain: Med on the one-time build; Complexity: M; Risk: Med
  (RAM pressure — must stay bounded); Families: all.*
- **F. MoE-generate instrumentation parity (PERF-5)** — emit prefill/decode split, first-token
  latency, tok/s + cache hit-rate for `controlled_moe_generate` (today only dense `generate`
  reports them). *Gain: visibility (enables future perf gates); Complexity: S; Risk: Low
  (timing only, zero numerics); Families: all.*

### High risk
- **G. MLA latent/absorb KV cache** — the absorbed-KV decode path (perf, not correctness).
  *Gain: large for long-context decode; Complexity: L; Risk: High (new attention math path);
  Families: DeepSeek (MLA).* **Explicitly deferred (not in any near-term perf milestone;
  correctness path untouched).**
- **H. GPU/VRAM expert offload for MoE** — stream experts to VRAM. *Gain: large; Complexity:
  L; Risk: High (the MoE path is CPU-only today); Families: all.*

---

## PHASE 6 — Optimization roadmap (ROI-ordered)

| Milestone | Scope | Expected benefit | Validation |
|---|---|---|---|
| **PERF-2 — auto-sized + bf16-resident expert cache** (B + A) | Auto-size `ATENIA_MOE_EXPERT_CACHE` to a RAM budget; store cached experts bf16 (decode per-use or compute-policy bf16). No change to the certified compute dtype. | **Highest ROI:** removes the cache=4 OOM *and* the cache=1 re-read thrash → the 402.7 s Mixtral forward should drop sharply; halves cache RAM. | Re-run the C5 active-path harnesses (Mixtral/Qwen/DeepSeek) — `max_abs_diff < 0.5` + argmax exact must still hold (numerics unchanged); measure forward time delta via the bench. |
| **PERF-3 — expert prefetch / async tier reads** (D) ✅ **DONE** | Overlap a token's selected-expert NVMe reads under the existing rayon pool (within-forward; next-layer is impossible — sequential dependency). Opt-in `ATENIA_MOE_PREFETCH=1`. | **Measured ~2× faster decode at cap=1, top-6** (191.74→95.32 ms); bit-exact, scale-cert 3/3 with prefetch on. See `HANDOFF_MOE_PERF_3_PREFETCH.md`. | C5/scale-cert parity (done: bit-identical, deterministic); bench forward time (done). |
| **PERF-4 — qint8 default tier for huge models (gated on a cert)** (C + E) | Make qint8 the default expert tier for models above a size threshold; parallel cold build. Requires a passing qint8 numeric certificate (NUMERIC-POLICY-3) first. | Med–large on startup + reads (¼ bytes, less AV scan; faster build). | qint8 numeric cert must pass before default; C5 parity within the qint8 envelope; bench build + load. |
| **PERF-5 — MoE-generate instrumentation parity** (F) ✅ **DONE** | `MoeGenTelemetry` (load / prefill / decode / first-token / tok-s + disk-tier expert-cache / prefetch / tier I/O) via `generate_instrumented` + `controlled_moe_generate_instrumented`; opt-in `ATENIA_MOE_TELEMETRY=1` (default unchanged). Bit-identical generation. | Unblocks measuring PERF-3/PERF-4 on real certified runs (the PERF-3-VALIDATION blocker). | `moe::telemetry` 4/4 + `moe_perf5_telemetry_test` 3/3 (bit-identical to `generate`); scale-cert 3/3 (no regression); `cargo test --lib` 886/0. See `HANDOFF_MOE_PERF_5.md`. |

Deferred (not scheduled): **G** (MLA latent cache) and **H** (GPU expert offload) — high risk,
separate tracks.

### PERF-3-VALIDATION reorder (evidence-driven, 2026-06-09)

`MOE-PERF-3-VALIDATION` measured the prefetch win at each certified family's **real top-k** on
the certified disk-tier `forward_cached` path (Mixtral top-2 / Qwen top-4 / DeepSeek top-6,
cap=1; the 87 GB Mixtral re-run was excluded as the documented OOM hazard). Findings:
**Mixtral = major win** (only RAM-starved, read-bound, widest-expert family; cap=1 *forced*),
**DeepSeek-V2-Lite = moderate** (top-6 ⇒ highest overlap fraction ~78 %, but ~4 GB tier mostly
RAM-resident ⇒ few real misses; MLA orthogonal), **Qwen-MoE = unmeasured** (block-level cert,
no disk-tier whole-model forward). Robust metric `overlap_saved` rose monotonically with top-k
(~9.5 → 36.7 → 69.5 ms; 40 % → 69 % → 78 % of read latency hidden); misses identical OFF/ON.

The validation **could not measure prefetch on a real certified run** because the MoE-generate
path lacks the dense path's prefill/decode/cache-hit telemetry (PERF-1 gap (b)). That promotes
**PERF-5 (instrumentation parity) AHEAD of PERF-4**: PERF-5 is the prerequisite to ever
validating PERF-3/PERF-4 on real models without a full-reload gamble. **PERF-4 (qint8 default
tier) remains high user ROI** (attacks expert *bytes* — the root — where PERF-3 attacks read
*latency* — the symptom; complementary), now **second**. See `HANDOFF_MOE_PERF_3_VALIDATION.md`.

### PERF-5-REAL-MEASURE: telemetry baseline + PERF-4 confirmed next (2026-06-09)

With PERF-5 landed, `MOE-PERF-5-REAL-MEASURE` captured the first real cache/prefetch/tier
telemetry on the certified `forward_cached` path (disk-tier scale fixtures at real top-k; the
87 GB Mixtral and 29 GB DeepSeek real runs were classified **too-heavy/skip** on a 12-GB-free
host — measure-only, no behavior change). The telemetry **confirms cache capacity is the dominant
cost**: `auto→cache=1` collapses the hit rate (Mixtral 6→20 misses +18 evictions; Qwen 19→40 +38)
— the PERF-1 cache↔OOM tension made concrete. Prefetch is observable and scales with top-k
(`parallel_prefetches`=misses; overlap Mixtral≈0.4 < Qwen≈1.9 ms). DeepSeek/MLA streams experts
uncached (timing-only — a coverage limitation). **Decision: PERF-4 (qint8 tier) remains next** —
it cuts read bytes ¼ **and** ~4×s the cache capacity at fixed RAM (relieves the measured thrash;
Mixtral cache=4 → ~22 GB vs ~90 GB OOM), compounding with prefetch. Highest risk: the qint8
numeric certificate. See `HANDOFF_MOE_PERF_5_REAL_MEASURE.md`. **Do not start PERF-4.**

---

## Validation

- `tests/moe_perf_scale_bench.rs` runs green (`#[ignore]`; release, --nocapture).
- `cargo test --lib` green (the bench is a `--test` integration target, not `--lib`; no `src/`
  change, so the lib suite is unaffected).

## Files

- `tests/moe_perf_scale_bench.rs` (new, test-only `#[ignore]` bench).
- `docs/MOE_PERF_AUDIT.md` (this audit + roadmap).

**Do not start PERF-2.** This milestone ends at the roadmap.
