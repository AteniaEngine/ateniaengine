# HANDOFF — MOE-PROD-7: BF16 backend tier + fast bulk reader

Milestone: **MOE-PROD-7** — attack the **warm reconstruction / backend loading**
cost that MOE-PROD-6 left as the new dominant term, without touching the MoE
math, routing, or outputs (bit-exact, safe fallback). Two levers, both reusing
the certified MOE-PROD-6 bf16 mechanism:

1. **BF16 backend tier** — the non-expert tensors (embed_tokens, lm_head,
   per-layer attention q/k/v/o + norms + biases, router, shared-gate) now also
   persist as **bf16** when losslessly representable (auto-detected per tensor),
   halving the bytes the warm reconstruction reads back. Experts were already
   bf16 (MOE-PROD-6); this extends it to the ~9 GB f32 backend.
2. **Fast bulk reader** — `read_f32_named` streamed the file then decoded it
   **element-by-element** (`from_le_bytes` in a loop over ~840 M elements). It
   now streams directly into the destination `Vec<f32>` byte view (memcpy-speed,
   bounded peak), bit-identical on little-endian targets. New
   `read_named_to_f32` detects the on-disk dtype by file size (`numel*4`→f32,
   `numel*2`→bf16, upcast via the SIMD `bf16_decode_bulk`).

## FASE 1 — Audit (where the warm 204 s actually goes) — KEY FINDING

MOE-PROD-6 *guessed* "warm reconstruction + CPU matmul" was dominant. The
profiling instrumentation added here (`ATENIA_MOE_CACHE_STATS=1` prints
`tier warm reconstruct: total=… (validate, layers[experts+attn], embed/lm_head)`)
**measures** it on the real model and **overturns the guess**:

```
tier warm reconstruct: total=34.6s (validate=1.57s, layers[experts+attn]=16.2s, embed/lm_head=16.8s)
warm wall (load + 2 tokens) = 204 s
```

**The whole warm reconstruction is only 34.6 s of 204 s (~17 %).** The other
**~168 s (~82 %) is generation** — 4 MoE forward rows × 24 layers ≈ 42 s/row,
dominated by the **CPU matmul** of the routed + shared expert FFNs and the
attention GEMMs (the per-token expert NVMe read is only ~3 s/row at this point).
So **backend loading was never the real bottleneck** — generation compute is.
This redirects the optimization block (see FASE 5).

## FASE 2 — Implementation

- `tensor/disk_tier.rs`: fast `read_f32_named` (stream into the `Vec<f32>` byte
  view) + `read_named_to_f32` (dtype-by-size → f32) + unit test.
- `moe/residency.rs`: `write_or_reuse(..., allow_bf16)` (delegates to the bf16
  expert path); `from_tier` reads router + shared-gate via `read_named_to_f32`.
- `moe/runtime.rs`: `persist_backend(..., allow_bf16)` (bf16 the backend); the
  warm `read` closure uses `read_named_to_f32`; per-phase profiling; manifest
  **v4** (backend may be bf16). The v3→v4 bump rebuilds an existing tier once.
- Default unchanged when the persistent tier is off; `ATENIA_MOE_TIER_BF16=0`
  forces f32 (backend + experts).

## FASE 3 — Bit-exactness + tests

- `disk_tier::test_read_named_to_f32_detects_dtype_by_size` (f32 direct, bf16
  upcast exact, half size, wrong size rejected).
- `tests/moe_bf16_tier_test` now also covers a bf16 backend (router/shared-gate
  were the bug that surfaced: `from_tier` read them f32 while the writer made
  them bf16 → fixed to `read_named_to_f32`); warm reconstruction after deleting
  the shards is bit-exact.
- Full MoE regression + lib suite green.

## FASE 4 — Benchmark (real Qwen1.5-MoE-A2.7B, prompt `22,25,29`, max-new 2)

| | MOE-PROD-6 warm (bf16 experts, f32 backend) | MOE-PROD-7 warm (bf16 backend + fast reader) |
|---|---|---|
| Total wall (load + 2 tokens) | **204 s** | **204 s** (unchanged — gen-bound) |
| of which: warm reconstruct (load) | ~50 s (est.) | **34.6 s** (measured) |
| of which: generation | ~150 s | **~168 s** |
| Tier on disk | 28.6 GiB | **26.7 GiB** |
| Cold wall / tier write | 1942 s | **1755 s** |
| Output (token ids) | `16, 15` | `16, 15` — **bit-exact** |

**Headline (honest):** MOE-PROD-7 made the **warm load** faster (reconstruct now
**34.6 s**; tier 28.6 → 26.7 GiB; cold 1942 → 1755 s) and bit-exact, **but the
total warm wall is unchanged (204 s)** because **generation compute dominates
(~168 s, ~82 %)**. The real value of this milestone is the **profiling that
proved load is not the bottleneck** — load-side ROI is now exhausted (eliminating
the entire 34.6 s reconstruction would save <17 %).

## FASE 5 — Audit / block redirection

- **Real gain:** warm **reconstruction** 34.6 s (was larger), tier −1.9 GiB,
  cold −187 s, all bit-exact — but **0 % on total warm wall** (gen-bound).
- **New dominant bottleneck (measured):** **generation compute** — the CPU
  matmul of the routed + shared expert FFNs and attention GEMMs (~168 s of
  204 s, ~82 %). The per-token NVMe read (MOE-PROD-3/6 domain) and the warm
  reconstruction (MOE-PROD-5/7 domain) are now both small terms.
- **Block status:** the **warm-load / backend-loading block is complete** — its
  ROI is exhausted. The next block targets **generation compute**, which is an
  architectural decision (GPU offload of the MoE expert/attention GEMMs via the
  existing M8.7 disk→VRAM streaming path, vs CPU GEMM parallelisation) — see
  the session report.

## Files modified

- `src/tensor/disk_tier.rs` — fast `read_f32_named`, `read_named_to_f32`, test.
- `src/moe/residency.rs` — `write_or_reuse(allow_bf16)`, `from_tier` dtype reads.
- `src/moe/runtime.rs` — `persist_backend(allow_bf16)`, warm `read` dtype, phase
  profiling, manifest v4.
- `docs/HANDOFF_MOE_PROD_7.md` (this) + `docs/STATUS.md`.

No new architecture/family/math/graph ops; routing and outputs unchanged;
default path unchanged.
