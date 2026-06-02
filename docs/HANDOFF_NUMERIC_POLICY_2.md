# HANDOFF — NUMERIC-POLICY-2: quantized (int8) expert tier

MOE-IO-1 proved the dominant generation cost is the **disk-tier resolve** (~50 %
of the wall), bottlenecked by the antivirus scanning the expert files on first
open — a **per-byte** cost. The only code lever is **fewer bytes**. This block
adds a **per-row int8 expert tier**: ~half the bf16 tier bytes → ~half the
AV-scan / NVMe on resolve, certified by tolerance vs Certified f64, with a safe
fallback. Predecessor: `f9f7e9f`.

## FASE 1 — Audit

- Tier today: experts bf16 (`numel*2`), backend bf16/f32, read via
  `read_named_to_f32` (dtype-by-size) and `from_tier` (size-detect). Manifest v4
  (per-tensor dtype). Resolve = `materialize` (read + decode to f32).
- **Quantizable:** the routed + shared **expert** weights (the bulk of the tier
  and ~all of the resolve bytes). **Not** quantised: router, shared-gate, embed,
  lm_head, attention (small, accuracy-sensitive, kept bf16/f32).
- **Format:** per-row **symmetric int8** (scale per output row) — the standard,
  most accurate weight quantization; ~1 byte/element + a tiny scale header.
- **Risk:** lossy → must certify (tokens / argmax) vs Certified; fall back to
  bf16/f32 if not enabled. Numerically de-risked **first** (see FASE 7).

## FASE 7 (done first) — Cheap certification before building the tier

A `ATENIA_MOE_QUANT_SIM=int8` mode quantise→dequantises the resolved expert
weights per-row **in memory** (reusing the bf16 tier, no cold rebuild) to
reproduce the int8 tier's *numerical* effect. On the real Qwen1.5-MoE:

```
max-new 2:  reference [16,15]                       int8-sim [16,15]            MATCH
max-new 8:  reference [16,15,15,15,15,15,15,15]     int8-sim [identical]        MATCH
```

**int8 certifies — identical tokens (8 tokens).** This gated the full
implementation.

## FASE 2-6 — Design + tier + dequant + manifest + policy

- `disk_tier.rs`: `DiskDtype::QInt8`; `quantize_per_row_i8` / `dequantize_per_row_i8`;
  `write_qint8_tensor_named` (file = `[rows×f32 scales][numel×i8]`);
  `read_qint8_to_f32`; `qint8_disk_bytes`.
- `residency.rs`: `TierFmt { F32, Bf16Auto, QInt8 }`; `write_or_reuse_expert`
  writes the chosen format; `place_at`/`from_dense_at`/`from_real_layer_at` thread
  `TierFmt`; `materialize` and `from_tier` dequantise QInt8 (rows = shape[0]).
  Router/shared-gate stay bf16. `TierEntry.bytes` records the real on-disk size.
- `tensor.rs`: the `Disk` arms of `from_disk` / `copy_to_cpu_vec` / `ensure_cpu`
  handle QInt8 (dequant to f32).
- `runtime.rs`: `ATENIA_MOE_TIER_QUANT=int8` → `expert_tier_fmt` = QInt8; manifest
  **v5** (per-entry `dtype` + `bytes`); v4→v5 bump rebuilds once.
- **Policy:** the quantized tier is a **storage** choice (env), orthogonal to the
  compute `NumericPolicy` — the dequantised f32 flows through Certified(f64) /
  Strict(f32). It is inherently non-bit-exact, so it carries its **own tolerance
  certificate** (vs Certified), exactly like a `Fast` path. (No new policy enum:
  it composes with the existing ones; documented here.)

## FASE 6 — Fallback

`ATENIA_MOE_TIER_QUANT` unset → bf16 tier (default, unchanged). A non-quantisable
case is not possible (every f32 row quantises; an all-zero row → scale 1, q 0).
Manifest version/size mismatch → certified shard path (existing fallback). The
warm `from_tier` size-detects f32/bf16/qint8; an unrecognised size → fallback.

## FASE 10 — Tests

- `disk_tier`: per-row quant/dequant round-trip within `scale/2`; all-zero row
  exact.
- `tests/moe_qint8_tier_test` (new): cold int8 load → experts (routed + shared)
  are **qint8** with `rows*4+numel < numel*2` (smaller than bf16), router **not**
  quantised; warm reconstruct **after deleting the shards** == cold (the int8
  bytes dequantise identically).
- Full MoE regression + lib suite green; default (bf16) path unchanged.

## FASE 8-9 — Real benchmark (Qwen1.5-MoE, prompt `22,25,29`, max-new 2, CPU)

| Tier | warm wall | resolve (disk) | tier on disk | token ids | certified? |
|---|---|---|---|---|---|
| bf16 (NUMERIC-POLICY-1 baseline) | ~180 s | ~86–100 s @ 68 MB/s | 26.7 GiB | `16, 15` | reference |
| **int8 (this)** | **142 s** | **45.3 s @ 287 MB/s** | **14.3 GiB** | `16, 15` | **yes (identical)** |

Cold (writes int8 tier): **1543 s** (vs ~1942 s bf16 — fewer bytes written).

**Headline:** int8 expert tier is **certified** (tokens identical `16, 15`) and a
**real, measurable win**: tier **26.7 → 14.3 GiB (−46 %)**, resolve **~90 s →
45 s (~−50 %, and 68 → 287 MB/s)**, warm wall **~180 → 142 s (~−21 %)**. The
MOE-IO-1 thesis — *the bottleneck is bytes; halve them* — is confirmed.

## FASE 9 — ROI

- **Tier size:** 26.7 → **14.3 GiB (−46 %)** (experts ≈ 1 B/elem vs bf16 2 B).
- **Resolve:** ~90 → **45 s (~−50 %)**, throughput 68 → 287 MB/s — exactly the
  "fewer bytes for the AV to scan" lever MOE-IO-1 pointed at.
- **Wall:** ~180 → **142 s (~−21 %)** warm; cold 1942 → 1543 s.
- **Numerical:** **zero token drift** — `16, 15` (max-new 2) and the 8-token
  sequence are identical to Certified; the certificate passes.
- **int4?** **Not yet — but promising.** int8 certified with full margin (0
  drift). int4 would roughly halve the tier again (~7 GiB, resolve ~25 s), but
  per-row symmetric int4 is far lossier (16 levels vs 256) and likely to flip
  tokens. **Recommendation:** run the same cheap **sim certification** first
  (`ATENIA_MOE_QUANT_SIM=int4`-style) and only implement if it certifies — never
  on intuition. Group-wise int4 (e.g. 64-wide groups) is the more likely-to-
  certify variant if plain per-row fails.

## Files

- `src/tensor/disk_tier.rs` — int8 quant/dequant, QInt8 dtype, write/read.
- `src/moe/residency.rs` — `TierFmt`, qint8 write/read/dequant, `TierEntry.bytes`,
  int8 sim.
- `src/tensor/tensor.rs` — QInt8 `Disk` arms.
- `src/moe/runtime.rs` — `ATENIA_MOE_TIER_QUANT`, manifest v5.
- `tests/moe_qint8_tier_test.rs` (new) + `docs/HANDOFF_NUMERIC_POLICY_2.md` (this)
  + `docs/STATUS.md`.

Default (bf16) unchanged; Certified f64 reference; Strict f32 fallback.
