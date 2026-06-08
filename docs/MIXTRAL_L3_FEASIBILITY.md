# Mixtral-8x7B L3 Feasibility — MIXTRAL-L3-FEASIBILITY (audit + estimation only)

> **RESOLVED (MIXTRAL-CERT-3).** This feasibility estimate is now history: Mixtral-8x7B-v0.1
> **reached MoE-certified L3 (active-path-certified)** exactly as estimated — C5 active-path
> worst `max_abs_diff` `3.185e-4` < 0.5, argmax exact 4/4, deterministic, via a
> one-layer-at-a-time F64 reference + Atenia's real forward on a persistent disk
> expert-tier (bounded expert cache to fit 32 GB). See `docs/HANDOFF_MIXTRAL_CERT_C5.md`.
> The "still L0" wording below is superseded.
>
> **Post-MLA-3 note (de-risking evidence).** Since this audit, **DeepSeek-V2-Lite
> reached MoE-certified L3** via the exact path this doc estimates for Mixtral — a
> one-layer-at-a-time F64 reference + Atenia's real full forward on a **disk
> expert-tier** (~4 GB RAM, whole-model `2.587e-5`). That confirms the C5 active-path
> methodology end-to-end on a second real MLA/MoE family; Mixtral's remaining blocker
> is provisioning the ~93 GB weights, not the method. Mixtral is still **L0**.

**Audit + estimation only — NO download, NO code change, NO certification, NO
real Mixtral forward, no commits.** Determines whether ADR-007 **L3** (and L1/L2)
is realistically achievable for Mixtral-8x7B on the target laptop **before**
committing to a ~93.4 GB download. Every number is derived from the **real
Mixtral config**, **ADR-007**, and the **measured** Qwen-MoE L1–L3 runs — not
assumed.

Target hardware: **Intel i9-14650HX · 32 GB DDR5 · RTX 4070 Laptop 8 GB · NVMe ·
Windows.** Measured free RAM during the Qwen runs: **~20.5 GB** (34 GB total minus
OS/apps).

## FASE 1 — Mixtral-8x7B architecture (from the real config)

`models/Mixtral-8x7B-v0.1/config.json`:

| Field | Value |
|---|---|
| layers | **32** |
| hidden_size | **4096** |
| expert FFN (intermediate_size) | **14336** |
| routed experts / layer | **8** |
| experts_per_token (top-k) | **2** |
| shared expert | **none** |
| attention | GQA **32 heads / 8 kv** (4:1), no qkv bias |
| dtype | bf16 |
| total / active params | **46.7 B / 12.9 B** |

**Derived sizes (exact arithmetic):**
- **Per expert** = 3·d_ff·hidden = 3·14336·4096 = **176.16 M params**.
  - bf16 on disk: 0.35 GB · **F64: 1.41 GB** · F32: 0.70 GB.
- **Per MoE layer (8 experts)** = 1.409 B params → **F64 11.27 GB** / F32 5.64 GB /
  bf16 2.82 GB. (+ attention ~42 M params → F64 ~0.34 GB.)
- **Embed / lm_head** = 32000·4096 = 131 M each → F64 1.05 GB each (vocab 32 000 is
  *small* — smaller than Qwen's 151 936).

**Qwen-MoE reference points (measured, for scaling):**
- Qwen expert = 3·1408·2048 = 8.65 M (F64 69 MB). Mixtral expert is **20.4×** larger.
- Qwen MoE layer (60 exp) = 519 M (F64 4.15 GB). Mixtral layer is **2.72×** larger.
- Qwen C1 worst 4.768e-7 (1440 experts); C2 set-match all 24 layers; **C5 1.866e-4**
  end-to-end, 4-token input, **950 s** disk-tier; the C5 f64 reference generator
  (HF decoder-layer, one layer at a time) was **measured at ~16.4 GB resident**
  peak for a 4.15 GB f64 layer.

## FASE 2 — C1 (per-expert) feasibility

C1 runs **one expert at a time** (the whole point of decomposition): load one
expert's 3 matrices, compute SwiGLU in F64 (reference) / Atenia forward, compare.

- **Peak RAM:** one Mixtral expert F64 (1.41 GB) + the F32 copy / torch overhead
  (~2–3 GB base) ⇒ **~4–5 GB peak**. Atenia side (MoeDenseExpert, F32 ~0.70 GB/
  expert) is even smaller.
- **Disk:** reads each expert once → ~**90 GB** of expert bytes streamed over the
  run (sequential, NVMe-friendly). No extra disk written (reference ~ tens of MB).
- **Experts to check:** 8 × 32 = **256** (vs Qwen's 1440 — **fewer**, but each 20×
  bigger). Compute ≈ 256 × 176 M MACs (F64) ≈ 45 G MACs → seconds of CPU; the run
  is **I/O-bound** (~90 GB read).
- **Time:** dominated by the ~90 GB read; on NVMe ≈ **10–25 min**.

**Can C1 run on this notebook?** **Yes — comfortably** (one expert at a time keeps
RAM at ~5 GB). **Minimum free RAM: ~6 GB.** → **🟢 VERDE.**

## FASE 3 — C2 (router parity) feasibility

C2 = per-layer router logits → top-k **set equality** vs reference + routing
margin. The router is `[8, 4096]` per layer (tiny).

- **RAM:** trivial (< 0.5 GB; routers are 32 × 8 × 4096 floats total).
- **CPU:** trivial (32 small mat-vecs).
- **Disk:** reads only the 32 router tensors (~few MB).

**Can C2 run on this notebook?** **Yes, trivially.** → **🟢 VERDE.**

## FASE 4 — C5 (active-path) feasibility — the binding constraint

C5 = Atenia's `MoeRuntime` **full forward of the real model** (disk-tier, low RAM)
vs an **external F64 reference**. Two cost centres: the **reference generation**
(host RAM) and the **Atenia run** (disk + time). The Atenia side is fine; the
**reference generation is the risk.**

### Reference generation — RAM (the problem)

The Qwen methodology instantiates HF's decoder layer (all experts), `.double()`s
it, and `load_state_dict`s the (classic→packed) F64 weights **one layer at a
time**. Decomposing the **measured** Qwen peak (16.4 GB for a 4.15 GB F64 layer):

| Component | Qwen (measured-consistent) | Mixtral (scaled by layer size) |
|---|---|---|
| `sd` packed F64 (the layer's expert weights) | 4.15 GB | **11.3 GB** |
| HF layer F64 (after `.double()`) | 4.15 GB | **11.3 GB** |
| F32 instantiation lingering pre-gc | ~2.0 GB | **~5.6 GB** |
| torch/python base + activations | ~2–4 GB | ~2–4 GB |
| **Peak during `load_state_dict`** | **~16.4 GB (measured)** | **~28–30 GB (estimated)** |

**~28–30 GB peak vs ~20.5 GB free ⇒ the naive Qwen method OOMs / heavy-swaps on
Mixtral.** This is the single decisive finding.

**Mitigations (each brings it back into budget, all ADR-007-compatible):**
1. **`load_state_dict(assign=True)` + build the layer directly in F64** (skip the
   F32 instantiation and the sd-vs-layer double copy): peak ≈ one F64 layer +
   torch ≈ **~13–14 GB** → fits ~20.5 GB. *Stays the strong F64 form.* **Preferred.**
2. **F32 cross-framework reference** (ADR-007's explicit C5 fallback when F64-active
   exceeds RAM): everything halves → sd 5.6 GB + layer 5.6 GB + torch ≈ **~13 GB**
   → fits. *Weaker oracle (F32 vs F32), drift documented, exact argmax required.*
3. **Active-only F64 reference** (load just the **top-2** routed experts/layer in
   F64 ≈ 2.82 GB + attention): peak **~5 GB** — but needs a custom forward (not
   HF's full layer), reintroducing the convention risk the Qwen path avoided
   (mitigable by reusing HF attention + the C1-validated SwiGLU).

### Reference generation — disk / thermal / time
- **Disk:** reads ~all expert weights once (~90 GB); writes a tiny reference
  (4 tokens × 32000 × 4 B ≈ 0.5 MB). NVMe pressure: sequential reads, fine.
- **Thermal:** sustained CPU F64 + I/O for the generation (~10–30 min) on an
  i9-14650HX laptop → expect fan ramp / mild throttling, not a correctness risk.

### Atenia run (disk-tier)
- **RAM:** disk-tier streams top-2 experts per layer (~1.4 GB/layer F32) → steady
  ~**4–6 GB**. Fine.
- **Time:** Qwen was 950 s (2.7 B active, 24 layers, 4 tokens). Mixtral has 4.78×
  active params, 32 layers, 20× bigger experts streamed from disk ⇒ estimate
  **~1500–2800 s (~25–47 min)** for the single 4-token prefill. Slow but bounded.

### Verdict
- **Naive Qwen-identical method: 🔴 ROJO** (~28–30 GB peak → OOM on 32 GB).
- **With mitigation 1 or 2 (assign=True F64, or F32 fallback) + ~20 GB free: 🟡
  AMARILLO** — feasible but tight; needs implementation care + closing other apps.

**Can C5 run safely on this notebook?** **Only with an adapted reference
generator** (assign-mode F64 or the F32 fallback) and **~20 GB free RAM**. As a
straight copy of the Qwen C5 generator it will **OOM**. → **🟡 AMARILLO.**

**Minimum free RAM for C5 reference: ~14 GB** (mitigated path). **Recommended:
~20 GB+** (close apps; enlarge Windows pagefile as a swap safety net). **OOM risk:
HIGH on the naive method, LOW–MEDIUM on the mitigated path. Thrashing risk:
MEDIUM** if free RAM dips below ~14 GB (per-layer F64 won't fit → page churn).

## FASE 5 — Qwen-MoE L3 vs Mixtral L3

| Dimension | Qwen-MoE L3 (measured) | Mixtral L3 (estimated) | Mixtral ÷ Qwen |
|---|---|---|---|
| Expert size (params) | 8.65 M | 176.2 M | **20.4×** |
| Experts / layer | 60 | 8 | 0.13× |
| MoE layer experts (F64) | 4.15 GB | 11.3 GB | **2.72×** |
| Layers | 24 | 32 | 1.33× |
| Total / active params | 14.3 B / 2.7 B | 46.7 B / 12.9 B | 3.27× / **4.78×** |
| C1 experts to check | 1440 | 256 | 0.18× (fewer) |
| C1/C2 RAM | low | low | ~1× |
| **C5 ref peak RAM** | **~16.4 GB (fit)** | **~28–30 GB naive / ~13–14 GB mitigated** | **~1.7× naive** |
| C5 Atenia run time | 950 s | ~1500–2800 s | ~1.6–2.9× |
| Disk read (C1 gen) | ~24 GB | ~90 GB | ~3.75× |
| Download | 27 GB (present) | 93.4 GB | 3.46× |
| Methodology complexity | reused HF layer as-is | **must adapt** C5 ref (RAM) | + |

**How much costlier is Mixtral?** Compute / time / disk: **~2–4×**. Download:
**~3.5×**. The **binding** difference is **C5 reference RAM**: Qwen *fit* the
naive F64 method (~16 GB < 20.5 GB free); **Mixtral does not** (~28–30 GB) and
**requires a methodology adaptation** to fit. That single factor — not raw FLOPs —
is what moves Mixtral L3 from "green" to "yellow".

## FASE 6 — Recommendation

1. **Worth downloading Mixtral now?** **Conditionally yes** — *if* the goal includes
   L1/L2 (clearly feasible, high value, ~1 day) and you accept that **L3 needs a C5
   reference adaptation**. If L3 must be the strong **F64** form with zero extra
   engineering, **not yet** — decide the C5 approach first.
2. **Can this notebook complete L1?** **Yes — 🟢** (C1+C2 are low-RAM, ~1 day +
   download + ~10–25 min run).
3. **Can this notebook complete L2?** **Yes — 🟢** (C4 already in hand; fold-in is
   docs/manifest only).
4. **Can this notebook complete L3?** **🟡 Yes, but only with an adapted C5
   reference generator** (assign-mode F64 or F32 cross-framework) and ~20 GB free.
   A naive copy of the Qwen C5 generator will **OOM (🔴)**.
5. **Conditions before downloading:** (a) decide the C5 methodology
   (assign-mode F64 *preferred*, else F32 fallback, else active-only); (b) ensure
   **~20 GB free RAM** at C5 time (close apps) + enlarge the Windows pagefile as a
   safety net; (c) confirm ~93.4 GB disk (have 813 GB); (d) accept a slow C5 run
   (~25–47 min) + fan ramp.
6. **Biggest real technical risk:** the **C5 reference peak RAM (~28–30 GB naive)
   exceeding ~20.5 GB free → OOM / thrashing.** Everything else (C1/C2 RAM, disk,
   compute, the Atenia disk-tier run) is comfortably in budget. The risk is
   **engineering-mitigable** (assign=True / F32 / active-only), not a hard wall.

## Verdict summary

| Level | Light | Min free RAM | Notes |
|---|---|---|---|
| **L1** | 🟢 VERDE | ~6 GB | C1 one-expert-at-a-time + C2 trivial; ~1 day + download |
| **L2** | 🟢 VERDE | ~6 GB | C4 already certified (`mixtral_scale` 1.490e-7… 1.639e-7); fold-in only |
| **L3** | 🟡 AMARILLO | ~14 GB (mitigated) / ~20 GB recommended | naive F64 C5 = 🔴 OOM; needs assign-mode F64 or F32 fallback |

- **Min RAM:** ~6 GB (L1/L2), **~14 GB** (L3 mitigated).
- **Recommended RAM (L3):** **~20 GB+ free** (close apps + pagefile).
- **Expected time:** L1 ~10–25 min run; L3 C5 reference ~10–30 min + Atenia run
  ~25–47 min.
- **OOM risk:** L1/L2 negligible; L3 **HIGH on the naive method, LOW–MEDIUM
  mitigated**.

*Audit + estimation only — no download, no source/runtime/loader/numerics/test
change, no model executed, no commits.*
