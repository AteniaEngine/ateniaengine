# Mixtral Certification Roadmap — Audit (MIXTRAL-CERT-2-AUDIT, design-only)

**Audit only — no code, no commits, no execution, no CI.** Estimates how much of
the Qwen-MoE ADR-007 certification machinery (MOE-CERT-2 / 2-ext / 3 / 4, which
took Qwen-MoE to whole-model **L3**) can be reused to certify **Mixtral-8x7B**,
and what remains. Sources (FASE 1, all read): `docs/decisions/ADR-007-moe-
certification-ladder.md`, `docs/MOE_CERTIFICATION_AUDIT.md`,
`docs/numcert/qwen1.5-moe-a2.7b.moecert.json`, `docs/HANDOFF_MIXTRAL_CERT_1.md`,
`docs/HANDOFF_MOE_FULL_15.md`, `fixtures/moe/moe_cert_manifest.json`, the
MOE-CERT-2/2-ext/4 harnesses + generators, and the local model tree.

> **Headline.** The certification **tooling is ~80–85 % reusable** (mostly tensor-
> name / topology adaptation), and **C4 (L0) is already in hand** for Mixtral
> (`mixtral_scale` 1.639e-7) with C3's attention mechanism also already certified.
> But the **real-weight evidence that L1/L2/L3 require (C1, C2, C5) is 0 %**,
> because the **real trained Mixtral-8x7B weights are NOT provisioned** — the
> local `models/Mixtral-8x7B-v0.1/` has only `config.json` + tokenizer, **no
> `*.safetensors`**. Provisioning the ~94 GB checkpoint is the dominant blocker;
> everything that needs real weights is gated on it.

## FASE 1 — State of Mixtral in-repo (evidence)

- **Real weights: ABSENT.** `models/Mixtral-8x7B-v0.1/` = `config.json`,
  `generation_config.json`, tokenizer only. No weight shards, no index. (Disk
  free: 758 GB — a ~94 GB bf16 download fits.)
- **Config (real Mixtral-8x7B):** `MixtralForCausalLM`, hidden 4096, intermediate
  (expert d_ff) 14336, 32 layers, 32 heads / **8 kv (GQA 4:1)**, **8 experts /
  top-2**, **no shared expert**, rope_theta 1e6, vocab 32000, no sliding window.
- **C4 already done (L0):** `mixtral_scale` (8 experts, top-2, GQA 4:1) end-to-end
  vs HF f64 = **1.639e-7** (MOE-FULL-15) — the designated ADR-007 C4 evidence.
- **C3 mechanism already done:** Mixtral attention (GQA, Mistral decoder) certified
  end-to-end at 7.451e-08 / 5.960e-08 (MOE-FULL-13) + the scale GQA path.
- **Runtime supports Mixtral:** `MoeRuntime` runs Mixtral (graph path, classic +
  packed layouts, GQA, disk-tier residency) — certified on fixtures, never on the
  94 GB real weights (no weights).
- **Real-format fixtures exist** (`full_mixtral`, `mixtral_classic`, `gqa_mixtral`,
  `mixtral_layer0`, `mixtral_scale`) — **random-weight**, not the trained model.

## FASE 2 — Reuse table per obligation (C1–C5)

Legend: ✅ reuse as-is · 🔧 reuse with adaptation · 🆕 new work. "+ weights" = also
gated on provisioning the real Mixtral-8x7B checkpoint.

| Obligation | Verdict | What transfers | What changes |
|---|---|---|---|
| **C1** per-expert | 🔧 **+ weights** | the whole pattern: numpy-f64 per-expert SwiGLU generator + Atenia `MoeDenseExpert::forward` vs reference, exhaustive over all experts/layers; Atenia already binds Mixtral classic experts (MIXTRAL-CERT-1) | tensor names `block_sparse_moe.experts.{e}.{w1,w3,w2}` (w1=gate, w3=up, w2=down) vs Qwen `mlp.experts.{e}.{gate,up,down}_proj`; **no shared expert**; 8 experts × 32 layers = **256** (vs 1440); router prefix |
| **C2** router | 🔧 **+ weights** | top-k **set-equality** hard gate + routing-margin harness (identical logic; selection is convention-independent) | **top-2** (k=2 vs 4); router name `block_sparse_moe.gate.weight`; combine **renormalizes** (Atenia convention) — affects weights, **not** the selected set |
| **C3** attention | ✅ **as-is** | the *same* move Qwen used: reuse the existing **Mixtral attention-mechanism cert** (GQA 4:1, Mistral decoder, no bias) — MOE-FULL-13 + `mixtral_scale` | none (mechanism evidence already exists); a real-weight per-layer attention re-cert stays a documented caveat, exactly as for Qwen |
| **C4** assembly/topology | ✅ **as-is** | `mixtral_scale` 1.639e-7 is **already the designated C4**; Mixtral **L0 is already satisfied** | none — fold-in only (the MOE-CERT-3 move) |
| **C5** active-path | 🔧🆕 **+ weights** | the methodology: HF decoder-layer **one-layer-at-a-time f64** reference + tiny-fixture driver validation + Atenia `MoeRuntime` disk-tier real forward + the classic→packed expert conversion idea | a **new Mixtral reference module** (`MixtralDecoderLayer` instead of `Qwen2MoeDecoderLayer`: no shared expert, **no qkv bias**, GQA, `w1/w3/w2`→packed `gate_up_proj/down_proj`); RAM per layer is **much larger** (see risks) |

**Net:** C3 + C4 ✅ as-is; C1 + C2 🔧 light adaptation (name/topology); C5 🔧🆕 the
one genuinely new piece (a Mixtral reference driver) — **all of C1/C2/C5 also
gated on the real weights**.

## FASE 3 — Technical differences: Qwen-MoE (done) vs Mixtral-8x7B

| Aspect | Qwen1.5-MoE-A2.7B (certified L3) | Mixtral-8x7B (target) | Cert impact |
|---|---|---|---|
| Routed experts | 60 | **8** | fewer per-expert C1 runs (256 vs 1440) — cheaper |
| Expert d_ff | 1408 | **14336** (10×) | each expert/layer F64 is ~10× bigger → **C5 RAM/time ↑** |
| top-k | 4 | **2** | C2 k=2; cheaper |
| Shared expert | yes + **sigmoid gate** | **none** | **simpler** (drop shared path in C1/C5 ref) |
| Renorm top-k | false (`norm_topk_prob`) | **true** (renormalizes) | C5/C4 convention = **Atenia** (already supported); C2 selection unaffected |
| Router tensor | `mlp.gate.weight` | `block_sparse_moe.gate.weight` | name map (Atenia already detects both) |
| Expert tensors | `mlp.experts.{e}.{gate,up,down}_proj` | `block_sparse_moe.experts.{e}.{w1,w3,w2}` | name map (Atenia binds both; MIXTRAL-CERT-1) |
| Attention | MHA 16/16 + **QKV bias** | **GQA 32/8**, **no bias** | C3 mechanism already certified for both |
| Decoder | Qwen2 | Mistral | HF module swap in the C5 ref |
| Layers / hidden | 24 / 2048 | **32 / 4096** | more layers, bigger hidden → C5 time ↑ |
| Total / active params | 14.3B / 2.7B | **46.7B / 12.9B** | bigger download + bigger active path |
| On-disk weights | **present (27 GB)** | **ABSENT (need ~94 GB)** | **the blocker** |
| Loader | sharded classic, disk-tier ✓ | sharded classic, disk-tier ✓ (runtime supports) | reuse as-is |

The architecture is, if anything, **simpler** than Qwen (no shared expert, no qkv
bias, fewer experts); the only structural step-ups are **scale** (10× expert d_ff,
46.7B total) and the **missing weights**.

## FASE 4 — Answers

**1. What % of the work is already done for Mixtral?**
- **Certification tooling reuse: ~80–85 %** (the Qwen harnesses + generators
  transfer with name/topology adaptation).
- **C4 (L0): 100 % done** (`mixtral_scale` 1.639e-7). **C3 mechanism: done.**
- **Real-weight evidence for L1/L2/L3 (C1, C2, C5): ~0 %** — blocked on the weights.
- **Weighted toward L3: ≈ 35–45 % done** (infra + C4 + C3-mechanism in hand; the
  real-weight obligations and the C5 reference module remain).

**2. The main blocker.**
**The real trained Mixtral-8x7B weights are not provisioned** (~94 GB bf16
download; only config + tokenizer are local). C1, C2 and C5 all require them — so
without the download, Mixtral is stuck at **L0** regardless of tooling. (The
nearest *code* item is the C5 Mixtral reference driver; but the *gating* item is
the download.)

**3. Estimated cost to L1.**
- Prereq: **download ~94 GB** (hours, I/O-bound) — the dominant real cost.
- Code: adapt the C1 generator + C2/C1 harness to Mixtral names / no-shared /
  k=2 (mirror MOE-CERT-2 + 2-ext) — **S–M, ~1 day**, high confidence.
- Run: C1 over 256 experts (fast) + C2 over 32 routers (fast).
- **Total: ~1 day code + the download + a short run.** (C3 reused.)

**4. Estimated cost to L2.**
- L1 **+ fold C4** (already done) — a manifest/docs update, the MOE-CERT-3 move.
- **Marginal cost over L1 ≈ near zero** (hours; C4 evidence already in hand).

**5. Estimated cost to L3.**
- L2 **+ C5**: a **new Mixtral reference driver** (HF `MixtralDecoderLayer`,
  one-layer-at-a-time f64, `w1/w3/w2`→packed conversion, no shared, no qkv bias),
  validated against a tiny Mixtral HF fixture, then the Atenia `MoeRuntime`
  disk-tier real forward + compare — **M, ~1–2 days code** + a **long slow run**
  (bigger experts ⇒ likely > Qwen's 950 s; expect tens of minutes), with a
  **RAM-tight** reference (see risks).
- **Total to L3 from L0: ~2–4 days of engineering + the ~94 GB download + slow
  runs.** L4 stays reserved/unreachable (global F64 ~373 GB).

## FASE 5 — Deliverable

### Gaps
1. **Real Mixtral-8x7B weights absent** (the blocker; ~94 GB).
2. **No Mixtral C1/C2 generator+harness** (adaptation of the Qwen ones).
3. **No Mixtral C5 reference driver** (the one genuinely new module).
4. (Not a gap) C3 mechanism + **C4/L0 already certified**; runtime already supports
   Mixtral.

### Reuse
- **As-is:** C3 (attention-mechanism cert), C4 (`mixtral_scale` 1.639e-7 = L0),
  the `MoeRuntime` Mixtral load/forward path, the ADR-007 ladder + reporting
  discipline, the manifest `schema_variant: "moe-decomposition"`.
- **With adaptation:** C1 + C2 generators/harness (names, no-shared, k=2), the C5
  methodology (HF-layer-at-a-time + tiny-fixture validation + classic→packed).
- **Reuse estimate: ~80–85 % of the tooling.**

### Risks
- **Weight provisioning (HIGH).** ~94 GB download is the gating cost; everything
  real-weight waits on it. Verify license/availability before committing.
- **C5 reference RAM (MEDIUM-HIGH).** Mixtral has 8 **large** experts (d_ff 14336):
  one layer in F64 ≈ 8×3×14336×4096×8 B ≈ **~11 GB**, with a transient ~17 GB peak
  during f32→f64 — on a 34 GB host (~20 GB free) this is **tighter than Qwen**
  (~4.5 GB/layer). Feasible but needs aggressive per-layer free / building the
  layer directly in f64; a real risk of swap/OOM if not careful.
- **C5 runtime time (MEDIUM).** Disk-tier forward of 46.7 B (32 layers × 8 big
  experts) is slower than Qwen's 950 s; expect a long single run.
- **Convention correctness (LOW).** Mixtral renormalizes top-k and has no shared
  expert; Atenia's **Atenia convention** already matches HF here (MIXTRAL-CERT-1,
  empirically). Low risk, but the C5 driver must use `MixtralDecoderLayer` (not
  Qwen) so the combine/renorm is HF-exact.
- **Coverage (LOW, inherited).** C1/C2 single probe input; C5 single canonical
  input — same documented bounds as Qwen.

### Roadmap (if/when Mixtral is prioritised)
1. **MIXTRAL-CERT-2** — provision the real Mixtral-8x7B weights; adapt the C1/C2
   generator + harness (names / no-shared / k=2); run → **L1**. *Download + S–M.*
2. **MIXTRAL-CERT-3** — fold C4 (`mixtral_scale`, already in hand) → **L2**.
   *Manifest/docs only.*
3. **MIXTRAL-CERT-4** — new Mixtral C5 reference driver (HF `MixtralDecoderLayer`
   one-layer-at-a-time f64 + classic→packed), validate on a tiny Mixtral fixture,
   run Atenia disk-tier real forward + compare → **L3**. *M + slow run; RAM-tight.*
4. **L4** stays reserved/unreachable (global F64 ~373 GB).

## Executive summary

Mixtral certification can **lean heavily on the Qwen-MoE work**: ~**80–85 % of the
tooling is reusable** (C1/C2 need only tensor-name + topology adaptation; the C5
methodology transfers with a new Mixtral reference module), and **C4 (L0) plus the
C3 attention mechanism are already certified** (`mixtral_scale` 1.639e-7,
MOE-FULL-13/15). Mixtral is structurally **simpler** than Qwen (no shared expert,
no qkv bias, 8 experts/top-2). **The one decisive blocker is data, not code: the
real trained Mixtral-8x7B weights are not in the repo** (only config + tokenizer);
C1, C2 and C5 all require the ~94 GB checkpoint, so until it is provisioned Mixtral
stays at **L0**. Once the weights land, **L1 is ~1 day**, **L2 is near-free** (C4
in hand), and **L3 is ~2–4 days total** plus a slow, RAM-tight C5 run. The main
*engineering* risk is the C5 reference's per-layer F64 memory (Mixtral's 8 large
experts ≈ 11 GB/layer, tight on this host).

- **Reuse: ~80–85 % of tooling; C4/L0 + C3-mechanism already done.**
- **Blocker: real Mixtral-8x7B weights absent (~94 GB).**
- **Cost: L1 ~1 day + download; L2 ~free; L3 ~2–4 days + slow run.**
- **Recommendation:** treat MIXTRAL-CERT as a **3-milestone series mirroring
  MOE-CERT-2/3/4**, but **gate it on a deliberate decision to download + store the
  94 GB checkpoint** and to accept the RAM-tight C5 run. If that data cost is not
  worth it now, **record Mixtral honestly at L0** (C4 done) and keep the L1–L3
  tooling adaptation as a ready, low-risk follow-up. Do **not** certify Mixtral
  above L0 on random-weight fixtures — that would be C4-grade evidence mislabelled.

*Audit only — no source/manifests changed, no commits, no execution, no CI.*
