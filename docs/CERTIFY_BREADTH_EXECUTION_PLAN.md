# CERTIFY-BREADTH-2 — Execution Plan (viability audit)

**Audit only — no code, no manifests, no commits, no execution.** A
hardware-grounded plan to turn the *infrastructure-ready, evidence-pending*
families (Gemma-2, Gemma-3, Phi-3) into **CERTIFIED** by running the committed
f64 harness. Builds on [HANDOFF_CERTIFY_BREADTH_1.md](./HANDOFF_CERTIFY_BREADTH_1.md),
[FAMILY_COVERAGE_AUDIT.md](./FAMILY_COVERAGE_AUDIT.md), and
[ADR-004](./decisions/ADR-004-f64-reference-as-default.md).

## What "certify" requires (recap)

ADR-004: assert Atenia's **F32** forward vs a **PyTorch F64** reference on the
canonical 4-token sequence `[1,100,200,300]`, `max_abs_diff < 0.5` + per-position
argmax match. Two independent steps, **non-overlapping in RAM**:
1. **F64 generation** (PyTorch, CPU): `generate_f64_reference.py <model_dir>
   <out_dir>` → loads `model.double()`, one forward, writes a small
   `expected_logits_f64.json` (~vocab×4×8 B ≈ 8 MB), then exits. **Peak RAM = the
   f64 model footprint.**
2. **Atenia comparison** (CPU): the `#[ignore]` harness loads the model in F32,
   runs one forward, diffs vs the fixture. **Peak RAM = Atenia's F32 load.**

Peak overall = `max(step1, step2)` — step 1 (f64) dominates.

## FASE 2 — What exists vs what's missing

**Already committed (infrastructure — done in CERTIFY-BREADTH-1):**
- `tests/fixtures/generate_f64_reference.py` — reusable f64 generator (RAM guard).
- `tests/certify_breadth_f64_validation_test.rs` — CPU-vs-F64 harness, 3 `#[ignore]`
  per-model tests (`gemma2_2b_…`, `gemma3_1b_…`, `phi35_mini_…`) + 6 CI units.
- `docs/numcert/{gemma-2-2b-instruct,gemma-3-1b-it,phi-3.5-mini-instruct}.numcert.json`
  — wired, with an `f64_certification_status` reproduction recipe; **drift = null**.
- Local checkpoints: `gemma-2-2b-it`, `gemma-3-1b-it`, `phi-3.5-mini-instruct`.
- `torch 2.5.1+cpu` + `transformers 5.6.2`.

**Missing (the only deliverables of CB-2):**
- The three `expected_logits_f64.json` fixtures (not yet generated).
- The measured `max_abs_diff_vs_f64` + argmax in each manifest (still null).
- (Optional, separate) the GPU-TF32 `certified_mode` number — needs a CUDA torch
  + the GPU path; **out of CB-2's CPU scope**.

## FASE 3 — Per-target feasibility (hardware-grounded)

Measured host state: **RAM 31.7 GB total / ~12 GB free** (idle headroom can be
raised by closing build/browser processes); **GPU RTX 4070 8 GB**; **torch is
CPU-only**. f64 peak ≈ `params × 8 B × ~1.15`.

| Target | On disk | f64 gen RAM (peak) | Atenia F32 load RAM | Feasible now? | Time (gen+cmp, est.) | Tech risk | P(success) |
|---|---|---|---|---|---|---|---|
| **Gemma-3-1B** (single) | 1907 MB | **~8 GB** | ~4 GB | ✅ **yes** (fits ~12 GB free) | ~3–10 min | medium (dual-RoPE + QK-norm + SWA) | **~75%** |
| **Gemma-2-2B** (sharded) | 4986 MB | **~21 GB** | ~11 GB | ⚠️ only after **freeing RAM** (close apps → ~24–28 GB free of 31.7) | ~8–20 min | medium-high (27× SoftCap, scaled embed, dual-norm) | **~65%** |
| **Phi-3.5-mini** (sharded) | 7288 MB | **~30 GB** | ~16 GB | ❌ **high OOM risk** (≈ 31.7 GB ceiling) | ~15–40 min if it runs | high (RAM + LongRoPE) | **~35%** here / high on a bigger box |

Notes:
- The **f64 gen step is the binding constraint**. Gemma-3-1B clears it on the
  current free RAM; Gemma-2-2B needs the box quiesced; Phi-3.5 is at/over the
  physical ceiling — its f64 pass would thrash swap or be refused by the script's
  RAM guard.
- **Numerical risk is real and is the point of the gate.** Gemma/Phi have more
  exotic ops (soft-caps, scaled embeddings, LongRoPE, dual-RoPE) than the
  Llama/Qwen fixtures (which land at 0.01–0.27 drift). A first run could exceed
  0.5 → that is a **genuine finding** (a real numeric discrepancy or a tolerance
  question), not a non-event. Behavioural GREEN (RUNTIME-REAL-3/4: coherent text,
  certified↔fast bit-identical) makes a *pass* likely but does not guarantee
  `< 0.5` on every logit.

## FASE 4 — Execution plan

### Recommended order
1. **Gemma-3-1B** — cheapest, fits current RAM, validates the gemma3 builder
   numerically. *The quick win.*
2. **Gemma-2-2B** — after freeing RAM (quiesce the box). Validates the gemma2
   builder (the soft-cap path).
3. **Phi-3.5-mini** — **deferred** to a host with ≥ ~36 GB RAM (or a swap-tolerant
   run accepting slowness). Do **not** force it on the 32 GB box.

### Quick wins
- **Gemma-3-1B today** is the single highest certified-coverage-per-effort move:
  one f64 gen (~8 GB, minutes) + one `cargo test --ignored` run, no new code.

### Blockers
- **RAM** for Gemma-2-2B (needs freeing) and Phi-3.5 (needs a bigger box).
- `psutil` may be absent → the generator's RAM guard silently skips (verify free
  RAM manually before Gemma-2/Phi).
- The **GPU-TF32 `certified_mode`** number is *not* obtainable here (CPU-only
  torch); CB-2 fills the **CPU-F32 / ADR-004 primary** metric only — document the
  distinction, do not conflate.

### Per-model procedure (already committed in each manifest)
```
python tests/fixtures/generate_f64_reference.py <model_dir> tests/fixtures/<ref_dir>
<ENV>_DIR=<model_dir> cargo test --test certify_breadth_f64_validation_test \
    --release -- --ignored <test_name> --nocapture
# then transcribe printed max_abs_diff + argmax into the manifest
```
(`<ref_dir>`/`<test_name>`/`<ENV>`: `gemma3_1b_reference` / `gemma3_1b_atenia_f32_matches_f64`
/ `GEMMA3_1B_DIR`; analogous for gemma2_2b and phi35_mini.)

### Success criteria
- `expected_logits_f64.json` generated, committed under `tests/fixtures/<ref>/`.
- Harness prints `max_abs_diff < 0.5` **and** argmax match → manifest
  `certified_mode.max_abs_diff_vs_f64` + `adr_004_strict_pass: true` filled with
  the **measured** number (never fabricated).
- Family moves VALIDATED → CERTIFIED (CPU-F32 / ADR-004 primary).

### Rollback / failure criteria
- If `max_abs_diff ≥ 0.5` or an argmax flips: **do not certify**. Record the
  measured drift + the divergent position in the manifest's status as a *finding*,
  open a numeric-investigation task (which op diverges: soft-cap? LongRoPE?
  scaled embed?), and STOP — never lower the ADR-004 threshold to force a pass
  (ADR-001/003).
- If the f64 gen OOMs / swaps unusably: abort that target, leave its manifest
  null, document "needs ≥ X GB RAM" — no partial/fabricated evidence.

### Estimated effort
- Gemma-3-1B: ~0.5–1 h (gen + run + transcribe + verify).
- Gemma-2-2B: ~1–2 h incl. quiescing the box.
- Phi-3.5: deferred (needs hardware); ~1 h once on an adequate host.
- Total for the two now-feasible families: **~2–3 h**, no new code, one small
  commit (fixtures + manifest numbers).

## FASE 5 — Answers

1. **Is CERTIFY-BREADTH-2 really the next best move?** **Yes — for Gemma-3-1B
   (now) and Gemma-2-2B (after freeing RAM).** It raises *trustworthy* coverage
   using infrastructure that already exists, with no new code/families/CUDA, and
   directly attacks the audit's #1 gap (functional ≫ certified). Phi-3.5 is the
   exception: it is **RAM-blocked on this host** and should wait for a bigger box.
2. **Which family first?** **Gemma-3-1B** — lowest RAM (fits the current ~12 GB
   free), smallest, fastest, and it numerically validates the most-novel builder
   (dual-RoPE + QK-norm). Highest coverage-per-effort.
3. **Any high-risk certification?** **Yes — two kinds.** (a) **Phi-3.5: execution
   risk** — ~30 GB f64 vs a 31.7 GB ceiling → likely OOM/swap here (P≈35%).
   (b) **Gemma-2-2B: numerical risk** — 27 SoftCap nodes + scaled embeddings +
   dual-norm are the most likely to surface a > 0.5 drift vs f64; a failing run is
   a real finding, not a non-event. Gemma-3-1B carries moderate numerical risk
   (dual-RoPE) but is cheap to test.
4. **Max certified coverage for least effort?** **Gemma-3-1B alone** — one f64 gen
   (~8 GB, minutes) + one test run certifies the Gemma-3 builder. If RAM is then
   freed, **Gemma-2-2B** adds a second family (~2 h total). That pair —
   **Gemma-2 + Gemma-3 certified for ~2–3 h of execution and zero new code** — is
   the best effort-to-coverage ratio. Phi-3 is the expensive third and should not
   gate the first two.

---

## Executive summary

The certification infrastructure is **complete and committed**; CB-2 is pure
*execution + evidence*. Hardware reality decides the order: **Gemma-3-1B is a
clean quick win on the current box; Gemma-2-2B is feasible after quiescing RAM;
Phi-3.5 is RAM-blocked here and must wait for a ≥36 GB host.** The only honest
risks are (i) Phi-3.5 OOM, and (ii) a genuine > 0.5 drift on Gemma's exotic ops —
which, if it happens, is a *finding to investigate*, never a number to fudge. The
GPU-TF32 `certified_mode` field is a separate, optional follow-up (needs a CUDA
torch); CB-2 fills the ADR-004 **primary CPU-F32 metric**, which is what the
Llama/Qwen fixtures themselves use.

**Recommendation:** proceed with **Gemma-3-1B now**, then **Gemma-2-2B after
freeing RAM**; **defer Phi-3.5** to adequate hardware. Expected result: **+2
certified families (Gemma-2, Gemma-3) in ~2–3 h, no new code.**

*Audit only — no source/manifests changed, no commits, no execution. Sources:
`docs/FAMILY_COVERAGE_AUDIT.md`, `docs/HANDOFF_CERTIFY_BREADTH_1.md`,
`docs/CERTIFICATION.md`, `docs/decisions/ADR-004-*.md`, the three target
`docs/numcert/*.json`, and live host probes (RAM/GPU/torch).*
