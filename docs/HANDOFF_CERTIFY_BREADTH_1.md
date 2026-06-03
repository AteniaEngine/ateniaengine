# HANDOFF — CERTIFY-BREADTH-1: certification infrastructure for Phi-3 / Gemma

**Goal:** raise *certified* coverage toward the functional coverage, without new
families, without touching Numeric Policy / CUDA / MoE / Model Intake, reusing
the existing certification infrastructure — and **without fabricating a single
number**.

## What CERTIFY-BREADTH-1 found (FASE 1 audit)

The gap between *functionally validated* and *numerically certified* for Phi-3 /
Gemma is exactly **one missing artefact per model: a PyTorch-F64 ground-truth
reference** (ADR-004). Concretely, `docs/numcert/{gemma-2-2b,phi-3.5-mini}.numcert.json`
already exist but carry `drift_envelope.certified_mode.max_abs_diff_vs_f64: null`
with the note *"the PyTorch F64 generation pipeline is not yet wired for this
family."* Everything else (adapters, builders, functional validation) is done.

The certification mechanics (CERTIFICATION.md + ADR-004): assert Atenia's **F32**
forward against a **PyTorch-F64** reference on the canonical 4-token sequence
`[1,100,200,300]`, `max_abs_diff < 0.5` + per-position argmax match. The existing
Llama-family CPU test (`llama_3_2_numerical_validation_test.rs`) is the template;
the F64 fixtures are produced offline by per-model `generate_f64.py` scripts.

**Why no drift number is filled in this milestone (the blocker):** generating
the F64 reference needs a PyTorch `model.double()` pass on the real checkpoint —
Gemma-2-2B ~21 GiB peak, Phi-3.5 ~30 GiB peak. Available free RAM at build time
was **12.7 GiB**. So an honest certification of the two named models could not be
run here now. Per the chosen path (**option 2**), this milestone delivers the
**reusable infrastructure, wired and documented, with drift left null** — no
fabricated data, no ADR-004 relaxation, no redefinition of "certified".

## What was built (the deliverable)

1. **Reusable F64 generator** — `tests/fixtures/generate_f64_reference.py`.
   Generalises the per-model `generate_f64.py` scripts into one parametrised
   tool (`<model_dir> <output_dir>`), same `expected_logits_f64.json` output
   format, with a RAM pre-flight guard (refuses to swap silently; `--force` to
   override). Works for any HF causal-LM family transformers can load.

2. **Family-agnostic CPU-vs-F64 harness** — `tests/certify_breadth_f64_validation_test.rs`.
   Drives the forward through the **adapter layer** (so Gemma 2 SoftCap/dual-norm,
   Phi-3 fused-QKV/LongRope, Gemma 3 dual-RoPE all run via their registered
   builders), pure **CPU F32** (no CUDA), single-file or sharded safetensors,
   then compares to the F64 fixture (`max_abs_diff < 0.5` + argmax). Three
   per-model validations (`gemma2_2b` / `gemma3_1b` / `phi35_mini`), `#[ignore]`
   until a fixture exists, panicking with an actionable "generate it with …"
   message. **6 pure-logic unit tests run in CI** (drift, argmax-match, fixture
   parse, missing-fixture error, arch-string read) — real coverage of the
   harness without needing a model.

3. **Wired manifests (drift still null)** —
   `docs/numcert/{gemma-2-2b-instruct,phi-3.5-mini-instruct}.numcert.json` now
   point `f64_fixture_test` at the new harness and carry a `f64_certification_status`
   block with the **exact 3-step reproduction recipe** (generate → run → transcribe).
   New `docs/numcert/gemma-3-1b-it.numcert.json` wires the Gemma-3 slot (drift
   null). No measured number is recorded anywhere.

## What is still missing to COMPLETE the real certification (FASE 6 gap)

Per model, in priority order (smallest first):

| Model | Needs | RAM for F64 pass | Notes |
|---|---|---|---|
| **Gemma-3-1B** | run generator + harness | ~8 GiB (fits 16 GiB) | **recommended first** — only one that fits a typical box now |
| **Gemma-2-2B** | run generator + harness | ~21 GiB free | needs a ≥24 GiB-free box |
| **Phi-3.5-mini** | run generator + harness | ~30 GiB free | near a 32 GiB ceiling — higher-RAM box recommended |

Exact recipe (committed in each manifest's `f64_certification_status`):
```
python tests/fixtures/generate_f64_reference.py <model_dir> tests/fixtures/<ref_dir>
<ENV>_DIR=<model_dir> cargo test --test certify_breadth_f64_validation_test \
    --release -- --ignored <test_name> --nocapture
# transcribe printed max_abs_diff + argmax into the manifest certified_mode
```
Then the harness flips from `#[ignore]`-pending to a real green certification,
and the manifest's `certified_mode` CPU-F32 drift is filled with measured data.

**Two distinct things to be clear about:** the harness certifies the engine's
**CPU-F32** forward vs F64 (the ADR-004 *primary* metric — exactly what the
Llama-family M4.6 tests do). The manifest's `certified_mode` field *also*
describes the **GPU-TF32** kernel path; filling that specific number additionally
needs a CUDA run (out of this milestone's scope, rule 3). They are complementary;
neither is weakened.

## Deliverable answers

1. **Families certified now:** still Llama + Qwen2 (unchanged). CERTIFY-BREADTH-1
   does **not** add a certified family — it delivers the infrastructure to do so
   honestly and leaves the evidence pending (by the chosen option-2 scope).
2. **Evidence obtained:** the harness + generator compile and the 6 CI unit
   tests pass; the 3 model validations are wired and correctly `#[ignore]`.
   No drift evidence (it requires the gated F64 run).
3. **Certificates generated/updated:** Gemma-2 + Phi-3.5 manifests re-pointed at
   the harness with a reproduction recipe; new Gemma-3-1B manifest. All
   `max_abs_diff_vs_f64` remain **null**.
4. **Pending:** the actual F64 fixtures + drift numbers for all three models
   (blocked on RAM here; recipe committed).
5. **Risks:** (a) a future run could reveal a real drift > 0.5 on a non-Llama
   builder — that is the *point* of the gate and would be a true finding, not a
   regression; (b) transformers-version drift in the generator (pinned by
   `local_files_only` + the committed script); (c) Gemma-3 long-context (sliding
   window) is out of the 4-token fixture's reach (documented).
6. **Limitations:** no number is claimed; CPU-F32 path certified by the harness,
   GPU-TF32 `certified_mode` number still needs CUDA; Phi-3.5 F64 is RAM-heavy.
7. **Current certified coverage:** Llama, Qwen2 (full f64). Phi-3, Gemma-2,
   Gemma-3: *functionally validated, certification-infrastructure-ready,
   evidence-pending.*
8-9. Commit / CI: single commit; CI green (new harness compiles cpu-only; its 6
   unit tests run; the 3 model tests are `#[ignore]`).
10. **Next step:** on a box with ≥24 GiB free RAM (and, for the GPU-TF32 field, a
   CUDA GPU), run the committed recipe **Gemma-3-1B → Gemma-2-2B → Phi-3.5** to
   fill the drift and flip the harness green. That is a pure *execution* step on
   adequate hardware — no further code needed.

## Exactly what is needed to finish (RAM / GPU / fixture / execution)

- **RAM:** ≥ ~8 GiB free (Gemma-3-1B), ~21 GiB (Gemma-2-2B), ~30 GiB (Phi-3.5)
  for the PyTorch F64 generation pass. (Build-time free RAM here was 12.7 GiB.)
- **GPU:** not needed for the CPU-F32 ADR-004 primary metric; **only** needed to
  additionally fill the manifest's GPU-TF32 `certified_mode` number.
- **Fixture:** run `generate_f64_reference.py` per model (one-time, committed).
- **Execution:** run the `#[ignore]` harness test per model; transcribe.

## Files

- `tests/fixtures/generate_f64_reference.py` (new) — reusable F64 generator.
- `tests/certify_breadth_f64_validation_test.rs` (new) — CPU-vs-F64 harness +
  6 CI unit tests + 3 `#[ignore]` model validations.
- `docs/numcert/gemma-2-2b-instruct.numcert.json`,
  `docs/numcert/phi-3.5-mini-instruct.numcert.json` — re-pointed + status block.
- `docs/numcert/gemma-3-1b-it.numcert.json` (new) — wired slot, null drift.
- `docs/HANDOFF_CERTIFY_BREADTH_1.md` (this) + `docs/STATUS.md` +
  `docs/MODEL_FAMILY_VALIDATION.md`.
