# HANDOFF — MIXTRAL-CERT-1: Mixtral family certification

Goal: certify the **Mixtral family** with representative checkpoints through
the experimental MoE path. Correctness-first, CPU-only, experimental,
fixture-based. The MOE-2 fail-loud guard is **unchanged** — the productive
loader still refuses MoE checkpoints. No CUDA, CLI, generation, Adapter Toolkit
changes. **No `src/` changes were needed** (both Mixtral naming schemes were
already supported). Predecessor: MOE-0..19 + QWEN-MOE-CERT-1 (`8840d29`).

## Models tested

| Model | Source repo | On-disk expert format | Experts | d_model / d_ff | Size | Committed fixture |
|---|---|---|---|---|---|---|
| mixtral | `hf-internal-testing/tiny-random-MixtralForCausalLM` | **packed** (`mlp.experts.gate_up_proj`) | 4 | 64 / 128 | ~8 MB | ✅ yes (small) |
| mixtral_titanml | `TitanML/tiny-mixtral` | **classic** `block_sparse_moe.experts.{E}.w1/w3/w2` | 8 | 1024 / 3584 | ~940 MB | ❌ no (layer-0 fixture ~352 MB) |

Both are **real** Mixtral checkpoints with actual weights and cover the two
distinct Mixtral on-disk layouts (modern packed vs original
`block_sparse_moe` / `w1/w3/w2`). `TitanML/tiny-mixtral` is "tiny" only in
layer count (2 layers) — it has **full Mixtral hidden dims** (d_model 1024,
d_ff 3584), so it is ~940 MB and its single-layer MoE fixture is ~352 MB.

## Models downloaded

- `mixtral` — already on local disk (MOE-15..18).
- `mixtral_titanml` (`TitanML/tiny-mixtral`, ~940 MB) — newly downloaded. Not
  committed. RAM/disk verified first (≈13.5 GB free / 1.2 TB free); the ~940 MB
  download + ~1.9 GB f32 working set fit comfortably.

Candidate repos `trl-internal-testing/tiny-MixtralForCausalLM-0.1` and
`katuni4ka/tiny-random-mixtral` were probed and **do not exist**
(`RepositoryNotFoundError`); skipped. Mixtral-8x7B (real, ~47 B params →
~187 GB f32) was **not** downloaded: the block-only harness materialises all
experts in f32 (infeasible; documented blocker in `docs/MOE_OVERVIEW.md`).

## Results — smoke (MOE-14 harness, real on-disk checkpoints)

| Model | layers | experts | shared | d_model | forward_pass_ok |
|---|---|---|---|---|---|
| mixtral | 2 | 8 (4/layer) | 0 | 64 | ✅ true |
| mixtral_titanml | 2 | 16 (8/layer) | 0 | 1024 | ✅ true |

`mixtral_titanml`'s smoke reads the **classic** on-disk naming
(`block_sparse_moe.experts.{E}.w1/w3/w2`, d_model 1024) and PASSES — confirming
Atenia detects + binds the original Mixtral tensor layout end-to-end on real
data, not just synthetically.

## Results — numerical (Atenia `forward_auto` vs HuggingFace transformers f64)

| Model | Resolved convention | MaxDiff | Argmax | Source |
|---|---|---|---|---|
| mixtral | Atenia | **1.164e-10** | ✅ | Rust cert test (committed fixture) |
| mixtral_titanml | Atenia | **1.49e-8** | ✅ | local generator run (fixture not committed) |

Both match HF under the **Atenia** convention. Mixtral renormalises the top-k
weights and has no shared expert, so the auto-resolver correctly picks `Atenia`
(no `shared_expert_gate` signal) and the result equals HuggingFace's
`MixtralSparseMoeBlock`. `mixtral` is gated in CI at `max_abs_diff < 0.5`
(achieving ~1e-10). `mixtral_titanml`'s larger residual (1.49e-8, still
argmax-matching and ~7 orders inside the gate) reflects its full-size
d_ff=3584 accumulation; it is recorded as a **local** validation because its
fixture is too large to commit.

> Note: `config.norm_topk_prob` is `false`/absent on these checkpoints, yet the
> HF Mixtral block renormalises unconditionally — so the Atenia (renormalise)
> convention is the correct match, confirmed empirically on both the packed and
> the classic checkpoint.

## Errors found & fixes applied

**None in `src/`.** Both Mixtral naming schemes were already supported:
- packed experts (MOE-15) — `mixtral`.
- classic `block_sparse_moe.experts.{E}.w1/w3/w2` (MOE-2 classifier) —
  `mixtral_titanml`, validated end-to-end for the first time on real data.

The only change was to the **offline fixture generator**
(`fixtures/moe/generate_reference.py`): a Mixtral classic `w1/w3/w2`
extraction branch + the `mixtral_titanml` entry (appended last so existing
fixtures' shared-RNG draws and bytes are unchanged). The generator skips
`mixtral_titanml` when the (uncommitted, large) model is absent.

## GGUF audit

No Mixtral GGUF checkpoint is present locally and Atenia has **no MoE GGUF
support**. GGUF MoE was **not** implemented (out of scope). Gap recorded; a
dedicated **MIXTRAL-GGUF-1** milestone is proposed if GGUF MoE becomes a target
(the GGUF reader would need MoE expert-block decoding + the same
detection/binding path; shared with the proposed QWEN-MOE-GGUF-1).

## Limitations

- No Mixtral-8x7B full model run; no full transformer forward (MoE **block**
  only — no attention / norms / embeddings / KV cache / decode).
- `mixtral_titanml`'s numerical result is **local-only** (fixture ~352 MB, too
  large to commit); only its smoke + classic-naming detection are exercised in
  CI. Reproduce via `fixtures/moe/generate_reference.py`.
- Productive loader still **fails loud** on MoE; nothing wired into
  loader / runtime / Adapter Toolkit / CLI.

## Certification status

**Mixtral family: partially certified (experimental).**

- ✅ MoE-block numerical parity with HuggingFace certified: `mixtral` (packed,
  1.164e-10, committed CI fixture) and `mixtral_titanml` (classic
  `w1/w3/w2`, 8 experts, d_model 1024, 1.49e-8, local-only) — both Atenia
  convention, both argmax-matching.
- ✅ Packed AND classic Mixtral on-disk layouts validated end-to-end on real
  data; auto-resolved to the correct convention.
- ❌ **Not** production-certified: no Mixtral-8x7B full model, no full
  transformer path, no end-to-end generation, no numcert manifest, fail-loud
  still active.

## Tests

- `tests/mixtral_cert_test.rs` — 4 integration tests: `mixtral_certifies`
  (forward_auto vs HF, Atenia convention, 1.164e-10), `mixtral_classic_naming_detected`
  (block_sparse_moe + w1/w3/w2 → gate/up/down), `mixtral_fail_loud_still_active`,
  `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**727 passed / 0 failed / 1 ignored** (unchanged — no `src/` change). Mixtral
cert suite: 4/4. Smoke (real, both Mixtrals): PASS. All prior MoE suites green
(qwen cert 6/6 re-verified).

## Files modified

* `fixtures/moe/generate_reference.py` — Mixtral classic `w1/w3/w2` branch +
  `mixtral_titanml` entry (large model, skipped when absent).
* `tests/mixtral_cert_test.rs` — new (4 cert tests; committed `mixtral` fixture).
* `docs/HANDOFF_MIXTRAL_CERT_1.md` — this file.
* `docs/MODEL_FAMILY_VALIDATION.md`, `docs/MOE_OVERVIEW.md` — cross-notes.

No `src/`, loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm,
Metal, tier-planner, or graph changes. Fail-loud preserved. No model committed.
