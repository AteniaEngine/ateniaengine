# HANDOFF — MIXTRAL-CERT (C1+C2): Mixtral-8x7B-v0.1 real-weight per-expert + router → toward L1

Milestone: **MIXTRAL-CERT-1 / C1+C2** — certify **C1 (per-expert)** and **C2 (router
parity)** on the **real Mixtral-8x7B-v0.1** weights, reusing the Qwen MOE-CERT-2 /
DeepSeek MLA-1 decomposition tooling. **No runtime/loader/numerics/Adapter-Toolkit
change** — a reference generator + a test-only harness. Predecessor:
MIXTRAL-DATA-PROVISION (87 GB, 19 shards, validated; `MIXTRAL_PROVISIONED.md`).
(Distinct from the older `HANDOFF_MIXTRAL_CERT_1.md`, which was tiny-fixture L0.)

## Result

**C1 + C2 PASS on the real weights, all 32 layers → the substantive L1 work.**

| Obligation | Result |
|---|---|
| **C1 per-expert** | **256 experts** (32 × 8), global worst `max_abs_diff` **1.907e-6** (layer 8, expert 1), gate `< 0.5` → PASS (~2.6e5× inside), **0 failures**, exhaustive |
| **C2 router** | top-2 **set equality** on **all 32 layers**, **0 failures**, **min routing margin 0.011413** (layer 13) |
| **C3 attention** | mechanism level (mixtral_scale topology 1.639e-7 incl. GQA attention) — same standard as Qwen/DeepSeek C3 |
| **C4 / C5** | C4 (mixtral_scale topology 1.639e-7) **available** (the L0 cert), not folded; C5 active-path **pending** |

→ **Mixtral-8x7B-v0.1: partial L1** (C1+C2 real-weight + C3 mechanism). Not the dense
ADR-004 `CERTIFIED`; not L2 (C4 fold) / L3 (C5) / L4.

> **MIXTRAL-CERT-2 update — C4 folded → L2.** C4 was verified (re-ran
> `mixtral_8x7b_topology_certifies` = **1.639e-7** vs HF f64, argmax exact,
> greedy→EOS, deterministic — the `mixtral_scale` reduced-dim Mixtral-8x7B topology:
> 8 experts / top-2 / GQA 4:1 / classic experts / no shared / renorm) and folded into
> the manifest (manifest/docs only, no code). Result: **Mixtral-8x7B-v0.1 —
> MoE-certified L2 (whole model) = L1 + C4** (`ladder_level_whole_model: L2`). Caveat:
> C4 is reduced-dim, **random-weight** topology — the mechanism, not the real
> 8-expert trained-weight assembly; it transfers only with the real-weight C1/C2
> (met). Not dense ADR-004 `CERTIFIED`; C5 (active-path) → L3 and L4 remain pending.

## Tooling (reuse + Mixtral deltas)

The Qwen/DeepSeek decomposition machinery transferred almost verbatim. Mixtral deltas:
router/experts under `block_sparse_moe` (classic `experts.{e}.{w1,w3,w2}` = gate/up/down),
**all 32 layers MoE** (no dense-first), **no shared expert**, **8 experts / top-2**,
renormalised top-k (the top-k *set* is convention-independent).

- **Reference generator** (`fixtures/moe/generate_mixtral_decomposition_reference.py`):
  config-driven port; reads the real bf16 weights → float64, **one expert at a time,
  one layer at a time** (never the full model in F64); a single deterministic probe
  input; emits `mixtral_decomp_ref.{safetensors,json}` (~4.2 MB).
- **Harness** (`tests/moe_mixtral_decomposition_test.rs`, `mixtral_real_c1_c2`,
  `#[ignore]` + env `MIXTRAL_DIR`): per layer opens the shard(s), `RealMoeLayer::assemble`,
  then **C1** per-expert `forward` vs the f64 reference (exhaustive) and **C2**
  `top_k_routing` set equality + routing margin.

## Resumability (operational note — important)

The model lives on an **HDD**, so each ~87 GB pass takes ~1.5 h, and this environment
**reaps long background processes (~60-70 min)**. Both the generator and the harness
were therefore made **resumable** with **atomic per-layer checkpoints**
(`fixtures/moe/.mixtral_decomp_ckpt/` for the generator; `%TEMP%/mixtral_c1c2_ckpt/`
for the harness): a reaped run loses at most the in-flight layer and a re-run skips
done layers. The harness asserts PASS/FAIL only once all 32 layers are checkpointed;
an incomplete run prints `INCOMPLETE n/32` and returns (no false pass). The C1+C2
result above completed across a few resumed windows; the final run printed
`MIXTRAL RESULT: ... C1+C2 PASS across all 32 layers`.

## Regression / scope

**No `src/` change** — the harness only *calls* `RealMoeLayer::assemble` + per-expert
`forward` + `top_k_routing` (the certified MoE primitives). The new integration test
is `#[ignore]` and not compiled by CI (`cargo test --lib` only); ADR-004 gate not
lowered. The checkpoint dirs are git-ignored / scratch (not committed). Weights are
git-ignored (not committed).

## Files

- `fixtures/moe/generate_mixtral_decomposition_reference.py` (new, resumable).
- `fixtures/moe/mixtral_decomp_ref.{safetensors,json}` (new, ~4.2 MB).
- `tests/moe_mixtral_decomposition_test.rs` (new, `#[ignore]`, resumable).
- `docs/numcert/mixtral-8x7b-v0.1.moecert.json` (new, partial-L1 manifest).
- this handoff + STATUS / MODEL_FAMILY_VALIDATION / FAMILY_COVERAGE updates.

## Next (toward L2 / L3)

- **MIXTRAL-CERT C4 fold → L2:** the `mixtral_scale` topology cert (1.639e-7) already
  exists; folding it into the manifest is a small docs/manifest step (mirrors DeepSeek
  MLA-1 C4).
- **MIXTRAL-CERT C5 → L3:** real-weight full forward (GQA + MoE) vs a one-layer-at-a-
  time F64 reference, via the disk expert-tier (RAM-feasible, as on Qwen/DeepSeek);
  HDD-slow → reuse the same resumable pattern.
- L4 (global F64, ~374 GB) reserved/unreachable.
