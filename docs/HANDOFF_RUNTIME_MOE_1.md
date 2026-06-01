# HANDOFF — RUNTIME-MOE-1: Mixtral real end-to-end validation — **BLOCKED**

Milestone: **RUNTIME-MOE-1** — validate load + real generation on a **real**
Mixtral checkpoint. **Status: BLOCKED** on an environment limitation (real
weights not present). No code changed; this milestone is parked, documented, and
re-openable the moment real weights are provisioned. Decision taken with the
user (FASE 1 STOP): **defer**.

Predecessors: RUNTIME-REAL-1..4 (dense breadth campaign closed, `3250f97`);
MOE-FULL-15 (`1a6fd0c`, MoE scale certification).

## FASE 1 — Audit (the blocker)

**Real Mixtral weights are not available locally.**

`models/Mixtral-8x7B-v0.1/` is a **config-only stub** (~2.3 MB total):

```
config.json  generation_config.json  special_tokens_map.json
tokenizer.json  tokenizer.model  tokenizer_config.json
```

- **No** `model.safetensors`, **no** `model.safetensors.index.json`, **no**
  shards, **no** GGUF.
- A repo-wide search of `F:` for Mixtral / 8x7b / 8x22b weight files > 100 MB
  returned **zero** results.
- The only "mixtral" weight files in-tree are the **synthetic
  topology-representative fixtures** from the MOE-FULL campaign
  (`fixtures/moe/{full_mixtral, gqa_mixtral, mixtral_classic, mixtral_layer0,
  mixtral_scale}.safetensors`) — **not real weights**. (MOE-FULL-15 already
  documented: "the multi-GB real Mixtral weights are NOT downloaded/certified.")

The stub's `config.json` confirms a genuine Mixtral-8x7B config (8 local experts,
top-2, 32 layers, hidden 4096, GQA 8 kv-heads, vocab 32000, rope_theta 1e6,
sliding_window null) — but with no tensors, **no real load or generation is
possible**.

### Hardware feasibility analysis (had the weights existed)

Host: RTX 4070 Laptop, 8 GB VRAM + ~32 GB RAM + ~830 GB free NVMe.

- Mixtral-8x7B at bf16 ≈ **~94 GB**. Does **not** fit VRAM+RAM (40 GB); would
  require **massive disk-tiering** (~54 GB on NVMe). The engine has the
  `ExpertTier::Disk` residency path (MOE-FULL-8) for this, but generation would
  be **extremely slow** (disk-streamed per expert per token).
- The controlled MoE path `MoeRuntime::load_from_dir` (runtime.rs:249) resolves a
  **single `model.safetensors`**; real Mixtral-8x7B ships **sharded** (~19
  shards + index). Loading real weights would likely **expose a sharded-loader
  gap in the MoE path** — itself a STOP, and a code change that conflicts with
  this milestone's "no new architecture / only validation" scope.

### Execution route that *would* be used

`atenia moe-generate --model <dir> --prompt-ids … --experimental-moe` (or
`ATENIA_ENABLE_MOE=1`) → `controlled_moe_generate` → manifest scope gate
(`certified_scaled`) → `MoeRuntime::load_from_dir` → router + top-2 experts.
This path is present and tested; it simply has **no real weights to run**.

### Risks

1. **Primary (blocking):** weights absent — cannot do real load/generation.
2. Even if downloaded: ~94 GB → disk-tiered → impractically slow; and the MoE
   loader expects single-file safetensors, not sharded (probable gap).
3. Downloading ~94 GB and/or adding a sharded MoE loader is out of scope for a
   "validation only, no optimizations, no new architecture" milestone.

## Decision (FASE 1 STOP)

Per critical rule 6, work stopped at the audit and the user was asked how to
proceed. **Chosen: defer RUNTIME-MOE-1 (BLOCKED).** Do **not** fabricate a
"real" validation from synthetic fixtures; do **not** unilaterally start a 94 GB
download or change the loader. No code touched.

## What IS validated for Mixtral today (not part of this milestone)

For honesty, the Mixtral MoE **runtime, router, top-2 gating, experts, manifest
gate, and CLI** are already certified at **topology scale** (MOE-FULL-15):
`mixtral_scale` fixture (8 experts, top-2, GQA 4:1) end-to-end vs HF f64 at
**1.639e-07**, manifest scope `certified_scaled`. Real-checkpoint **layer-0 MoE
block** certified at **1.164e-10** (`mixtral_layer0`). What is **missing** is
exactly the thing this milestone targets: a run against the **real multi-GB
Mixtral weights**.

## FASE 9 — Strategic review

**Can real Mixtral be considered GREEN?** → **No — BLOCKED.** The MoE
runtime/routing is topology-certified, but no real-weight run exists. Real
Mixtral stays **experimental**.

**Is real MoE starting to be validated?** → Not yet at the real-weight level.
The correctness substrate (routing/experts/gating) is certified on
topology/sub-reference fixtures; the real-weight end-to-end step is unstarted
because the weights are absent.

**What's missing for real Qwen-MoE?** Same blocker class: a real Qwen-MoE
checkpoint (e.g. Qwen1.5-MoE-A2.7B ~stored, or Qwen2-57B-A14B) is **not present
locally** (only `qwen_scale` / `qwen2_moe_layer0` fixtures). Plus likely the
single-file-vs-sharded loader question.

**What's missing for real DeepSeek-MoE?** The hardest: no real DeepSeek-V2
checkpoint locally (only synthetic `deepseek_scale`); DeepSeek-MoE is the weakest
in the cert matrix (synthetic weights only, f32 drift 7.806e-3); MLA + real
weights + sharded loader all unvalidated against real tensors.

## What unblocks RUNTIME-MOE-1

1. **Provision real Mixtral-8x7B weights** on a host that can hold them (or
   accept disk-tiered ~94 GB on the 830 GB NVMe, with very slow generation), AND
2. confirm/extend the MoE controlled path to load **sharded** safetensors (real
   Mixtral is multi-shard) — a scoped loader task, its own milestone.

Until both land, real MoE generation cannot be validated here.

## Files modified

- `docs/HANDOFF_RUNTIME_MOE_1.md` — this file (block record).
- `docs/STATUS.md` — RUNTIME-MOE-1 BLOCKED note.

No production code, tests, architecture, families, math, or graph ops changed.

## Deliverable answers

1. **Loaded correctly?** N/A — no real weights present (BLOCKED).
2. **Coherent text?** N/A.
3. **MoE actually used?** N/A for real weights (topology runtime already
   certified, MOE-FULL-15).
4. **Experts/routing validated?** Only at topology scale (prior work): 8
   experts, top-2, GQA — not on real weights.
5. **Important differences?** N/A.
6. **Problems found?** Real Mixtral weights absent; MoE loader is single-file
   while real Mixtral is sharded (would need a loader task).
7. **Memory used?** N/A (would be ~94 GB → disk-tiered).
8. **Load time?** N/A.
9. **Generation time?** N/A.
10. **Mixtral → GREEN?** No — BLOCKED; stays experimental.
11. **Files modified:** see above.
12. **Commit:** see git log (RUNTIME-MOE: Mixtral real end-to-end validation —
    blocked).
13. **CI:** docs-only commit → skipped by design (`paths-ignore`).
14. **Next recommendation:** see below.

## Next recommendation

Real MoE validation is **environment-gated**, not code-gated. Recommended order:
1. A **scoped loader milestone**: extend the MoE controlled path to load
   **sharded** safetensors (real Mixtral/Qwen-MoE/DeepSeek are all multi-shard).
   This is a real prerequisite and is a code task, not validation.
2. Then **provision real Mixtral-8x7B** weights and re-open RUNTIME-MOE-1 (expect
   slow, disk-tiered generation — correctness-first).
3. The dense breadth campaign (Llama/Qwen2.5/Gemma2/Phi-3, all GREEN) remains the
   solid, shippable real-generation story in the meantime.
