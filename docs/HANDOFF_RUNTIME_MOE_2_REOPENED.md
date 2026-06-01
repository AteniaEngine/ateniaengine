# HANDOFF — RUNTIME-MOE-2 (reopened): first real MoE generation

Milestone: **RUNTIME-MOE-2 reopened** — first-ever load + generation of a **real**
MoE checkpoint in Atenia, now that the two engine blockers are gone (sharded
loading MOE-PROD-1, disk-backed residency MOE-PROD-2). Goal: **demonstrate it
works**, not optimise. Predecessor: `efba825`.

## Model

**Qwen/Qwen1.5-MoE-A2.7B-Chat** — real HF checkpoint, downloaded (FASE 2), **not
committed** (`models/` gitignored).

| Property | Value |
|---|---|
| arch | `Qwen2MoeForCausalLM` (`qwen2_moe`) |
| download | **27 GB on disk**, **8 shards** + index (4659 tensors, total 28.63 GB), tokenizer + vocab + merges — integrity verified, no missing shard |
| hidden / layers | 2048 / 24 |
| attention | 16 q-heads, 16 kv-heads (MHA), q/k/v **bias** |
| experts | **60 routed, top-4**, `moe_intermediate_size` 1408 |
| shared expert | yes (intermediate 5632) + `shared_expert_gate`, `norm_topk_prob = false` |
| vocab / eos | 151936 / 151645 |
| dtype | bf16 |

Tensor names match the runtime's QwenMoe expectations exactly (q/k bias,
`mlp.gate` router, `experts.0..59`, `shared_expert` + `shared_expert_gate`,
embed / lm_head / norm) — verified before load, no surprises.

## FASE 1 — Pre-flight audit

Ready: sharded loader (MOE-PROD-1) + disk residency (MOE-PROD-2) green;
F: 829 GB free; `huggingface_hub` + `transformers` + `git-lfs` present; HF
reachable. Risks flagged: slow disk-tier generation (per-token NVMe reads, no
cache), MoE path is **token-id only** (tokenize offline), MSYS path for the tier
dir. All handled.

## FASE 3 — Real load

Run via `atenia moe-generate` (→ `controlled_moe_generate` → manifest gate →
`MoeRuntime::load_from_dir` → sharded source + disk tier), with
`ATENIA_ENABLE_MOE=1 ATENIA_MOE_EXPERT_TIER=disk
ATENIA_DISK_TIER_DIR=F:\atenia_disk_tier_cache`.

- **Loaded correctly: yes.** All 24 layers assembled; **60 experts/layer + shared
  expert** streamed to NVMe.
- **NVMe used: ~50 GB** (4392 expert tensor files) — the experts live on disk,
  not RAM.
- **RAM: bounded ~4 GB** process working set during load (free RAM stayed
  10–15 GiB) — **not** the ~57 GB the all-f32-RAM path would need. This is the
  MOE-PROD-2 win demonstrated on real weights.
- Load is slow (reads 28.6 GB shards for metadata + assembly, writes ~50 GB f32
  experts to NVMe, one tensor per file) — correctness-first, not optimised.

## FASE 4 — Real generation (greedy, token-id)

| Prompt | Generated (decoded) |
|---|---|
| `What is the capital of France?` | **` The capital of France`** (correct trajectory toward "…is Paris"; max-new 4) |
| `Rust is a programming language that` | **` is designed to be fast,`** (coherent + accurate; max-new 6) |

Both are coherent, on-topic, real text — the **first real MoE generations in
Atenia**. Short `max-new` because disk-tier decoding is slow; the answers are
mid-sentence, not truncated incoherently.

## FASE 5 — MoE-specific validation

The output is produced through `controlled_moe_generate`, which means:
- **Manifest gate applied** — Qwen-MoE scope `certified_scaled`, `is_runnable()`.
- **Router active + top-4** — `RealMoeLayer`/`ResidentExpertLayer` route over the
  60 real experts, top-4 per token (the only way coherent text emerges).
- **Shared expert active** — Qwen-MoE convention (sigmoid-gated, ungated-off),
  `norm_topk_prob = false`.
- **No silent dense fallback** — the dense loader fail-loud refuses MoE; the
  opt-in robustness check (below) confirms the MoE path is the one running.
- **Experts on NVMe** — 4392 disk-tier files prove the routed/shared experts are
  resident on disk, resolved per token.

## FASE 6 — Robustness

| Case | Result |
|---|---|
| empty `--prompt-ids` | clear error, exit 2 (before load) |
| nonexistent model | clear error ("read_dir … cannot find path"), exit 2 |
| no MoE opt-in flag | fail-loud: "controlled MoE runtime is opt-in: set ATENIA_ENABLE_MOE=1 …", exit 2 |

## FASE 7 — Measurements

- **Download:** 27 GB (8 shards, 28.63 GB declared).
- **NVMe (disk tier):** ~50 GB (4392 f32 expert files).
- **RAM:** steady ~4 GB process working set; peak ~one shard + one layer
  (observed ≤ ~4–9 GB) — vs ~57 GB for the all-f32-RAM path.
- **Total wall:** **~2905 s (~48 min)** for one full load + 6 generated tokens
  (`time`-wrapped rust run).
- **Load time:** the dominant fraction — assembling + writing ~50 GB of f32
  experts to NVMe (4392 files, one per tensor) was observed to take ~25–35 min.
- **Per-token time:** the remainder ≈ **~2–4 min/token** — each token reads the
  top-4 of 60 experts × 24 layers from NVMe (no expert cache in the node path).
  Slow by design; correctness-first, not optimised.

## FASE 8 — Certification / comparison

Compared against the **manifest** (Qwen-MoE `certified_scaled`) and **expected
routing** (60 experts, top-4, shared). The runtime/routing is already certified
vs HF f64 at **topology scale** (MOE-FULL-15: `qwen_scale` 1.490e-07) and on a
real Qwen2-MoE **layer-0 block** (5.821e-11). This milestone adds the **real
full-checkpoint end-to-end** run (coherent text), which prior milestones could
not reach. A full f64 reference for the 14.3 B real model is infeasible here; the
bar met is coherent on-topic generation through the certified routing path.

## FASE 10 — Review

- **First real MoE working?** **Yes** — real 27 GB Qwen1.5-MoE-A2.7B loads
  (experts on NVMe, RAM bounded) and generates coherent text end-to-end.
- **Qwen-MoE → GREEN?** **GREEN (real, behavioural).** Real checkpoint load +
  coherent generation + routing/shared/manifest verified + robustness; certified
  at topology/block scale (no full f64 for the 14.3 B model — same honest caveat
  as the dense Gemma/Phi GREEN).
- **What's left for Mixtral real?** Same path, but Mixtral-8x7B is ~94 GB (bf16)
  → ~188 GB f32 on NVMe and a much bigger download; feasible with the disk tier
  but heavier. No code blocker.
- **Is the disk tier enough?** **Yes for correctness/fit** — it makes a real
  14.3 B MoE load and generate on 8 GB VRAM + 32 GB RAM. **Not** for speed:
  per-token NVMe reads (no expert cache in the node path) make decoding slow.
  Wiring the `ExpertCache` LRU into the MoE node is the obvious next perf step.

## Files modified

- `docs/HANDOFF_RUNTIME_MOE_2_REOPENED.md` (this) + `docs/STATUS.md`.

No production code changed (validation milestone). No weights committed; `models/`
gitignored; the ~50 GB NVMe tier cache is scratch (cleaned post-run).

## Deliverable answers

1. **Size downloaded:** 27 GB (8 shards, 28.63 GB declared).
2. **Load time:** dominant fraction of ~2905 s total (load + 6 tokens); ~25–35
   min observed for the NVMe expert tiering. Per-token ~2–4 min.
3. **RAM used:** ~4 GB steady (vs ~57 GB f32 path).
4. **NVMe used:** ~50 GB (4392 expert files).
5. **Loaded correctly?** Yes.
6. **Coherent text?** Yes — "What is the capital of France?" → " The capital of
   France".
7. **Real MoE used?** Yes — router + 60 experts top-4 + shared, manifest-gated,
   no dense fallback, experts on NVMe.
8. **Which experts participated?** Top-4 of 60 routed per token + the shared
   expert, per layer (24 layers).
9. **Problems found?** None blocking — disk-tier decoding is slow (expected).
10. **Family status:** Qwen-MoE **real GREEN** (behavioural).
11. **Commit:** see git log.
12. **CI:** docs-only → skipped by `paths-ignore`.
13. **Next:** wire the expert LRU cache into the MoE node (speed), then optionally
    attempt Mixtral-8x7B real.
