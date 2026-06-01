# HANDOFF — RUNTIME-MOE-2: Small MoE real model validation — **BLOCKED**

Milestone: **RUNTIME-MOE-2** — find, provision, and validate a **small real MoE**
checkpoint to move from topology-fixture certification to a real-weight MoE
load+generate. **Status: BLOCKED.** No small real MoE of a *supported* family is
loadable by the current MoE runtime on this host without code changes that the
milestone explicitly excludes. Decision taken with the user (FASE 1 STOP):
**defer**. No download performed; no code changed.

Predecessors: RUNTIME-MOE-1 (`509a21e`, real Mixtral BLOCKED — no weights);
dense breadth campaign closed (`3250f97`).

## FASE 1 — Candidate audit (via real HuggingFace metadata, no download)

| Candidate | architectures | Shards | Size (bf16) | f32-in-RAM | Verdict |
|---|---|---|---|---|---|
| **Qwen1.5-MoE-A2.7B-Chat** | `Qwen2MoeForCausalLM` ✓ supported | **8** (+index) | 28.6 GB | **~57 GB** | BLOCKED: sharded **and** 57 GB > 32 GB RAM |
| **Qwen1.5-MoE-A2.7B** (base) | `Qwen2MoeForCausalLM` ✓ | **8** (+index) | 28.6 GB | **~57 GB** | BLOCKED: sharded **and** 57 GB > 32 GB RAM |
| **Phi-mini-MoE-instruct** | `PhiMoEForCausalLM` ✗ **unsupported** | **4** (+index) | 15.3 GB | ~30 GB | BLOCKED: new architecture **and** sharded |

(Qwen1.5-MoE-A2.7B: 14.3B total / 2.7B active, 60 experts, top-4, shared expert.
Phi-mini-MoE: `model_type: phimoe`, 16 experts, top-2, hidden 4096, 32 layers.)

### Two structural blockers, both excluded by this milestone's scope

**1. Every small real MoE of a supported family is sharded.** Atenia's MoE loader
`MoeRuntime::load_from_dir` (runtime.rs:239) picks the **first `*.safetensors`**
in the directory — it does **not** read `model.safetensors.index.json` or merge
shards. Qwen1.5-MoE ships **8 shards**. The milestone rule (FASE 4) is explicit:
*"If the model is sharded and the MoE loader does not support shards: STOP and
report. Do not implement a sharded loader unless it is a minimal, safe change."*
Wiring 8-shard merging into the single-file `SafetensorsReader::open` MoE path is
**not** minimal/safe.

**2. f32-in-RAM footprint exceeds host RAM.** The MoE load path converts every
tensor to f32 (`reader.get(name).to_vec_f32()`) and holds experts resident in
RAM (`ResidentExpertLayer::from_real_layer(&moe, ExpertTier::Ram)` — RAM tier
hardcoded, **no disk spill** in this path). Qwen1.5-MoE-A2.7B at f32 ≈ **57 GB**
→ **exceeds 32 GB RAM**. So even if sharded loading were solved, the model still
would not fit resident. The path has neither a disk tier nor bf16 residency.

**3. Phi-mini-MoE additionally is an unsupported architecture.**
`PhiMoEForCausalLM` / `phimoe` is not in Atenia's MoE families
(Mixtral / QwenMoe / DeepSeekMoe); `classify_family` would reject it → the
"new architecture → STOP" rule applies.

### Conclusion

There is **no supported-family real MoE small enough** to be (a) single-file and
(b) fit as f32 in 32 GB RAM. The smallest (Qwen1.5-MoE-A2.7B, 14.3B) needs **two
excluded code changes**: a sharded MoE loader **and** disk-tier / bf16 residency
in the MoE load path. Per critical rules 4–6 and 9, work stopped at the audit;
**no 28.6 GB download** was performed to demonstrate a guaranteed load failure.

## What WAS done

- Read the MoE loader to establish exact constraints (single-file, f32-in-RAM,
  RAM-tier-only).
- Pulled real HF file/architecture metadata for all three candidates (sizes,
  shard counts, `architectures`).
- Confirmed all three are blocked; took the user's decision to **defer**.

## Decision (FASE 1 STOP)

**Defer RUNTIME-MOE-2 (BLOCKED).** Do not download; do not add a sharded loader
or residency changes under a "validation only / no big loader changes" milestone.
No code touched.

## FASE 10 — Strategic review

**Do we have a usable real MoE yet?** → **No.** The MoE runtime/router/experts/
gating/manifest/CLI are certified at **topology scale** (MOE-FULL-15) and on
real-checkpoint **layer-0 blocks**, but no real full-model MoE has been loaded —
the loader's single-file + f32-RAM design caps viable real MoEs below the size of
any real supported-family MoE.

**Which MoE family goes GREEN?** → **None at real-weight scale.** All remain
topology-certified / experimental.

**What's missing for real Mixtral?** (RUNTIME-MOE-1) ~94 GB real weights (absent)
+ sharded MoE loader + disk residency. Largest gap.

**What's missing for a larger real Qwen-MoE?** Sharded MoE loader **and** a way to
fit ~57 GB (disk tier or bf16 residency, or a ≥64 GB RAM host). The weights are
downloadable (28.6 GB, disk OK on the 830 GB NVMe) — the blocker is the runtime,
not the disk.

## What unblocks real small-MoE validation

A scoped **engine milestone** (code, not validation), in order:
1. **Sharded safetensors loading in the MoE path** — read
   `model.safetensors.index.json` and assemble experts across shards (the dense
   loader already has `ShardedSafetensorsReader`; the MoE path does not use it).
2. **Disk-tier and/or bf16 residency in the MoE load path** — so a 28.6 GB
   (bf16) / 57 GB (f32) model fits via NVMe spill instead of requiring 57 GB RAM.
   The residency substrate exists (`ExpertTier::Disk`, MOE-FULL-8) but
   `load_from_files` hardcodes `ExpertTier::Ram`.
3. Then download **Qwen1.5-MoE-A2.7B-Chat** (supported `Qwen2MoeForCausalLM`,
   28.6 GB) and re-open RUNTIME-MOE-2 to validate real load + generation
   (expect slow, disk-tiered generation — correctness-first).

Until (1)+(2) land, real MoE generation cannot be validated on this host.

## Files modified

- `docs/HANDOFF_RUNTIME_MOE_2.md` — this file (block record + candidate audit).
- `docs/STATUS.md` — RUNTIME-MOE-2 BLOCKED note.

No production code, tests, architecture, families, math, or graph ops changed.
No model weights downloaded; `models/` stays gitignored.

## Deliverable answers

1. **Model chosen?** None provisioned — all candidates blocked (see table).
2. **Why?** Sharded (loader single-file) + f32-RAM > 32 GB (Qwen-MoE) /
   unsupported arch (Phi-mini-MoE).
3. **Size downloaded?** 0 — no download (guaranteed-fail avoided per rules).
4. **Loaded correctly?** N/A.
5. **Coherent text?** N/A.
6. **Real MoE used?** N/A (topology runtime already certified, MOE-FULL-15).
7. **Experts/routing observed?** N/A on real weights.
8. **Problems found?** MoE loader is single-file + f32-RAM-only; all real small
   supported-family MoEs are sharded and exceed 32 GB f32.
9. **Memory used?** N/A.
10. **Times?** N/A.
11. **Family status?** All MoE families remain topology-certified / experimental;
    none real-weight GREEN.
12. **Files modified:** see above.
13. **Commit:** see git log (RUNTIME-MOE: validate small real MoE model —
    blocked).
14. **CI:** docs-only commit → skipped by design (`paths-ignore`).
15. **Next recommendation:** see below.

## Next recommendation

Real MoE validation is **runtime-gated**, not download-gated. The unblock is a
**scoped engine milestone**: (1) sharded safetensors loading in the MoE path,
then (2) disk-tier / bf16 residency in the MoE load path — after which
**Qwen1.5-MoE-A2.7B-Chat (28.6 GB, supported family)** is the right first real
MoE to validate. This is explicitly a code task, outside "validation only", so it
needs its own milestone with its own rules. Meanwhile the dense breadth campaign
(Llama / Qwen2.5 / Gemma 2 / Phi-3, GREEN) is the shippable real-generation story.
