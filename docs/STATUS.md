# Status

An honest snapshot of what is **cabled to production signals** versus what is
still **scaffolding**. This is the document to read before depending on Atenia
for anything. For the history of how each piece landed, see
[MILESTONES.md](./MILESTONES.md); for direction, see
[ROADMAP.md](../ROADMAP.md).

**One-line summary.** Atenia Engine is **early research, single-author, active
development**. The execution-layer thesis is proven end-to-end on the dev
hardware (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, NVMe); the public CLI surface
is stable; numeric certification is real and auditable. It is **not**
production-ready for unsupervised deployment, **not** multi-vendor, and several
execution paths are deliberately conservative or opt-in.

---

## Cabled to production signals

These are exercised end-to-end against real checkpoints (not synthetic) and
locked by regression tests.

- **Tier-aware placement.** A pure planner over
  `(metadata, free_ram, free_vram, kernel_dtype)` assigns every tensor to
  VRAM / RAM / Disk at load time. Llama 2 13B Chat runs end-to-end on
  8 GB VRAM + 32 GB RAM + NVMe with the transparency contract
  `argmax(clean) == argmax(forced 50 % LRU spill)` holding **bit-exactly** at
  13B scale.
- **Numeric certification (certified mode).** The default `certified` path
  (Path B + TF32) is gated on every code change by the four-model F64 fixture
  (TinyLlama, SmolLM2, Qwen 2.5 1.5B, Llama 3.2 1B), threshold
  `max_abs_diff < 0.5` per
  [ADR-004](./decisions/ADR-004-f64-reference-as-default.md). Per-checkpoint
  manifests live in [numcert/](./numcert/) and are reproducible offline.
- **Public CLI.** `atenia generate`, `atenia run`, `atenia probe` are stable
  across the M4.9 → M11 series and documented in [CLI.md](./CLI.md).
  `cargo install --path .` produces a working binary on Windows and Linux.
- **Real end-to-end (RUNTIME-REAL-1).** Re-confirmed on a real 2.2 GB
  TinyLlama-1.1B HF checkpoint, RTX 4070: loads (201 params, ~4.46 GiB
  resident, VRAM/RAM tiered) in ~13 s and greedily generates coherent,
  factually-correct text ("The capital of France is" → "Paris."), deterministic
  vs the committed fixture, EOS + bad-input handling intact. Certified vs f64
  (max_abs_diff 0.076, argmax 4/4). See
  [HANDOFF_RUNTIME_REAL_1.md](./HANDOFF_RUNTIME_REAL_1.md). Dense Llama-family
  text generation is **GREEN**.
- **Non-Llama end-to-end (RUNTIME-REAL-2).** Re-confirmed on a real 2.88 GB
  Qwen2.5-1.5B-Instruct HF checkpoint (`Qwen2ForCausalLM`,
  `llama-2-with-qkv-bias`: tied embeddings + QKV bias + 151936 vocab + θ=1e6),
  RTX 4070: loads 338 tensors (~2.87 GiB resident, VRAM/RAM tiered) in ~12 s,
  generates coherent + factually-correct text ("…is Paris."), deterministic,
  EOS + bad-input intact. Matches f64 to **0.000335** (argmax 4/4 — tighter than
  PyTorch BF16's own 1.53). See
  [HANDOFF_RUNTIME_REAL_2.md](./HANDOFF_RUNTIME_REAL_2.md). Qwen2.5 dense text
  generation is **GREEN**.
- **Gemma 2 end-to-end (RUNTIME-REAL-3).** Re-confirmed on a real 4.87 GiB
  sharded Gemma-2-2B-it HF checkpoint (`Gemma2ForCausalLM`: softcaps @50/30,
  scaled embeddings ×√2304, query_pre_attn_scalar, GeGLU, dual-norm, multi-EOS
  [1,107]), RTX 4070: loads 288 tensors / 2 shards (~6.75 GiB resident,
  VRAM/RAM tiered) in ~90 s, generates coherent + factually-correct text
  ("…is **Paris**. 🇫🇷"), deterministic, EOS (stop=107) + bad-input intact;
  SoftCap regression 8/8; certified↔fast bit-identical (numcert). **Caveat:** no
  f64 reference exists for Gemma 2 (numcert null — PyTorch f64 pipeline not
  wired for this family), so it is GREEN **behaviourally** but lacks the
  Llama/Qwen numerical-certification bar. See
  [HANDOFF_RUNTIME_REAL_3.md](./HANDOFF_RUNTIME_REAL_3.md).
- **Phi-3 end-to-end (RUNTIME-REAL-4).** Re-confirmed on a real ~7.6 GB sharded
  Phi-3.5-Mini-Instruct HF checkpoint (`Phi3ForCausalLM`: fused QKV split,
  fused gate_up split, LongRoPE; MHA), RTX 4070: loads 195 tensors / 2 shards
  (~8.99 GiB resident, VRAM 32 / RAM 163) in ~138 s, generates coherent +
  factually-correct text ("…is Paris."), deterministic, bad-input intact; phi3
  unit suite 23/23 (fused split + LongRoPE); certified↔fast bit-identical
  (numcert). Same caveat as Gemma 2: **no f64 reference** (numcert null), so
  GREEN **behaviourally** only. See
  [HANDOFF_RUNTIME_REAL_4.md](./HANDOFF_RUNTIME_REAL_4.md).
- **Dense breadth campaign — closed.** Four structurally-distinct families
  validated end-to-end on real checkpoints (RUNTIME-REAL-1..4): **Llama**
  (f64 0.076) · **Qwen2.5** (f64 0.000335) · **Gemma 2** (behavioural) ·
  **Phi-3** (behavioural). Remaining gaps (f64 pipeline for Gemma/Phi,
  throughput, larger variants) are tracked as separate milestones, not new
  breadth validation.
- **Real MoE generation (RUNTIME-MOE-1) — BLOCKED.** Attempted real Mixtral
  end-to-end; the real Mixtral-8x7B weights are **not present** locally
  (`models/Mixtral-8x7B-v0.1/` is a config+tokenizer stub; the only mixtral
  weights in-tree are the synthetic MOE-FULL fixtures). MoE runtime/routing
  stays **topology-certified** (MOE-FULL-15: `mixtral_scale` 1.639e-07,
  `mixtral_layer0` 1.164e-10, scope `certified_scaled`) but **no real-weight run
  exists**. Real Mixtral is **experimental / BLOCKED** pending: (1) provisioning
  ~94 GB real weights (disk-tierable on NVMe but slow), and (2) a sharded-loader
  task for the MoE path (real Mixtral is multi-shard; current MoE loader is
  single-file). See [HANDOFF_RUNTIME_MOE_1.md](./HANDOFF_RUNTIME_MOE_1.md).
- **Small real MoE (RUNTIME-MOE-2) — BLOCKED.** Audited the smallest real MoEs
  for a real-weight load: **Qwen1.5-MoE-A2.7B(-Chat)** (`Qwen2MoeForCausalLM`,
  supported) is **8-shard / 28.6 GB / ~57 GB as f32** and **Phi-mini-MoE**
  (`PhiMoEForCausalLM`) is an **unsupported architecture** + 4-shard. The MoE
  loader is **single-file + f32-into-RAM** (`MoeRuntime::load_from_dir` picks the
  first `.safetensors`; `ExpertTier::Ram` hardcoded, no disk spill), so no
  supported-family real MoE fits on the 32 GB host. Deferred without downloading
  (avoiding a guaranteed-fail 28.6 GB fetch). Unblock = a scoped **engine**
  milestone: (1) sharded safetensors loading in the MoE path, (2) disk-tier /
  bf16 residency — then validate Qwen1.5-MoE-A2.7B. See
  [HANDOFF_RUNTIME_MOE_2.md](./HANDOFF_RUNTIME_MOE_2.md).
- **Sharded MoE loading (MOE-PROD-1) — done.** The MoE runtime now loads
  **sharded** checkpoints (`model.safetensors.index.json` + multiple shards) via
  a new `MoeWeightSource` (single-file **or** sharded), proven **bit-for-bit
  identical** to single-file (`max_abs_diff == 0.0`) with clear errors for
  missing shard / missing tensor / corrupt index. This removes the **first** of
  the two RUNTIME-MOE-2 blockers. The **second** — the compute backend holding
  every weight as f32 in RAM (~57 GB for Qwen1.5-MoE > 32 GB) — is **still
  open** and needs a separate bf16/disk-backed-residency engine milestone. See
  [HANDOFF_MOE_PROD_1.md](./HANDOFF_MOE_PROD_1.md).
- **Disk-backed MoE expert residency (MOE-PROD-2) — done.** The **second**
  RUNTIME-MOE-2 blocker (experts held as f32 in RAM) is removed:
  `ATENIA_MOE_EXPERT_TIER=disk` streams each graph-MoE layer's experts onto NVMe
  during load (peak RAM ~one layer, not the whole model) and runs them through
  the certified `ResidentExpertLayer`. Disk-tier output is **bit-for-bit
  identical** to RAM (`max_abs_diff == 0.0`, `tests/moe_residency_tier_test.rs`);
  the RAM default is byte-identical to before. Estimate: Qwen1.5-MoE-A2.7B drops
  from ~57 GB f32 RAM to ~3 GB steady (experts ~28.6 GB on NVMe) → **fits** the
  32 GB host. Both engine blockers are now gone; reopening RUNTIME-MOE-2 is an
  environment step (download + run). Caveats: disk-tier generation is slow (no
  per-token expert cache yet); DeepSeek/MLA not tiered. See
  [HANDOFF_MOE_PROD_2.md](./HANDOFF_MOE_PROD_2.md).
- **First real MoE generation (RUNTIME-MOE-2 reopened).** Downloaded the real
  **Qwen1.5-MoE-A2.7B-Chat** (27 GB, 8 shards, `Qwen2MoeForCausalLM`, 60 experts
  top-4 + shared, 24 layers) and ran it end-to-end via `atenia moe-generate`
  with `ATENIA_MOE_EXPERT_TIER=disk`: **loads with experts streamed to NVMe
  (~50 GB), RAM bounded ~4 GB** (vs ~57 GB f32), and generates **coherent real
  text** — "What is the capital of France?" → " The capital of France"; "Rust is
  a programming language that" → " is designed to be fast,". Routing (top-4 of
  60 + shared), manifest gate, and fail-loud opt-in all verified; no dense
  fallback. Slow (~2–4 min/token, disk-tier, no expert cache). **Qwen-MoE is now
  real-GREEN behaviourally** (no full f64 for the 14.3 B model; certified at
  topology/block scale, MOE-FULL-15). See
  [HANDOFF_RUNTIME_MOE_2_REOPENED.md](./HANDOFF_RUNTIME_MOE_2_REOPENED.md).
- **Expert-cache integration (MOE-PROD-3).** The disk-tier MoE node now threads
  the existing per-layer `ExpertCache` (`forward_cached`) so repeated routed
  experts are served from RAM (`ATENIA_MOE_EXPERT_CACHE`, default `2*top_k`;
  `ATENIA_MOE_CACHE_STATS=1` reports the hit ratio). Bit-exact (disk+cache == RAM
  `max_abs_diff==0.0`; cached==uncached unit test). **Honest finding:** on real
  Qwen1.5-MoE it gets a **22.7 % routed hit ratio** but **no end-to-end speedup**
  (~3101 vs ~2905 s) — the bottleneck is **load-time NVMe tiering** (50 GB f32 as
  4392 files) + the **shared expert read every token** + per-file overhead, not
  routed re-reads. The real levers are bf16 tier storage + persisting the tier
  across runs. See [HANDOFF_MOE_PROD_3.md](./HANDOFF_MOE_PROD_3.md).
- **Persistent expert tier (MOE-PROD-4).** `ATENIA_MOE_TIER_PERSIST=1` writes the
  disk tier under deterministic names in `<base>/moe_tier/<model_id>/` with a
  `tier_manifest.json` and **reuses** it on a later load instead of rewriting the
  ~50 GB. Bit-exact (reuse mtimes unchanged + identical generation;
  `tests/moe_tier_persist_test.rs`); regenerates on file loss. **Real result** on
  Qwen1.5-MoE: cold load+gen **3757 s** → warm **2445 s** (**~35 % / ~22 min
  saved**; 4392 expert files reused, 0 rewritten, output identical) — the lever
  MOE-PROD-3 pointed at. New bottleneck: shard read + f32 expert assembly (scope
  A still assembles experts; scope B would skip that). Default off = unchanged
  ephemeral tier. See [HANDOFF_MOE_PROD_4.md](./HANDOFF_MOE_PROD_4.md).
- **Warm backend reconstruction (MOE-PROD-5, scope C).** A warm load now rebuilds
  the **whole** MoE backend (experts + attention + embed + lm_head + router +
  gate) **directly from the persistent tier — no shard read, no expert
  assembly** ("reconstructing N layers"), with a safe fallback to the certified
  shard path on any manifest/file doubt. Bit-exact (proved by deleting the shard
  files and still generating identically; `tests/moe_tier_reconstruct_test.rs`).
  **Real result** on Qwen1.5-MoE: warm load+gen **2445 s → 701 s (~71 % / ~29 min
  faster)**, output identical. New bottleneck is the per-token disk-tier
  generation, not the load. (Fixed a shared-expert FFN-width bug found by the
  real run.) Default off = unchanged. See
  [HANDOFF_MOE_PROD_5.md](./HANDOFF_MOE_PROD_5.md).
- **BF16 tier + shared-expert cache (MOE-PROD-6).** Cuts the per-token
  generation I/O that MOE-PROD-5 left as the bottleneck, without touching the
  MoE math/routing/outputs. (a) **BF16 expert tier**: routed + shared experts
  persist as bf16 (half the NVMe bytes), **auto-detected per tensor** — only
  bf16-representable values are truncated (lossless `>>16`/`<<16` round-trip);
  arbitrary f32 stays f32. The read path upcasts bf16→f32 via the existing
  `ensure_cpu` disk arm; warm reads detect the dtype by file size. (b) **Pinned
  shared-expert cache**: the shared expert (read *every* token, ~4× a routed
  expert on Qwen-MoE) is materialised once per layer and reused. Both are
  bit-exact and have env escape hatches (`ATENIA_MOE_TIER_BF16=0`,
  `ATENIA_MOE_SHARED_CACHE=0`) plus the certified shard fallback; manifest
  bumped to v3 (per-tensor dtype). **Real result** on Qwen1.5-MoE (rigorous
  same-prompt A/B): warm load+gen **350 s → 204 s (~42 %, −146 s)** — bf16 tier
  −120 s + shared cache −26 s — with the on-disk tier shrinking **53.3 → 28.6
  GiB**; output identical (token ids `16, 15`). New bottleneck: warm
  reconstruction + CPU expert matmul. See
  [HANDOFF_MOE_PROD_6.md](./HANDOFF_MOE_PROD_6.md).
- **BF16 backend tier + warm profiling (MOE-PROD-7).** Extends the bf16 tier to
  the **backend** (embed/lm_head/attention/router/gate) and replaces the
  element-by-element `read_f32_named` with a memcpy-speed bulk reader
  (`read_named_to_f32`, dtype-by-size). Adds **per-phase warm profiling**.
  **Key measured finding:** the warm **reconstruction is only 34.6 s of the
  204 s** warm wall (~17 %); the other **~168 s (~82 %) is generation compute**
  (CPU matmul of the expert FFNs + attention). So backend loading was **not** the
  real bottleneck — the bf16 backend + bulk reader make the load faster (tier
  28.6 → 26.7 GiB, cold 1942 → 1755 s) and bit-exact (`16, 15`), but the warm
  wall is unchanged because generation dominates. **The warm-load block is
  complete (ROI exhausted); the next block is generation compute** (an
  architectural GPU-offload-vs-CPU-GEMM decision). See
  [HANDOFF_MOE_PROD_7.md](./HANDOFF_MOE_PROD_7.md).
- **Generation compute (MOE-PROD-8) — closes the bit-exact warm-path block.**
  Attacks the generation graph execution that MOE-PROD-7 proved dominant.
  (a) **Parallel `matvec`** (dense + MLA): the expert-FFN f64 matmul now runs its
  independent per-output-row reductions across the 24 cores via rayon —
  **bit-identical** (same f64 accumulation, only the thread assignment changes).
  (b) **Single-copy expert resolve**: the Disk-tier `materialize` read the file
  then cloned the buffer; it now reads straight into one `Vec<f32>` (half the
  alloc/memcpy per miss). Adds generation-phase profiling. **Real:** warm
  **204 → 184 s (~10 %)**, bit-exact (`16, 15`). **Block conclusion:** the warm
  reconstruction is now ~34 s (~18 %); the remaining ~94 s is CPU graph
  execution using f64 reference accumulation. Further speedup needs an
  **architectural decision** (GPU offload or f32 accumulation) that **breaks
  bit-exactness** with the certified f64 reference — so the bit-exact CPU block
  ends here. See [HANDOFF_MOE_PROD_8.md](./HANDOFF_MOE_PROD_8.md).
- **MOE-PROD series CLOSED (1 → 8).** Warm path **350 → 184 s (~47 %, same-prompt,
  bit-exact)**; tier 53 → 26.7 GiB; cold 3757 → 1755 s; reconstruction now ~34 s.
  Full closing audit (per-milestone, cumulative metrics, bottlenecks, lessons,
  f64-path limits) in [HANDOFF_MOE_PROD_SERIES.md](./HANDOFF_MOE_PROD_SERIES.md).
  The residual bottleneck is CPU f64 graph execution, which cannot shrink further
  without changing the numerics — designed (not implemented) as the next series
  **NUMERIC-POLICY / MOE-PERF** (tiered numeric policy + tolerance certificate;
  f32 / bf16 / GPU offload / Tensor Cores evaluation) in
  [PROPOSAL_NUMERIC_POLICY_MOE_PERF.md](./PROPOSAL_NUMERIC_POLICY_MOE_PERF.md).
- **NUMERIC-POLICY-1 + MOE-PERF-1 — first block of the new series.** Makes MoE
  compute precision an **explicit, selectable, certifiable** choice without
  breaking the certified f64 path. `NumericPolicy { Certified, Strict, Fast }`
  (`ATENIA_NUMERIC_POLICY`, default + fallback **Certified**); `Strict` runs the
  **expert FFN** in f32 (router stays f64 → routing identical); a
  `PolicyCertificate` (logit `max_abs_diff`/`rmse`/argmax-rate + **token
  equality**) certifies a non-Certified run vs the f64 reference, else falls back.
  **Real Qwen1.5-MoE (CPU, same-prompt A/B):** Certified 206 s → **Strict 180 s
  (~13 %, identical tokens `16,15`)**. **CUDA feasibility audit:** the GPU is
  present (RTX 4070, CUDA 13.2) but enabling it gives **0 speedup** — the expert
  FFN **bypasses the GPU dispatch** (direct CPU call), so GPU offload needs
  explicit wiring + VRAM streaming + a `Fast` certificate = the next block
  (**MOE-PERF-2**, scoped, not implemented). See
  [HANDOFF_NUMERIC_POLICY_1.md](./HANDOFF_NUMERIC_POLICY_1.md) +
  [HANDOFF_MOE_PERF_1.md](./HANDOFF_MOE_PERF_1.md).
- **MOE-PERF-2 — GPU expert FFN (`NumericPolicy::Fast`): implemented, certified,
  but SLOWER (honest STOP).** Wires the expert FFN matmul to `cuda_matmul` under
  `Fast` (f32 GPU, automatic CPU-f32 fallback, router stays f64). It is correct,
  **certified** (real tokens identical `16,15`, GPU used — VRAM 625 MiB / 8 GB),
  and fallback-safe — **but ~15 % slower** on the real model (Certified 180 s /
  Strict 179 s / **Fast 207 s**). **Why exactly:** the expert FFN is per-token
  `M=1` GEMVs over **transient** weights (each routed expert resolved from disk,
  used once), so every GPU call pays a full PCIe weight upload that exceeds the
  tiny GEMV compute; the 24-core CPU reading already-resident RAM wins. The GPU
  needs **VRAM-resident reused weights** (the cached shared expert) or **batched
  `M=seq` prefill** to win — the next block. `Strict` (f32 CPU) stays the fast
  default; `Certified` the reference. See
  [HANDOFF_MOE_PERF_2.md](./HANDOFF_MOE_PERF_2.md).
- **MOE-PERF-3 — shared-expert VRAM residency: STOP (no ROI), with hard
  evidence.** Before building a risky CUDA-resident path, **measured the
  ceiling**: instrumentation (`MoE fwd compute: shared/routed`) shows the
  shared-expert matmul is **0.24 s (max-new 2) / 0.88 s (max-new 8) — < 0.4 % of
  the 185–237 s wall**. Even instant residency saves **< 1 s**. The generation is
  **I/O/load-bound** (disk-tier resolves ~13 GiB + GraphBuilder attn/lm_head),
  **not** expert-matmul-bound (shared+routed matmul < 1 s total). So the whole
  "expert FFN compute" direction (f32/GPU/Tensor-Cores/residency) is a dead end
  here — no expert-matmul change can move a wall that is < 1 % matmul. The only
  remaining MoE-perf levers are **I/O/load** (bigger cache — hit ratio already
  55 % at max-new 8; fewer tier files), each gated by the same ROI-ceiling
  measurement. `Strict` stays the fast default. See
  [HANDOFF_MOE_PERF_3.md](./HANDOFF_MOE_PERF_3.md).
- **MOE-IO-1 — I/O profiling: dominant bottleneck is antivirus, code candidates
  eliminated (STOP).** New line attacking I/O/load. Instrumentation (`tier
  resolve(disk)=..s @ ..MB/s`) shows the **disk-tier resolve is ~50 % of the
  wall** (~100 s at max-new 2; expert matmul < 1 s). Measured ROI ceiling of each
  code candidate and **eliminated all**: (a) **bigger cache** — no ROI (misses
  are first-touch: 297→291, hit 22.7→24.2 %); (b) **consolidate the 4659 tier
  files** — a read benchmark shows cold 68 MB/s vs warm 1282 MB/s with only
  ~1 ms/file open overhead, i.e. the cost is **cold content scanning** (Windows
  Defender RTP is ON; 68 MB/s = Defender throughput) which is **per-byte** —
  consolidation would make Defender scan whole 330 MB layer blobs instead of the
  top-k ~28 MB, *worse*. **Root cause: AV real-time scan of tier files on first
  open — environmental, not a code defect.** The real fixes live elsewhere:
  **operational** (exclude the tier dir from AV → ~NVMe speed → ~50 % off the
  wall; the CLI now detects + recommends it) and **NUMERIC-POLICY** (quantised
  experts = fewer bytes to scan, tolerance-certified). `Strict` stays the fast
  default. See [HANDOFF_MOE_IO_1.md](./HANDOFF_MOE_IO_1.md).
- **NUMERIC-POLICY-2 — int8 quantized expert tier (certified, real win).** Acts
  on MOE-IO-1's only code lever (**fewer bytes**): the routed + shared **experts**
  persist as **per-row symmetric int8** (`rows*4 + numel` B ≈ half the bf16 tier);
  the router/shared-gate/backend stay bf16/f32. The dequantised f32 flows through
  the unchanged forward (Certified f64 / Strict f32). `ATENIA_MOE_TIER_QUANT=int8`
  (default off → bf16); manifest **v5** (per-entry dtype + bytes); warm
  reconstruction size-detects f32/bf16/qint8; mismatch → certified shard
  fallback. **Certified first, cheaply:** an in-memory int8 *simulation*
  (`ATENIA_MOE_QUANT_SIM=int8`, reusing the bf16 tier) produced **identical
  tokens** (`16,15` and the 8-token sequence) on the real model. **Real
  benchmark:** tier **26.7 → 14.3 GiB (−46 %)**, resolve **~90 → 45 s (~−50 %,
  68 → 287 MB/s)**, warm wall **~180 → 142 s (~−21 %)**, tokens identical →
  certified; cold 1942 → 1543 s. int4 deferred (sim-certify first, never on
  intuition). `Strict` stays the fast default; `Certified` the reference. See
  [HANDOFF_NUMERIC_POLICY_2.md](./HANDOFF_NUMERIC_POLICY_2.md).
- **NUMERIC-POLICY-3 — certification governance.** Closes the audit's #1 gap
  (certification was manual/offline/unpersisted). Adds a persisted
  **`NumericCertificate`** (JSON, per model × policy × tier dtype: expected vs
  observed tokens, argmax/tokens match, drift, manifest/code/cert/prompt-set
  versions, pass/fail), an offline **validation prompt set** (6 deterministic
  greedy cases, ids `0..=9`), a **runner** (`atenia moe-certify` / `certify_model`
  — Certified-vs-candidate on one lossless load via policy toggle + int8 sim), a
  **loader/validator** (`is_valid_for`: every field + pass), and a **runtime
  guard** (`ATENIA_NUMERIC_REQUIRE_CERT=1`): a **lossy (int8) tier without a valid
  cert is refused**, a non-Certified compute policy on a **lossless tier without a
  cert falls back to `Certified`**; the effective mode is logged
  (`numeric mode: policy=… tier=… cert=…`). **Default off → unchanged opt-in;
  `Certified` default + bit-exact path untouched.** Tests: cert units + governance
  integration (certify → refuse-uncertified → allow-with-cert) + full regression
  green. See [HANDOFF_NUMERIC_POLICY_3.md](./HANDOFF_NUMERIC_POLICY_3.md).
- **MODEL-INTAKE-1 — architecture compatibility layer ("say yes more often,
  safely").** Closes the coverage audit's #1 blocker (an unknown
  `architectures[0]` was a blanket hard-reject). Native resolution is
  **unchanged** (certified/supported families byte-identical); only on a registry
  miss does a new explicit layer (`model_adapters::compat`) decide: a curated
  **evidence-gated allowlist** of distinct-but-Llama-compatible arch strings
  (`LLaMAForCausalLM`, `YiForCausalLM` → `LlamaForCausalLM`), or an **opt-in
  generic decoder path** (`ATENIA_INTAKE_GENERIC=1`) that runs an unknown arch as
  Llama **iff** `check_llama_topology` passes (positive dims, head/GQA
  divisibility, **no** specialised-family fields = Gemma soft-caps / Gemma-3
  dual-RoPE / Phi-4 partial rotary), loudly **UNCERTIFIED** and fail-loud at
  weight binding. No silent fallback: every reject names the reason + the opt-in.
  `atenia capabilities` surfaces the allowlist + opt-in. Tests: 9 compat units +
  5 crate-boundary integration, green. **Default off → unchanged behaviour.** See
  [HANDOFF_MODEL_INTAKE_1.md](./HANDOFF_MODEL_INTAKE_1.md) +
  [MODEL_COVERAGE_EXECUTIVE_AUDIT.md](./MODEL_COVERAGE_EXECUTIVE_AUDIT.md).
- **CERTIFY-BREADTH-1 — numeric-certification infrastructure for Phi-3 / Gemma
  (evidence pending, no fabrication).** The coverage gap between *functionally
  validated* and *numerically certified* for Phi-3 / Gemma is exactly one missing
  artefact per model: a **PyTorch-F64 reference** (ADR-004). This milestone
  delivers the **reusable infrastructure** to close it — a parametrised F64
  generator (`tests/fixtures/generate_f64_reference.py`, generalising the
  per-model `generate_f64.py` scripts) and a **family-agnostic CPU-F32-vs-F64
  harness** (`tests/certify_breadth_f64_validation_test.rs`, driving the forward
  through the adapter layer; pure CPU, no CUDA; single-file + sharded) with
  `#[ignore]` validations for Gemma-2-2B / Gemma-3-1B / Phi-3.5 and **6 CI unit
  tests** for the pure logic. The Gemma-2 + Phi-3.5 manifests are re-pointed at
  the harness with an exact reproduction recipe; a new Gemma-3-1B manifest wires
  the slot. **Every `max_abs_diff_vs_f64` stays `null` — no number is
  fabricated, ADR-004 is not relaxed, "certified" is not redefined.** Completing
  the real certification is a pure *execution* step on adequate hardware (free
  RAM ≈ 8 GiB Gemma-3-1B / 21 GiB Gemma-2-2B / 30 GiB Phi-3.5 for the F64 pass;
  build-time free RAM here was 12.7 GiB). Certified families remain **Llama +
  Qwen2**; Phi-3 / Gemma are *certification-infrastructure-ready, evidence-pending*.
  See [HANDOFF_CERTIFY_BREADTH_1.md](./HANDOFF_CERTIFY_BREADTH_1.md).
- **CERTIFY-BREADTH-2 (Gemma-3-1B, Gemma-2-2B) — real evidence landed.** Ran the
  committed harness on **Gemma-3-1B-IT** (**max_abs_diff = 0.001249**, ≈400×
  inside the 0.5 ADR-004 gate, argmax 4/4) and **Gemma-2-2B-IT** (**max_abs_diff
  = 0.058711**, ≈8.5× inside the gate, argmax 4/4 — higher as expected from the
  SoftCap/scaled-embedding ops, but comfortably passing). Both manifests now carry
  the measured numbers (were null) + their committed f64 fixtures
  (`tests/fixtures/gemma{3_1b,2_2b}_reference/`). **Gemma 3 and Gemma 2 →
  CERTIFIED (CPU-F32, ADR-004 primary metric)**; the GPU-TF32 production-kernel
  number is a separate pending field for each. Certified families: **Llama,
  Qwen2, Gemma 3, Gemma 2 (CPU-F32)**. **Phi-3.5** (f64 pass needs ~30 GiB)
  remains evidence-pending.
- **MOE-INTEGRATE-1/2 — MoE reachable from the normal runtime (opt-in).**
  INTEGRATE-1 added a declarative **`moe` section** to the Adapter Toolkit DSL
  (`MoeFamilyKind`, typed `ExpertLayout`/`RouterNaming`/`SharedExpertNaming`,
  `auto` defaults deferring to `config.json`; describe + validate only). 
  INTEGRATE-2 wired the **routing**: `atenia generate` now runs `diagnose_moe`
  on a checkpoint and `decide_route` decides — **dense → unchanged**; a runnable,
  certified **Mixtral / Qwen-MoE** family **with the opt-in** (`ATENIA_ENABLE_MOE=1`)
  → routed to the controlled `MoeRuntime` (tokenize → `controlled_moe_generate` →
  decode); **without the opt-in → fail loud**; unsupported variant / DeepSeek-MoE
  (deferred) / unrecognised → fail loud. The dense loader's MoE guard and the
  MoE runtime are untouched; default behaviour for dense models is identical.
  See `docs/MOE_ADAPTER_SPEC_AUDIT.md` + `docs/ADAPTER_TOOLKIT_V2.md` §17.
- **MOE-CERT-1 — MoE certification ladder formalised (ADR-007, docs only).**
  ADR-004's global-F64 contract is infeasible for real MoE (full weights don't
  fit F64 RAM — Mixtral ~373 GB; and a forward only exercises the top-k routed
  experts). `docs/decisions/ADR-007-moe-certification-ladder.md` formalises
  **certification by decomposition**: obligations **C1–C5** (per-expert / router
  / attention / assembly-topology / active-path) over an **L0–L4 ladder**
  (topology → component → assembly → active-path → dense-equivalent), **reusing
  the ADR-004 `max_abs_diff < 0.5` + argmax bar verbatim — no threshold changed**.
  Reporting discipline added so a partial **`MoE-certified Ln`** status is never
  read as the dense ADR-004 `CERTIFIED` (mandatory level qualifier; headline =
  lowest fully-passed level; distinct `schema_variant: "moe-decomposition"`
  manifest). All three MoE families sit at **L0** today; raising them is the
  MOE-CERT-2/3/4 work. **Definition only — no harness, no execution, no runtime /
  loader / `MoeRuntime` / Adapter Toolkit change.** Cross-referenced from
  ADR-004, `CERTIFICATION.md`, `FAMILY_COVERAGE_AUDIT.md`,
  `MODEL_FAMILY_VALIDATION.md`; built on `docs/MOE_CERTIFICATION_AUDIT.md`.
- **MOE-CERT-2 — Qwen-MoE certified by decomposition (ADR-007 C1+C2, real
  weights).** First above-L0 evidence on the ladder. A test-only harness
  (`tests/moe_cert2_qwen_decomposition_test.rs`) certifies, on the **real**
  Qwen1.5-MoE-A2.7B **layer-0** weights vs a **float64** reference computed **one
  expert at a time** (`fixtures/moe/generate_qwen_moe_decomposition_reference.py`
  → `qwen_moe_decomp_ref.safetensors`): **C1** — all **60** routed experts under
  the ADR-004 gate, worst `max_abs_diff` **1.192e-7** (~4.2e6× inside 0.5),
  exhaustive (no sampling); **C2** — top-k expert **set equality** vs the f64
  reference (`[10,24,39,42]`, hard gate) + **routing margin 0.160333** reported.
  **C3** (attention) reused from the existing mechanism cert (MOE-FULL-13,
  5.960e-08). Result: **Qwen-MoE — MoE-certified L1 on the layer-0 representative
  scope**; whole-model headline stays **L0** until C1/C2 extend to all 24 layers
  (`docs/numcert/qwen1.5-moe-a2.7b.moecert.json`, `schema_variant:
  moe-decomposition`). A non-ignored CI smoke exercises the harness on the tiny
  committed fixture. **No `src/` change** — no runtime / loader / `MoeRuntime` /
  Adapter Toolkit / numerics touched; gate not lowered; fail-loud preserved.
- **MOE-CERT-2-ext — Qwen-MoE whole-model L1 (ADR-007 C1+C2, all 24 layers).**
  Extends MOE-CERT-2 from layer 0 to the **entire** real Qwen1.5-MoE-A2.7B. The
  generator (`--all`) emits a float64 reference for all layers (one expert at a
  time, one layer at a time; 7 layers span two shards →
  `qwen_moe_decomp_ref_all_layers.safetensors`, ~11.8 MB); the harness reads each
  layer from whichever shard(s) hold it and checks: **C1** all **1440** routed
  experts (24×60) under the ADR-004 gate, **global worst `max_abs_diff` 4.768e-7**
  (layer 6, expert 37), **0 failures**; **C2** top-k **set equality** on all 24
  layers, **0 failures**, **min routing margin 0.001834** (layer 6, still no
  flip). **C3** reused (MOE-FULL-13). Result: **Qwen-MoE — MoE-certified L1
  (whole model)**; manifest updated to `ladder_level_whole_model: L1`. **No
  `src/` change**; ADR-004 gate not lowered; fail-loud preserved. L2/L3/L4 remain
  pending. `tests/moe_cert2_qwen_decomposition_test.rs::cert2_real_qwen_moe_all_layers`
  (real run: 1440 experts, 380 s).
- **MOE-CERT-3 — Qwen-MoE whole-model L2 (fold C4; manifest/docs only).** Folds
  the existing **C4** (assembly/topology) evidence into the MoE manifest to reach
  **L2 = L1 + C4**. C4 = the Qwen-MoE **scale-topology** end-to-end certificate
  (`tests/moe_scale_cert_test.rs::qwen_moe_scale_topology_certifies`,
  MOE-FULL-15): full-forward vs HF f64 **max_abs_diff 1.490e-7** (< 0.5; the test
  asserts the stricter 1e-3) + per-position argmax + generate→EOS + determinism,
  on the Qwen-MoE topology (16-expert / top-4 / shared-sigmoid / GQA / qkv-bias,
  reduced dim, random weights). Per ADR-007 this is the designated C4 evidence; it
  certifies the **assembly mechanism**, gaining real-weight force combined with
  the whole-model C1/C2 (real weights, all layers). Result: **Qwen-MoE —
  MoE-certified L2 (whole model)**; manifest `ladder_level_whole_model: L2`.
  **No `src/` change** — manifest + docs only; ADR-004 gate not lowered; C4 test
  re-run green (1.490e-7). **L3 (C5 active-path) and L4 (global F64) remain
  pending** — not L3, not L4, not the dense ADR-004 `CERTIFIED`.
- **MOE-CERT-4 — Qwen-MoE whole-model L3 (C5 active-path).** Adds the final
  reachable obligation **C5** (active-path parity) → **L3 = L2 + C5**. A
  test-only harness (`tests/moe_cert4_qwen_active_path_test.rs`) runs Atenia's
  controlled **`MoeRuntime` full forward of the REAL Qwen1.5-MoE-A2.7B**
  (disk-tier, ~few GB RAM) on a canonical 4-token input and gates it against a
  **float64 reference computed one decoder layer at a time** with HuggingFace's
  own Qwen2Moe module (`fixtures/moe/generate_qwen_moe_c5_reference.py`,
  driver validated against the tiny HF fixture at 1.348e-8 before use; classic→
  packed expert conversion matching Atenia's certified MOE-15 convention). Result:
  end-to-end **max_abs_diff 1.866e-4** (< 0.5; ~2680× inside) + **per-position
  argmax exact 4/4** (`[18493,1,1,1]`) + deterministic. This is the **F64-active**
  form (one layer at a time, never the whole model in F64 → **not L4**). Result:
  **Qwen-MoE — MoE-certified L3 (active-path-certified, whole model)**; manifest
  `ladder_level_whole_model: L3`. **No `src/` change** — the harness only *calls*
  the existing runtime; ADR-004 gate not lowered; fail-loud preserved. **Only L4
  (global F64) remains, reserved/unreachable** — not L4, not the dense ADR-004
  `CERTIFIED`. (Real run ~950 s, disk-tier.)
- **MLA-0 — DeepSeek MLA: YaRN + dense-first layer + routing convention
  (experimental).** Closes the three config-confirmed prerequisites for a faithful
  DeepSeek-V2-Lite cert (per `docs/DEEPSEEK_V2_LITE_FEASIBILITY.md`), in the
  experimental MLA path only. **(1) YaRN** (`src/moe/mla.rs`): `inv_freq`
  NTK-by-parts reparam + **`mscale²`** on the attention softmax scale (active at
  *every* position), a faithful port of HF `DeepseekV2YarnRotaryEmbedding`.
  **(2) Dense-first** (`first_k_dense_replace`): `DeepseekFfn { Moe | Dense }` +
  `DenseFfn`; `load_core` skips MoE assembly for dense layers; `build_deepseek`
  builds the dense SwiGLU FFN. **(3) Routing convention**: `norm_topk_prob` drives
  renorm (`true`→Atenia, `false`→HuggingFaceQwen no-renorm+ungated-shared).
  Validated on a tiny real `DeepseekV2ForCausalLM` (`.double()`) configured like
  V2-Lite (YaRN + `first_k_dense_replace=1` + `q_lora_rank=null` +
  `norm_topk_prob=false`): full forward vs HF **max_abs_diff 9.072e-5** (< 0.5,
  argmax exact) + greedy→EOS. **Backward-compatible**: no `rope_scaling`→plain RoPE,
  `first_k_dense_replace=0`→all-MoE, `norm_topk_prob` default `true`→renorm; Mixtral/
  Qwen untouched. Regression: DeepSeek 4/4+2/2 unchanged, scale 3/3, qwen 6/6,
  mixtral 4/4, **full lib 838 passed**. **No threshold lowered; no real download/
  cert; no latent cache / Q-LoRA / V3 / productive-loader lift.** See
  `docs/HANDOFF_MLA_0.md`. Next: **MLA-1** (provision V2-Lite → ADR-007 C1–C5).
- **MLA-1 (C1+C2) — DeepSeek-V2-Lite per-expert + router certified (real, partial
  L1).** Reusing ~95% of the Qwen MOE-CERT-2-ext decomposition tooling (DeepSeek's
  expert/router tensor names are identical), certifies on the **real**
  DeepSeek-V2-Lite weights: **C1** all **1664** routed experts (26 MoE layers × 64;
  dense layer 0 skipped) under the ADR-004 gate, **global worst `max_abs_diff`
  1.907e-6** (layer 26, expert 57), exhaustive, 0 failures; **C2** top-6 expert
  **set equality** on all 26 MoE layers (hard gate, 0 failures), **min routing
  margin 0.011981** (layer 23). **C3** reused at the mechanism level (DeepSeek MLA
  cert 9.999e-6 + MLA-0 9.072e-5); **C4** (deepseek_scale 7.806e-3) available, not
  folded; **C5** pending. Result: **DeepSeek-V2-Lite — partial L1** (C1+C2
  real-weight + C3 mechanism); manifest `docs/numcert/deepseek-v2-lite.moecert.json`
  (`schema_variant: moe-decomposition`). **No `src/` change** — harness only calls
  the certified MoE primitives; ADR-004 gate not lowered; MLA/YaRN/runtime/loader/
  numerics untouched. Real run ~505 s. See `docs/HANDOFF_MLA1_C1_C2.md`. Next:
  C4 fold → L2, C5 active-path → L3.
- **MLA-1 (C4) — DeepSeek-V2-Lite whole-model L2 (fold C4; manifest/docs only).**
  Folds **C4** (assembly/topology) into the manifest → **L2 = L1 + C4**. Primary C4
  evidence: the **MLA-0 V2-Lite-topology full-forward cert** vs HF f64 = **9.072e-5**
  (the EXACT V2-Lite conventions — YaRN + dense-first + no-renorm + MLA + q_lora=null,
  end-to-end, argmax exact); corroborated by the **deepseek_scale** DeepSeek-MoE
  topology cert = **7.806e-3**. Chosen over deepseek_scale-alone because MLA-0
  carries V2-Lite's *correct* conventions (deepseek_scale uses renorm / no dense-first
  / no YaRN). Result: **DeepSeek-V2-Lite — MoE-certified L2 (whole model)**; manifest
  `ladder_level_whole_model: L2`. C3+C4 mechanism caveat; C5 (active-path) → L3 and
  L4 (global F64) remain pending. **No `src/` change** — manifest/docs only (both C4
  tests re-run green); ADR-004 gate not lowered. Not dense `CERTIFIED`, not L3/L4.
- **MLA-2 — disk/bf16 expert-tier for the experimental DeepSeek MLA forward
  (unblocks C5).** The MLA forward previously consumed a RAM-f32 `RealMoeLayer`
  per layer, forcing the whole real DeepSeek-V2-Lite into RAM (~58.5 GB f32 on a
  31.7 GB host) — the C5 blocker (purely **residency**, not numerics). MLA-2 routes
  `DeepseekFfn::Moe` through the already-certified `Arc<ResidentExpertLayer>`
  (uncached `forward`) at the env tier: default `Ram` (bit-identical to before) or
  `ATENIA_MOE_EXPERT_TIER=disk` (experts on NVMe, ~0 host RAM → peak **~4 GB**).
  Convention is set from `norm_topk_prob` (ungated shared, so the gate-based
  `resolve_convention` would mis-key V2-Lite); load-time `self_validate_residency`
  now checks under that convention. **Bit-identical**: MLA-0 full forward on the
  disk tier = the RAM tier exactly (both 9.072e-5 vs HF, argmax exact, greedy
  identical, deterministic). MLA attention stays RAM; no cache, no perf work, no
  numerics change; Qwen/Mixtral untouched (full lib suite 838/0 single-threaded).
  **C5 not run yet — DeepSeek-V2-Lite stays MoE-certified L2**; L3 is now
  *technically unblocked* but **not certified**. See `docs/HANDOFF_MLA_2.md`.
  *(Superseded by the MLA-1 C5 + MLA-3 bullet below: C5 was run and **PASSED** →
  DeepSeek-V2-Lite is now **MoE-certified L3**.)*
- **MLA-1 (C5) + MLA-3 — DeepSeek-V2-Lite reaches MoE-certified L3
  (active-path-certified).** The first real C5 run FAILED (`max_abs_diff 2.032`,
  argmax 3/4) and the per-layer diagnosis (a HF f32-vs-f64 control proved the model
  is f32-stable to `3.1e-5`, so it was a bug, not drift) root-caused a **YaRN
  `mscale`** error: `attn_scale()` multiplied the *whole* `q·k` softmax scale by
  `mscale²`, but HF's `attention_scaling = get_mscale(factor,mscale)/get_mscale(factor,
  mscale_all_dim)` (= **1.0** for V2-Lite, `mscale==mscale_all_dim`) attaches only to
  the decoupled-RoPE part. **MLA-3** fixed it (`src/moe/mla.rs`: `attn_scale` = base;
  new `attention_scaling()` folded into `q_pe`/`k_pe`); non-YaRN configs and
  Qwen/Mixtral untouched. **Re-run C5 (real DeepSeek-V2-Lite, MLA-2 disk tier, ~4 GB
  RAM) vs a one-layer-at-a-time HF f64 reference: `max_abs_diff 2.587e-5 < 0.5`,
  argmax exact 4/4, deterministic → PASS.** Result: **DeepSeek-V2-Lite —
  MoE-certified L3 (whole model)** = L1 (C1 real 1664 experts + C2 real top-6) + C4
  (topology) + **C5 (active-path-certified, real weights)**; manifest
  `ladder_level_whole_model: L3`. MLA-0 improved to `5.306e-5`; disk==RAM still
  bit-identical. **Not dense ADR-004 `CERTIFIED`; L4 (global F64) reserved/unreachable.**
  See `docs/HANDOFF_MLA_3.md` + `docs/HANDOFF_MLA1_C5_ROOT_CAUSE.md`.
- **MOE-PERF-5-REAL-MEASURE — real telemetry baseline before PERF-4 (measure only).** Used the
  PERF-5 instrumentation to capture real cache/prefetch/tier telemetry on the certified runtime
  path. **No code in `src/`** (test-only `#[ignore]` sweep). **Workloads:** Mixtral-87GB real =
  **too heavy/skip** (host ~12 GB free of 32; cache=1 forward peaks ~29 GB, default ~90 GB→OOM),
  DeepSeek-V2-Lite real = skip (load spikes; MLA uncached → no cache telemetry anyway), Qwen real =
  N/A (no whole-model path); **scale fixtures on the disk tier = safe** (exact `forward_cached` path
  at real top-k). **Findings:** cache size is the **dominant lever** — `auto→cache=1` forces
  re-reads (Mixtral 6→**20 misses/18 evict**, Qwen 19→**40/38**, hit-rate→0): the PERF-1
  cache↔OOM tension, telemetry-confirmed. Prefetch is observable and scales with top-k
  (`parallel_prefetches`=misses, `overlap_saved_ms` Mixtral≈0.4 < Qwen≈1.9 ms). Graph families are
  **I/O-bound** on disk; MoE compute trivial (DeepSeek 4354 tok/s RAM). **DeepSeek/MLA streams
  experts uncached** (`cache?=NO`; tier I/O only via timing gap 0.38→14.7 ms) — coverage limitation,
  not a zero. (Absolute resolve_ms confounded by OS page-cache warming; analysis rests on the
  deterministic structural counters.) **PERF-4 readiness: YES, remains next** — qint8 cuts expert
  bytes ¼ → ~4× smaller reads **and** ~4× more experts fit the RAM budget (relieves the cache
  thrash; turns Mixtral's forced cache=1 into a viable cache≥4, ~22 GB vs 90 GB); compounds with
  prefetch. **Highest risk: the qint8 numeric certificate** (per-expert int8 must clear the ADR-004
  gate). Nothing precedes PERF-4. See `docs/HANDOFF_MOE_PERF_5_REAL_MEASURE.md`. **Do not start
  PERF-4.**
- **MOE-PERF-5 — MoE-generation telemetry (observability parity with dense; instrumentation
  only).** The dense path reported load/total/tok-s; MoE-generate returned only token ids. PERF-5
  adds `MoeGenTelemetry` (load / prefill / decode / first-token / total / **tok-s**; expert-cache
  **hits/misses/evictions/resident_bytes**; prefetch **parallel/overlap_saved/resolve**; tier
  **materialized_bytes/reads**) via `MoeRuntime::generate_instrumented` +
  `controlled_moe_generate_instrumented`, surfaced behind **opt-in `ATENIA_MOE_TELEMETRY=1`**
  (default output byte-for-byte unchanged). **No optimization/numerics/routing/MLA/cache/loader/
  cert/manifest/ADR change.** New `src/moe/telemetry.rs`; `aggregate_resident_cache_stats` extended
  (+evictions/parallel_prefetches/prefetch_wall) + `aggregate_resident_cache_resident_bytes`;
  `*_timed` generate variants (existing fns are thin wrappers ⇒ **bit-identical** generation);
  snapshot-diff isolates one generation from the cumulative registry. Coverage: **timing = all
  families**; **cache/prefetch/tier = graph families (Mixtral/Qwen) on the disk tier** (RAM tier =
  true zero); **DeepSeek/MLA streams experts uncached ⇒ timing only**, flagged by
  `cache_telemetry_available`. Validation: `moe::telemetry` 4/4 + `moe_perf5_telemetry_test` 3/3
  (instrumented == `generate`, deterministic) + disk-tier demo shows cache/prefetch/tier metrics;
  **scale-cert 3/3** (no regression); **`cargo test --lib` 886/0**. Unblocks measuring PERF-3/PERF-4
  on real certified runs (the PERF-3-VALIDATION blocker). See `docs/HANDOFF_MOE_PERF_5.md`. **Do
  not start PERF-4.**
- **MOE-PERF-3-VALIDATION — how much of the prefetch win survives on certified workloads
  (measure only).** Measured prefetch at each certified family's **real top-k** on the certified
  disk-tier `forward_cached` path (Mixtral top-2 / Qwen-MoE top-4 / DeepSeek-V2-Lite top-6,
  cap=1); the 87 GB Mixtral re-run was excluded as the documented OOM hazard, so the surrogate
  is the **exact certified runtime** at real fan-out (rationale documented). **No
  runtime/numerics/routing/MLA/cache/cert/manifest/ADR change; test-only harness extension.**
  Robust metrics: **misses identical OFF/ON** (29/58/89 — order, not count); **overlap_saved
  rose monotonically** ~9.5 → 36.7 → 69.5 ms (**40 % → 69 % → 78 %** of read latency hidden,
  ∝ top-k); decode always >1.6× (noisy). Verdict: **Mixtral = MAJOR win** (only RAM-starved,
  read-bound, widest-expert family; cap=1 *forced*), **DeepSeek-V2-Lite = MODERATE** (highest
  overlap fraction but ~4 GB tier mostly RAM-resident ⇒ few real misses; MLA orthogonal),
  **Qwen-MoE = UNMEASURED** (block-level cert, no disk-tier whole-model forward). **Roadmap
  reorder:** the validation **could not measure a real certified run** (MoE-generate lacks the
  dense path's telemetry) ⇒ **promote PERF-5 (instrumentation parity) ahead of PERF-4**; PERF-4
  (qint8 default tier) stays high ROI but second (attacks bytes/root vs PERF-3's latency/symptom;
  complementary). 2 bench tests pass (`#[ignore]`, 4 reps); `src/` untouched ⇒ lib unaffected.
  See `docs/HANDOFF_MOE_PERF_3_VALIDATION.md`. **Do not start PERF-4.**
- **MOE-PERF-3 (prefetch) — async (parallel) expert prefetch for the disk tier.** Overlaps a
  token's selected-expert NVMe reads under the existing rayon pool (no async runtime, no
  uncontrolled threads), the lever PERF-2-VALIDATION identified for the `cap=1` re-read
  latency. **Opt-in `ATENIA_MOE_PREFETCH=1` (default off); no numerics/routing/MLA/attention/
  generation/cert/manifest/ADR change.** `forward_cached` batch-resolves the selected experts
  in parallel before the FFN loop (`resolve_selected`); bit-exact + deterministic (same
  experts, same combine order), bounded (≤top_k), works at cap=1, safe fallback. **Measured
  ~2× faster decode at cap=1, top-6** (191.74→95.32 ms; 76 ms of read latency overlapped:
  `wall read` 22.4 ms vs `Σ read` 98.8 ms); misses unchanged (order, not count). Gain scales
  with read latency + top-k. 2 new residency tests (20 pass); **scale-cert 3/3 both default
  AND with `ATENIA_MOE_PREFETCH=1`** (no cert regression); full lib green. Runtime auto-picks
  it up (`ExpertCache::new` reads the env; productive path already uses `forward_cached`).
  (Note: stale `HANDOFF_MOE_PERF_3.md` is an unrelated VRAM attempt — this handoff is
  `HANDOFF_MOE_PERF_3_PREFETCH.md`.) **Do not start PERF-4.**
- **MOE-PERF-2-VALIDATION — real impact measurement (measure only).** Quantified PERF-2 with
  existing instrumentation (`CacheStats`), **no runtime/numerics/cert change**. Measured cache
  mechanism (32-expert disk-tier layer, 24-token decode): **bf16 halves resident at every
  capacity, miss count identical for a given capacity** (cap=1: 48 vs 48); a **larger cache
  cuts misses** 48→23 (**−52%** at cap=all). Conclusion = **case B: PERF-2 reduced RAM (2×) +
  removed the ~90 GB OOM, but on a 32 GB host the auto-size picks cap=1 so rematerializations
  (and thus the 402.7 s Mixtral forward) are unchanged here**; the runtime win needs cap>1,
  which bf16 makes affordable (Mixtral cap=4: 90 GB→45 GB) on ≥~50 GB-free hosts. The 87 GB
  real forward was not re-run (heavy; conclusion determinable without it). Roadmap: **keep
  PERF-3 (prefetch/async)** — the only lever that helps the forward at cap=1 (hides NVMe read
  latency under compute), now better justified. New `#[ignore]` measurement harness
  `tests/moe_perf2_cache_validation.rs`; `cargo test --lib` unchanged. **Do not start PERF-3.**
  See `docs/HANDOFF_MOE_PERF_2_VALIDATION.md`.
- **MOE-PERF-2 (expert cache) — auto-sized cache + bf16-resident experts.** Implemented the
  highest-ROI item from the PERF-1 audit, **numerics/cert/manifest/ADR/routing/MLA/attention/
  generation all unchanged**. **PERF-2A:** per-layer expert-cache capacity is **auto-sized** to
  a RAM budget (default 50% available; `ATENIA_MOE_CACHE_RAM_FRACTION`) instead of the fixed
  `2·top_k` that OOM'd Mixtral (~90 GB); explicit `ATENIA_MOE_EXPERT_CACHE` override still wins.
  **PERF-2B:** cached experts stored **bf16 when lossless** (low-16-bits-zero, the bf16-tier
  case), else f32 — decode-on-hit is **bit-exact** (`bf16_truncate_lossless` round-trip
  identity), opt-out `ATENIA_MOE_CACHE_BF16=0`. Measured: **2× cache-RAM reduction** (exact:
  `resident_bytes_f32_equiv == 2×resident_bytes`); Mixtral per-expert 704 MB→**352 MB**, so
  cap=4 = 90 GB→**45 GB** and auto-size picks **cap=1 on a 32 GB host with no OOM / no manual
  tuning**. Output bit-exact + deterministic at any capacity. 4 new residency tests (18 pass);
  scale-cert (Mixtral/Qwen/DeepSeek) **3/3 unchanged**; full lib suite green. (Note: a stale
  `HANDOFF_MOE_PERF_2.md` documents an unrelated earlier GPU-FFN attempt — this milestone's
  handoff is `HANDOFF_MOE_PERF_2_EXPERT_CACHE.md`.) **Do not start PERF-3.**
- **MOE-PERF-1 — MoE performance audit + optimization roadmap (measurement only).** Measured
  where MoE time goes; **no runtime/numerics/cert/manifest/ADR change** (only a test-only
  `#[ignore]` timing bench, `tests/moe_perf_scale_bench.rs`, that *calls* the runtime).
  **Fresh** scale-fixture compute (reduced dim, RAM backend): load 8–19 ms, prefill 0.4–3 ms,
  219–3397 tok/s → the routing/attention/expert/MLA **compute is trivial; the bottleneck is
  weight I/O at real scale**. **Captured** real-weight numbers (from the L3 cert runs): Mixtral
  cold tier build ≈88 GB, warm reconstruct load **4.5 s**, forward (seq=4, bounded cache)
  **402.7 s**, default cache ≈90 GB→OOM; DeepSeek-V2-Lite disk tier ≈4 GB RAM. Bottleneck rank:
  (1) cold tier build I/O, (2) **expert re-materialization / cache thrash** (the recurring
  cost), (3) F32 expert decode (2× bytes), (4) warm backend reads, (5) AV scan, (6) compute
  (NOT a bottleneck). Roadmap (ROI-ordered): **PERF-2** auto-sized + bf16-resident expert cache
  (highest ROI: kills cache=4 OOM *and* cache=1 thrash), **PERF-3** expert prefetch/async tier
  reads, **PERF-4** qint8 default tier (gated on a numeric cert), **PERF-5** MoE-generate
  instrumentation parity. MLA latent cache + GPU expert offload deferred (high risk). Full lib
  suite green. **Do not start PERF-2.** See `docs/MOE_PERF_AUDIT.md`.
- **MOE-PRODUCT-2 — opt-in DeepSeek-V2-Lite in the productive `generate`.** Enabled the
  opt-in productive routing of **DeepSeek-V2-Lite (MLA)** in `atenia generate` via
  `MoeSpecResolver` + the unchanged `MoeRuntime`: `arch_for_productive_routing(DeepSeekMoe)
  → DeepSeekV2Lite` and `decide_route` no longer defers DeepSeek, so a clean V2-Lite shape
  (MLA, `q_lora_rank=null`, no V3 router) routes **RunMoe** behind `ATENIA_ENABLE_MOE=1`
  (and **NeedsOptIn** without it). A new **unsupported-variant guard** flags the
  **DeepSeek-V3 routing marker** (`e_score_correction_bias`) as non-runnable (defence in
  depth; real V3/V2-236B already caught by the Q-LoRA `q_a_proj` guard) → **V3 stays
  non-runnable**. **Dense path intact; fail-loud default; no numerics/threshold/cert-manifest
  change; no Q-LoRA / latent cache / perf work.** Only the certified V2-Lite shape reaches
  the runtime (Q-LoRA + V3-routing checkpoints refused at `diagnose_moe`). New production +
  resolver + integration tests (DeepSeek-V2-Lite shape via the `deepseek_scale` fixture
  routes NeedsOptIn→RunMoe); full lib suite green. The ~15.7B real generate is heavy → not
  run in CI (EOS/tokenizer covered by the `deepseek_scale` generate→EOS cert). `MoE-certified
  L3 = active-path-certified, NOT dense ADR-004 CERTIFIED; L4 reserved/unreachable`. Next:
  **MOE-PERF-1** (MoE-generate throughput) + CLI UX. See `docs/HANDOFF_MOE_PRODUCT_2.md`.
- **MOE-PRODUCT-1 — opt-in MoE in the productive `generate` (resolver-backed).** `atenia
  generate <model>` now routes a recognised MoE checkpoint through the **declarative
  resolver bridge** (`MoeSpecResolver::route`): **dense passes through unchanged**; a MoE
  checkpoint **without** `ATENIA_ENABLE_MOE=1` **fails loud** (exit 2, family-aware message);
  a **runnable, productively-routable family (Mixtral / Qwen-MoE) with the opt-in** runs via
  the controlled `controlled_moe_generate`. **DeepSeek productive routing deferred** (refused)
  and **DeepSeek-V3 routing is non-runnable mechanism-only** — both refused. New
  `route`/`arch_for_productive_routing` map a diagnosed family → `MoeArch` → resolver
  (`resolve` equivalence guard + `runnable`), then apply the opt-in; behaviour-equivalent to
  the lower-level `decide_route` for the runnable set (asserted by test) but now flowing
  through the spec resolver. **No numerics/threshold/manifest/ADR-007 change; dense generate
  untouched; no new family support claimed.** `MoE-certified L3 is active-path-certified, NOT
  dense ADR-004 CERTIFIED; L4 reserved/unreachable.** 4 new resolver tests + integration test
  `moe_product_routing_test`; `moe_integrate_routing`/`moe_production` regressions + full lib
  suite green. Next: **MOE-PRODUCT-2** (DeepSeek-V2-Lite productive routing) + performance.
  See `docs/HANDOFF_MOE_PRODUCT_1.md`.
- **MOE-INTEGRATE-2 — opt-in resolver bridge (declarative spec → runtime).** Added
  `src/adapter_toolkit/moe_resolver.rs` (`MoeSpecResolver`): resolves a `MoeStructuralSpec`
  into a typed `ResolvedMoeRuntimePlan` (family, execution convention `Atenia`/
  `HuggingFaceQwen`, `RoutingPlan`, attention, expert layout, dense-first, YaRN, disk-tier
  hint, representative `V3RouterConfig`, manifest cert scope, **runnable** flag) and, **behind
  the opt-in** (`ATENIA_ENABLE_MOE=1`), delegates a runnable plan to the **unchanged**
  certified `MoeRuntime::load_from_dir`. **Handwritten certified paths remain default** — no
  numerics/threshold/manifest change; the productive `decide_route` is untouched. **No new
  family support claimed.** Resolves Mixtral / Qwen-MoE / DeepSeek-V2-Lite as runnable and
  **DeepSeek-V3 routing as mechanism-only / non-runnable** (fail-loud before disk).
  Equivalence guard rejects any spec whose renorm/shared convention diverges from the
  authoritative `MoeFamily::descriptor()`; the resolved convention matches
  `RealMoeLayer::resolve_convention` exactly. 10 new tests; `adapter_toolkit::` + `moe::` +
  full lib suite green. Next: **MOE-PRODUCT-1** (wire a resolver-selected plan into the
  productive `generate`, opt-in + CLI). See `docs/HANDOFF_MOE_INTEGRATE_2.md`.
- **MOE-ATK-DECL-1 — declarative MoE family structural spec (Adapter Toolkit).** Added a
  **declarative spec layer parallel to the handwritten paths** (`src/adapter_toolkit/
  moe_family_spec.rs`): a typed `MoeStructuralSpec` over every structural axis (expert
  layout, router naming, shared expert + gating, routing scheme incl. **V3 noaux/
  group-limited**, renorm, attention `MHA/GQA/MLA`, qkv bias, YaRN, dense-first, disk-tier)
  with `preset`s that **reproduce the certified families** — Mixtral, Qwen-MoE,
  DeepSeek-V2-Lite — plus the **DeepSeek-V3-like routing mechanism** (L0). **Describe +
  validate only — not replacing certified paths yet; no execution/loader/manifest change;
  no new family support claimed** (DeepSeek-V3 as a *model* stays unsupported /
  provisioning-blocked; `is_runnable_model()` = false for it). Equivalence tests assert
  each preset matches the authoritative runtime sources (`MoeFamily::descriptor()` renorm/
  shared, MLA-deepseek-only, and the V3 preset builds a valid `v3_router::V3RouterConfig`
  and routes) so the spec **cannot silently diverge**; fail-loud on inconsistent specs.
  Extends MOE-INTEGRATE-1 `moe_spec.rs` (Mixtral/Qwen YAML). 13 new tests; `cargo test
  --lib adapter_toolkit::` + full lib suite green; INTEGRATE-1 + dense ATK untouched. Next:
  **MOE-INTEGRATE-2** (resolver bridge → `MoeRuntime` + fail-loud lift). See
  `docs/HANDOFF_MOE_ATK_DECL_1.md`.
- **MOE-V3-ROUTE-1 — DeepSeek-V3-like routing mechanism → L0.** Implemented the modern
  router primitives (`src/moe/v3_router.rs`, isolated reference — no runtime/loader/CUDA/
  Adapter-Toolkit wiring): **sigmoid scoring + `e_score_correction_bias` selection +
  group-limited top-k (`n_group`/`topk_group`) + `routed_scaling_factor`**, matching
  `transformers` v5.6.2 `DeepseekV3MoE.route_tokens_to_experts` (bias used for **selection
  only**; combine weight = original sigmoid score, renormalised, × scale; fail-loud on any
  missing/invalid param or non-`sigmoid` scoring). Certified at **L0 (mechanism/topology
  only)** on a reduced-dim DeepSeek-V3 MoE block vs a HF **float64** reference
  (`tests/moe_v3_route_scale_cert_test.rs`): **router set-equality 6/6**, worst MoE-block
  `max_abs_diff` **3.891e-8** (< 1e-3), worst combine-weight diff **1.192e-7**, min
  selection margin 5.56e-2, deterministic. **No real V3 weights, no download; NOT L1/L2/L3;
  not dense ADR-004 `CERTIFIED`; L4 reserved/unreachable.** Out of scope (still pending for
  a real V3 forward): Q-LoRA q-path, FP8-in-MoE, MTP, V3.2 DSA. Real-weight V3 stays
  **provisioning-blocked** (no small V3-family checkpoint). 8 new router unit tests; scale-
  cert (Mixtral/Qwen/DeepSeek) + full lib suite green; the three L3 families untouched.
  See `docs/HANDOFF_MOE_V3_ROUTE_1.md`.
- **MIXTRAL-CERT-3 (C5 active-path) — Mixtral-8x7B-v0.1 → MoE-certified L3
  (active-path-certified).** Ran Atenia's **real full forward** of the real trained
  Mixtral-8x7B-v0.1 (embeddings + 32 layers of GQA attention + RoPE + top-2 MoE +
  lm_head, experts streamed from a persistent bf16 **disk expert-tier**) on the
  canonical input `[1,100,200,300]` and compared it **end-to-end** against an external
  **float64** reference computed **one decoder layer at a time** (HF attention in f64 +
  manual MoE one **active expert** at a time — the F64 form of C5 over the **active
  subgraph**, never the whole model in F64 → **not L4**). **C5 PASS:** worst
  `max_abs_diff` **3.185e-4** (position 0) `< 0.5` (ADR-004 bar, unchanged; ~1570×
  inside), **per-position argmax exact 4/4** `[422,327,160,327]`, **deterministic**
  (two forwards bit-identical). Load (warm tier reconstruct) 4.5 s; forward 402.7 s;
  test wall 955.74 s. → **Mixtral-8x7B-v0.1: MoE-certified L3 (active-path-certified)**
  (`docs/numcert/mixtral-8x7b-v0.1.moecert.json`, `ladder_level_whole_model: L3`).
  **No `src/` change** — a resumable F64 reference generator (per-layer atomic
  hidden-state checkpoints) + a `#[ignore]` harness that *calls* the runtime; ADR-004
  gate not lowered. Operational note: on a 32 GB host the forward must run with a
  **bounded expert cache** (`ATENIA_MOE_EXPERT_CACHE=1`, numerically identical) — the
  default per-layer cache of 4 reconstructed-F32 experts × 32 layers commits ~90 GB and
  OOMs; cache=1 peaks ~29 GB. **Not dense ADR-004 `CERTIFIED`; L4 (global F64 ~374 GB)
  reserved/unreachable.** See `docs/HANDOFF_MIXTRAL_CERT_C5.md`.
- **MIXTRAL-DATA-PROVISION + MIXTRAL-CERT-1 (C1+C2) — Mixtral-8x7B-v0.1 real weights
  + partial L1.** Provisioned the real Mixtral-8x7B-v0.1 (19 safetensors shards,
  **87 GB / 86.99 GiB**, BF16, index validated; serial per-shard verified fetch after
  a paused parallel download — `docs/MIXTRAL_PROVISIONED.md`). Then, reusing the
  Qwen/DeepSeek decomposition tooling, certified on the **real weights**: **C1**
  per-expert exhaustive over **all 32 layers × 8 experts = 256 experts**, global worst
  `max_abs_diff` **1.907e-6** (layer 8 / expert 1), 0 failures; **C2** top-2 router
  **set equality** on all 32 layers, 0 failures, min routing margin **0.011413**
  (layer 13). C3 (GQA attention) at the mechanism level (`mixtral_scale` 1.639e-7).
  **MIXTRAL-CERT-2 then folded C4** — the `mixtral_scale` Mixtral-8x7B-topology
  full-forward cert vs HF f64 = **1.639e-7** (8 experts / top-2 / GQA 4:1 / classic
  experts / no shared / renorm — the exact conventions; argmax exact, greedy→EOS,
  deterministic) → **L2 = L1 + C4**. → **Mixtral-8x7B-v0.1: MoE-certified L2 (whole
  model)** (`docs/numcert/mixtral-8x7b-v0.1.moecert.json`, `ladder_level_whole_model:
  L2`). **No `src/` change** — a resumable reference generator + a resumable
  `#[ignore]` harness (model on an HDD; per-layer atomic checkpoints survive the
  environment's ~60-min background reaping); C4 via the existing scale-cert test.
  C3+C4 mechanism caveat. **(Superseded by MIXTRAL-CERT-3 → C5 active-path PASS →
  MoE-certified L3, above.)** Not dense ADR-004 `CERTIFIED`; not L4.
  See `docs/HANDOFF_MIXTRAL_CERT_C1C2.md`.
- **FORMAT-INTAKE-1 — PyTorch `.bin` intake.** Closes the coverage audit's #2
  gap (otherwise-supported checkpoints unloadable purely because they ship as
  `pytorch_model.bin`). A new `src/v17/loader/pytorch_bin.rs` **transcodes** a
  single-file `torch.save` `.bin` (ZIP + restricted pickle) into an in-memory
  safetensors buffer consumed by the existing `SafetensorsReader::from_bytes` —
  so the weight mapper, **adapter layer**, transforms and tier planning are
  reused **unchanged**. Hand-rolled STORED-zip reader + a **restricted**
  unpickler (whitelist of `OrderedDict` / `_rebuild_tensor_v2` /
  `_rebuild_parameter` / `torch.{Float,Half,BFloat16}Storage` — any other
  global/opcode is a hard error, never executed). Detection slots **after**
  GGUF/safetensors (safetensors always preferred); **sharded `.bin` and every
  unsupported shape fail loud** (no silent fallback). Proven **byte-identical**
  to a torch-saved reference (CI round-trip) and **end-to-end identical greedy
  text** to safetensors on a real SmolLM2-135M (`.bin` ↔ safetensors). Limits:
  single-file, contiguous F32/F16/BF16 only. `atenia capabilities` lists `.bin`.
  See [HANDOFF_FORMAT_INTAKE_1.md](./HANDOFF_FORMAT_INTAKE_1.md).
- **FORMAT-INTAKE-2 — sharded PyTorch `.bin`.** Extends FORMAT-INTAKE-1 to
  multi-file `pytorch_model-0000k-of-000NN.bin` + `pytorch_model.bin.index.json`.
  Reuses the existing `ShardIndex` parser (the `.bin` index schema is identical
  to the safetensors one) and the FI-1 per-shard transcode: each shard is
  transcoded and **assembled into one in-memory safetensors buffer** that flows
  through the unchanged adapter/mapper/pipeline. **Fail-loud consistency** —
  missing shard, duplicate tensor across shards, weight_map ghost tensor,
  undeclared shard tensor, malformed weight_map all error (no silent fallback).
  Proven **byte-identical** to an assembled reference (CI) and **end-to-end
  identical greedy text** to safetensors on a real **2-shard** SmolLM2-135M. All
  single- and multi-file `.bin` checkpoints of supported families now load with
  no external conversion. See [HANDOFF_FORMAT_INTAKE_2.md](./HANDOFF_FORMAT_INTAKE_2.md).
- **FP8-SAFETENSORS-1 — FP8 safetensors read.** Reads `F8_E4M3` (e4m3fn) and
  `F8_E5M2` tensors by **decoding to F32 at read time** inside `SafetensorsReader`
  (a side buffer; the FP8 tensor surfaces as a plain F32 entry) — so the weight
  mapper, graph, kernels, tier planner and adapters are **unchanged** and never
  see FP8 (no Numeric Policy / CUDA touched; F32/F16/BF16 paths intact). Works
  for single-file, sharded, and `.bin`-transcoded safetensors. Decoders are
  **bit-identical to PyTorch's `fp8.to(float32)`** (CI fixture test), and a real
  all-FP8 SmolLM2-135M loads + generates coherent text end to end. Fail-loud on
  body-length mismatch / unsupported dtypes. `atenia capabilities` lists FP8.
  See [HANDOFF_FP8_SAFETENSORS_1.md](./HANDOFF_FP8_SAFETENSORS_1.md).
- **STREAMING-LOADER-1 — memory-mapped safetensors load.** `SafetensorsReader::open`
  now **memory-maps** the file (via `memmap2`) instead of `fs::read`-ing the whole
  thing into a heap `Vec`, behind an unchanged reader API (a private
  `Backing{Owned|Mapped}` enum derefs to `&[u8]`) — so the weight mapper, adapter
  layer, graph and tier planner are untouched, and single-file **and per-shard**
  safetensors both benefit (the sharded loader and the `.bin` transcode funnel
  through `open`). The mapped read-only pages are file-backed + reclaimable, so
  they leave committed RAM. **Benchmark (Qwen2.5-1.5B, 2945 MB single-file, warm):
  peak commit 12865 → 9483 MB (−3382 MB ≈ file size), peak working set 4461 →
  3790 MB, wall 16.7 → 16.6 s (no penalty)** — scales to tens of GB on large
  models. `ATENIA_DISABLE_MMAP=1` + automatic byte-identical `fs::read` fallback
  keep the old path; mmap vs owned proven identical (by name).
  See [HANDOFF_STREAMING_LOADER_1.md](./HANDOFF_STREAMING_LOADER_1.md).
- **Loaders.** Single-file and sharded HuggingFace safetensors
  (F32 / F16 / BF16 / **FP8 E4M3+E5M2**), **memory-mapped at load**; GGUF
  (F16 / Q8_0 / Q4_K_M / Q5_K / Q6_K); single-file **and sharded** PyTorch `.bin`
  (transcoded + assembled).
  BF16 parameter storage (50 % RAM saving),
  BF16 KV cache (default on), RAM↔NVMe spill with chunked streaming.
- **Adapter layer.** Llama / Qwen 2 / Qwen 3 / Mistral / Phi-3 / Gemma 2 /
  Gemma 3 (text) family logic lives in `src/model_adapters/`; the
  execution core is family-agnostic
  (Phases 11–15). The **config** boundary is closed: all family-specific
  config semantics — defaults, validation, `rope_scaling` parsing (including
  the GGUF path) — are adapter-owned via `ConfigPolicy`
  (`default_*` / `validate_config` / `parse_rope_scaling` /
  `apply_config_defaults`). `LlamaConfig` and `gguf_config.rs` are now
  structural / format parsers only. Phase 16 closed the symmetric
  **weight-mapping** boundary: the GGUF→HF tensor-name mapping is
  adapter-owned via `GgufNameMapper` (default = arch-agnostic common
  rules; Phi-3 / Gemma 2 override), so `pipeline::build_gguf_name_map`
  no longer branches on `arch`. The execution core is fully
  family-agnostic for both config and weight mapping.
- **Determinism.** Greedy generation is reproducible bit-exact (D67 fixture);
  the lib test suite (726 tests passing, 1 ignored) is green.
- **CI.** A minimal GitHub Actions workflow runs on push / PR to `main`
  with **two blocking jobs**: a `cuda-toolkit` job (mirrors the
  locally-validated environment; no GPU, device tests auto-skip) running
  `cargo test --lib -- --test-threads=1` + `cargo test --test
  tinyllama_config_test`, and a `cpu-only` job that **enforces** the
  declared vendor-agnostic invariant (ROADMAP: "the engine's core never
  assumes NVIDIA-specific behaviour"; ADR-003 "GPU as inference
  baseline") — the CUDA-less build links and the non-GPU test subset
  passes (CPU-1 + CPU-2; promoted from non-blocking to blocking in
  CPU-5). Heavy on-disk GGUF / F64 drift tests stay operator-run
  (`#[ignore]`).
- **Operability hardening (M12, complete & CI-green).** The engine now
  explains failures instead of hiding them. The M12.1–M12.5 series closed
  the diagnostics/error-surface gaps: CUDA root-cause & VRAM-probe failures
  propagate instead of being swallowed (M12.1); `atenia run` load failures
  are clean errors with exit 2, not panics (M12.2); a consolidated
  env/hardware diagnostics block + shared tier-plan summary print on the
  run path (M12.3); silent CPU/RAM fallbacks are now visible — once-per-run
  `cuda_matmul` warn, M6 residency failed-layer aggregate, `try_all_paths`
  reasons, `impl Display for LoaderError` + actionable CLI hints (M12.4);
  `cuInit` CUresult is checked and a JSON-render serialise failure exits
  ≠ 0 (M12.5). No control-flow / numeric / tiering changes; success path
  unchanged.
- **Vendor-agnostic CPU-only build (closed).** `cargo build --lib` links
  with no CUDA toolkit installed and the non-GPU test subset passes.
  Previously a tracked link-time debt (the Rust FFI declared the CUDA
  kernel static libs unconditionally), now resolved: CPU-1 (build.rs
  emits an auto-detected `atenia_cuda` cfg) + CPU-2 (every CUDA `extern`
  / `#[link]` block gated `#[cfg(atenia_cuda)]` with
  identical-signature `#[cfg(not(atenia_cuda))]` stubs;
  `cuda_available()` is `false` without the backend). The CUDA build is
  byte-identical (full lib suite, currently 726 tests + 1 ignored). Enforced by the
  now-blocking `cpu-only` CI job (CPU-5). Not a multi-vendor compute
  backend — see *Single vendor* below.
- **Phi-3.5 GGUF end-to-end (closed).** `Phi-3.5-mini-instruct`
  Q4_K_M GGUF loads and generates coherent text end-to-end. Added on
  the GGUF path: Q5_K block decode, Phi-3 LongRope read from the
  GGUF `rope_factors_{short,long}` tensors (not just metadata), and
  the fused-tensor name mapping for both `attn_qkv` -> `qkv_proj` and
  `ffn_up` -> `gate_up_proj`. The post-consolidation validation
  warning that surfaced this is closed; no tracked debt remains.
- **Gemma 2 GGUF end-to-end (closed).** `gemma-2-2b-it` Q4_K_M
  GGUF loads and generates text identical to the HF safetensors
  reference. Six never-validated Gemma 2 GGUF-path defects were
  fixed in isolated, regression-tested layers: a mis-named
  post-FFN norm tensor (GAP-N1), wrong GGUF config keys for
  head_dim / softcaps (GAP-C1/C2), the 4-norm `ffn_norm` ->
  `pre_feedforward_layernorm` mapping with extra-first adapter
  composition (GAP-N2), and the root cause — llama.cpp pre-folds
  the RMSNorm `+1` into the Gemma 2 GGUF weights (measured exactly
  +1.0 element-wise vs HF), so the Gemma 2 GGUF transform table is
  the HF table minus the norm `+1` fold (GAP-T1). No RopeUnpermute
  needed. TinyLlama / Phi-3.5 GGUF regressions unaffected; no
  tracked debt remains.
- **Adapter Toolkit v1 (closed).** ADR-006's microplan is fully
  delivered: AT-0 (the ADR), AT-2 (conformance harness, the
  executable freeze oracle), AT-1a/b/c (the declarative
  `FamilyTensorSpec` data plus golden A/B oracle, then the GGUF
  name maps and load transforms rewired onto it, behaviour-
  preserving), AT-3a (the two hand-copied GGUF completeness gates
  in `pipeline.rs` collapsed into one helper), and AT-3b (each
  adapter declares its required GGUF dtype coverage and a
  conformance test asserts decoder support — preventing the
  Phi-3.5 Q5_K class of runtime `UnsupportedDType` bug). Validated
  end-to-end against two real families (Phi-3.5 + Gemma 2 GGUF
  both identical to HF) without using the imperative escape
  hatch. AT-4 (YAML / JSON templates, scaffolding generator,
  public Adapter SDK) remains explicitly deferred — no
  serialized contract is frozen and no automatic / magic model
  support is promised. A new family still requires a graph
  builder, numeric validation and explicit review.
- **Qwen3 family supported (HF safetensors + GGUF).** `Qwen3-0.6B`,
  `Qwen3-4B`, and `Qwen3-8B` (Q4_K_M GGUF) load and generate
  coherent text on the dev box. Topology delta vs Llama: per-head
  QK-Norm RMSNorm applied to Q and K after reshape-to-heads and
  before RoPE (γ shape `[head_dim]`), plus explicit `head_dim`
  (128, ≠ `hidden_size / num_attention_heads`) threaded through
  the projection shapes (q/k/v/o use `n_heads * head_dim` instead
  of `hidden`), and `attention_bias = false` (no QKV biases,
  opposite of Qwen2's family default). The 1 / √head_dim
  attention scale lives in the K-Norm γ (a pre-normalize scale
  would be stripped by RMSNorm; a post-normalize γ scale
  survives). No new AMG ops introduced. **LM head**: `build_qwen3`
  honours `tie_word_embeddings` — the small variants (0.6B / 1.7B)
  ship a redundant physical `lm_head.weight` while the larger
  variants (4B / 8B / 14B / 32B) are genuinely tied and ship
  none; the tied branch reuses `embed_tokens` transposed and is
  correct for every variant. **GGUF**: `general.architecture =
  "qwen3"` is accepted; the QK-Norm γ tensors map from
  `blk.N.attn_{q,k}_norm.weight`; llama.cpp does not row-permute
  Qwen3 q/k, so the GGUF path reuses the HF transform table
  (no `LlamaRopeUnpermuteRows`).
- **Model family validation.** The per-family mastery batteries
  below are consolidated, with their per-checkpoint tables, real
  fixes and out-of-scope notes, in `docs/MODEL_FAMILY_VALIDATION.md`
  — functional / family validation, distinct from ADR-004 numeric
  certification.
- **Llama-family mastery battery.** End-to-end load + 1-turn
  chat generation on the dev box across the Llama-family scope.
  Prompt `"What is the capital of France?"`, greedy decoding,
  certified mode, no code changes — every case routed through
  the existing `LlamaFamilyAdapter` unchanged. **9 / 9 PASS**:
  TinyLlama 1.1B safetensors, Llama 3.2 1B safetensors,
  Llama 3.2 1B Q4_K_M GGUF, Llama 3.2 3B safetensors,
  Llama 3.1 8B safetensors, Llama 3.1 8B Q4_K_M GGUF,
  Llama 3.1 8B Q5_K_M GGUF, Llama 3.1 8B Q6_K GGUF, and
  DeepSeek-R1-Distill-Llama-8B Q4_K_M GGUF (chain-of-thought
  reasoning trace coherent, max-tokens 120). All emitted
  *"The capital of France is Paris."* and halted on `<|eot_id|>`
  for the Llama-3.x instruct cases (single-EOS path correct;
  `tokenizer_config.json::eos_token` resolves to id 128009).
  The 8B-class loads through tier-aware spill (~3.5 GiB VRAM
  + RAM/Disk tiers) at 0.07–0.10 tok/s — bounded by spill, not
  a correctness regression. Llama 2 7B Chat is gated on HF and
  was not downloaded; Llama 2 13B Chat coverage stays via
  `m5_dc_llama2_13b_coherence_test` (integration test, locked
  regression). No adapter / config / tokenizer / transform
  change was required to land this battery — the audit at
  Phase 2 surfaced no real gap for the family beyond what
  Phases 11–16 already closed. Logs in
  `target/validation/llama_mastery/`.
- **Qwen-family mastery battery.** End-to-end load + 1-turn
  chat generation across the Qwen2 + Qwen3 scope, greedy,
  certified mode. **11 / 11 PASS**: Qwen2.5 1.5B / 3B / 7B
  safetensors, Qwen2.5 7B Q4_K_M / Q5_K_M / Q6_K GGUF, Qwen3
  0.6B / 4B safetensors, Qwen3 8B Q4_K_M GGUF, Qwen2.5-Coder
  1.5B safetensors, and DeepSeek-R1-Distill-Qwen-7B Q4_K_M GGUF
  (coherent chain-of-thought). Unlike the Llama battery this one
  surfaced real gaps and required adapter/config fixes (no core
  change):
  (1) **Qwen3 LM-head tie** — `build_qwen3` always registered a
  separate `lm_head.weight`, which the loader left zero-filled
  for the genuinely tied 4B/8B/14B/32B variants → uniform logits
  → constant degenerate token. Fixed to honour
  `tie_word_embeddings` like `build_llama`.
  (2) **Qwen2/Qwen3 GGUF unsupported** — `general.architecture =
  "qwen2"/"qwen3"` was rejected by the GGUF config arch
  whitelist. Added both, with `qwen2`/`qwen3` metadata-key
  prefixes and `model_type` mapping.
  (3) **Qwen2 GGUF QKV biases** — `COMMON_NAME_TABLE` gained the
  `attn_{q,k,v}.bias` suffixes, and the hard-coded
  `attention_bias = Some(false)` in the GGUF config parser became
  `None` so the adapter default resolves it (Qwen2 → biases on).
  (4) **Qwen3 GGUF QK-Norm** — `QWEN3_SPEC.name_extra` maps
  `blk.N.attn_{q,k}_norm.weight`; the Qwen3 adapter's GGUF weight
  mapper now uses the QK-Norm-aware HF table (llama.cpp does not
  row-permute Qwen2/Qwen3 q/k, so no `LlamaRopeUnpermuteRows`).
  Logs in `target/validation/qwen_mastery/`. The 7–8B-class loads
  run through tier-aware spill at 0.07–0.12 tok/s — bounded by
  spill, not a regression.
- **Gemma-family mastery battery.** End-to-end load + 1-turn chat
  generation across the Gemma 2 + Gemma 3 (text) scope, greedy,
  certified mode. **8 / 8 in-scope PASS**: Gemma 2 2B safetensors,
  Gemma 2 9B safetensors, Gemma 2 9B Q4_K_M / Q5_K_M / Q6_K GGUF,
  Gemma 3 1B safetensors, Gemma 3 1B Q4_K_M GGUF, Gemma 3 4B
  Q4_K_M GGUF (text path) — all emitted a coherent
  *"...the capital of France is **Paris**."* and halted on the
  chat turn-terminator. Gemma 3 (`Gemma3ForCausalLM` /
  `model_type = gemma3_text`) is a **new supported family**: it is
  Gemma 2's topology (dual-norm, GeGLU, embedding scale, soft-cap
  *removed*) plus per-head QK-Norm on q/k and a dual RoPE base
  frequency (local sliding-window layers use `rope_local_base_freq`,
  global layers use `rope_theta`, selected by
  `sliding_window_pattern`). New `Gemma3Adapter` +
  `build_gemma3` builder; no new AMG ops. Gaps found and fixed
  (adapter / config / tokenizer / decoder — no core change):
  (1) **Gemma 3 unsupported** — added the config fields
  (`rope_local_base_freq`, `sliding_window_pattern`), the adapter,
  the builder, the `GEMMA3_SPEC` transform tables, and the GGUF
  `general.architecture = "gemma3"` route.
  (2) **Multi-EOS** — generation collapsed a model's
  `eos_token_id` array to its first element. A Gemma instruct turn
  ends with `<end_of_turn>`, not `<eos>`, so Gemma 3 ran past its
  natural stop into off-distribution garbage. `LlamaConfig` now
  keeps the full `eos_token_ids` set and the generator halts on
  any of them; the pipeline additionally resolves the standard
  chat turn-terminators (`<end_of_turn>` / `<|eot_id|>` /
  `<|im_end|>`) by name from the vocabulary — needed for the GGUF
  path, whose metadata carries only a scalar `eos_token_id`.
  (3) **Q5_0 decoder** — the Gemma 3 1B Q4_K_M GGUF mixes Q5_0
  into its attention tensors (standard llama.cpp quant recipe for
  small models); added the `decode_q5_0` decoder (legacy ggml
  `block_q5_0`, 32-element 22-byte blocks).
  **Out of scope:** Gemma 3 4B *safetensors* is the multimodal
  `Gemma3ForConditionalGeneration` wrapper (vision tower, text
  config nested under `text_config`) — classified out of scope per
  the text-only focus; its text-only Q4_K_M GGUF is validated
  instead. Logs in `target/validation/gemma_mastery/`.
- **Phi-family mastery battery.** End-to-end load + 1-turn chat
  generation across the Phi-3 / Phi-3.5 / Phi-4 scope, greedy,
  certified mode. **8 / 8 PASS**: Phi-3.5-mini safetensors,
  Phi-3.5-mini Q4_K_M / Q5_K_M / Q6_K GGUF, Phi-3-mini-4k
  safetensors, Phi-3-mini-128k safetensors, Phi-4-mini
  safetensors, Phi-4-mini Q4_K_M GGUF — all generated a coherent
  *"The capital of France is Paris…"* continuation. Phi-3 / 3.5
  were already supported (fused QKV + fused gate_up + LongRope);
  this battery extended the existing `Phi3Adapter` / `build_phi3`
  to the rest of the family. Gaps found and fixed (builder /
  config — no core change):
  (1) **Plain-RoPE Phi-3** — `build_phi3` panicked unless the
  config carried a LongRope block, so Phi-3-mini-4k (4k context,
  no `rope_scaling`) crashed on load. Plain RoPE is now
  represented as a unit-factor `LongRope` (`original == max`
  position embeddings, all-1.0 factors) which degenerates exactly
  to standard RoPE — one builder code path, no separate branch.
  (2) **Phi-4 partial rotary** — Phi-4-mini sets
  `partial_rotary_factor = 0.75`: RoPE rotates only the first
  `round(0.75 · head_dim)` dims of q / k, the tail passes through
  un-rotated. Added `LlamaConfig::partial_rotary_factor` +
  `rotary_dim()`; the builder slices q / k into rotary / pass
  halves, rotates the first, and concatenates — built from
  existing `SliceLastDim` / `Concat` / `RoPE` nodes. A Phi-4 GGUF
  carries the rotated count as `phi3.rope.dimension_count`, which
  the GGUF config parser converts back to the fraction.
  (3) **Phi-4 grouped-query attention** — Phi-4-mini is GQA
  (24 q / 8 kv) where Phi-3 / 3.5 are MHA. Phi-3's fused
  `qkv_proj` is sliced into Q/K/V activations (not tiled at load
  like the standalone Llama K/V weight), so the builder now
  repeats each KV head `kv_groups` times in the graph
  (interleaved, matching HF `repeat_kv`) before the attention
  matmul — no-op for the MHA variants. Logs in
  `target/validation/phi_mastery/`. The 3.8B-class loads run
  through tier-aware spill at ~0.2–0.3 tok/s.
- **Mistral-dense mastery battery.** End-to-end load + 1-turn
  generation across the Mistral 7B dense scope, greedy, certified
  mode. **7 / 7 PASS**: Mistral-7B-v0.3 base safetensors,
  Mistral-7B-Instruct-v0.3 safetensors, Mistral-7B-Instruct-v0.3
  Q4_K_M / Q5_K_M / Q6_K GGUF, Mistral-7B-Instruct-v0.2
  safetensors, Mistral-7B-Instruct-v0.2 Q4_K_M GGUF — all
  generated a coherent *"…the capital of France is Paris…"*
  continuation (the base model via `--no-chat-template`).
  **No code change** — the audit confirmed Mistral dense is pure
  Llama topology (GQA 32/8, SwiGLU, RMSNorm, RoPE, no QKV bias,
  untied LM head); `MistralAdapter` already delegates graph build
  and weight mapping to the Llama path, and Mistral GGUF exports
  under `general.architecture = "llama"` so it loads through the
  Llama-family adapter unchanged. v0.3's dropped sliding-window
  and v0.2's `sliding_window = 4096` are both below the smoke
  context length, where sliding-window attention is equivalent to
  full causal — no divergence observed. Mixtral / Mistral-MoE
  remain explicitly out of scope (no MoE path was added or
  exercised). Logs in `target/validation/mistral_mastery/`.
- **SmolLM-family mastery battery.** End-to-end load + 1-turn
  generation across the SmolLM / SmolLM2 scope, greedy, certified
  mode. **9 / 9 PASS** (+ Falcon-3 7B regression): SmolLM2
  135M / 360M / 1.7B-Instruct safetensors, SmolLM2-1.7B-Instruct
  Q4_K_M / Q5_K_M / Q6_K GGUF, SmolLM 135M / 360M / 1.7B-Instruct
  safetensors, and Falcon-3-7B-Instruct safetensors — all loaded
  and generated coherent text (the 1.7B variants emitted
  *"The capital of France is Paris."* and halted on EOS; the
  135M models are verbose but coherent; SmolLM2-360M produced a
  well-formed refusal — a small-model behaviour, not an engine
  defect). **No code change** — SmolLM / SmolLM2 are built on the
  Llama architecture (`architectures = ["LlamaForCausalLM"]`,
  `model_type = "llama"`, MHA, RMSNorm, RoPE, SwiGLU, tied LM
  head), so they resolve to `LlamaFamilyAdapter` directly; there
  is no separate SmolLM adapter and none is needed. Falcon-3 is
  likewise `LlamaForCausalLM` (GQA, explicit `head_dim`) and rides
  the same path. SmolLM2-1.7B is one of the four ADR-004 F64
  fixture models, so its numeric certification was already
  locked. Logs in `target/validation/smollm_mastery/`.
- **Falcon-family mastery battery.** End-to-end load + 1-turn
  generation across the Falcon3 scope, greedy, certified mode.
  **6 / 6 PASS**: Falcon3-1B-Instruct, Falcon3-3B-Instruct,
  Falcon3-7B-Instruct safetensors, and Falcon3-7B-Instruct
  Q4_K_M / Q5_K_M / Q6_K GGUF — all emitted *"The capital of
  France is Paris."* and halted on EOS. Falcon3 is pure Llama
  topology (`architectures = ["LlamaForCausalLM"]`,
  `model_type = "llama"`, GQA, explicit `head_dim = 256`, RMSNorm,
  RoPE `theta = 1000042`, SwiGLU, untied LM head), so it resolves
  to `LlamaFamilyAdapter` directly and Falcon3 GGUF exports under
  `general.architecture = "llama"`. **One config-layer fix:**
  Falcon3-1B/3B-Instruct omit `bos_token_id` from `config.json`
  (it lives only in `generation_config.json`); the parser now
  falls back to `eos_token_id` when the field is absent — a
  generalizable tolerance fix, `LlamaConfig.bos_token_id` is not
  consumed by the generation path (the tokenizer owns BOS). No
  core / adapter / graph change. **Classic Falcon
  (`FalconForCausalLM` / `RWForCausalLM`) is out of scope:** it
  uses LayerNorm (not RMSNorm), parallel attention, and
  multi-query fused QKV — a distinct architecture that would
  require a new graph builder and new AMG nodes. Both classic
  paths fail loud cleanly (safetensors: missing
  `num_key_value_heads`; GGUF: unsupported
  `general.architecture = "falcon"`). Verdict: **Falcon3 dominada;
  Falcon clásico fuera de scope actual.** Logs in
  `target/validation/falcon_mastery/`.
- **Adapter Toolkit v2 (complete).** A declarative layer on top of
  the v1 model-adapter system, in `src/adapter_toolkit/`. A model
  is described by a YAML/JSON DSL instead of a hand-written Rust
  adapter; the toolkit parses it (`dsl`), resolves it to an IR
  (`spec` — `ResolvedAdapterSpec` + the normalised feature
  catalog), and generates a `GeneratedAdapter` that implements the
  v1 `AteniaModelAdapter` supertrait by **pure delegation** to the
  hand-written v1 adapter for the family. The `AdapterRegistry` is
  dynamic and v2-first / v1-fallback. **No core, graph-builder or
  v1-adapter code was modified; no Rust is generated dynamically.**
  Declarative validators (`validate`) fail loud on inconsistent
  specs (`gqa` without `kv_heads`, `fused_qkv` without
  `split_strategy`, contradictory `partial_rotary_factor`, unknown
  family/architecture). Three CLI subcommands — `atenia load`
  (parse + validate + build adapter; never runs generation),
  `atenia debug` (verbose + warnings), `atenia inspect` (auto-detect
  a `config.json` / `*.gguf` model dir and emit a loadable YAML).
  Five shipped examples under `config/adapters/`. **475 / 475**
  `cargo test --lib` (411 v1 baseline + 64 toolkit), zero
  regressions; `inspect → load` round-trips for TinyLlama, Llama
  3.2, Qwen 2.5, Gemma 2, Phi-3.5; v1 `atenia generate` unchanged.
  A post-completion technical-debt audit hardened three points:
  GGUF inspection cannot recover the RoPE variant (llama.cpp folds
  it into the `rope_factors` tensors) so it now emits an explicit
  note instead of guessing; the DSL `config`/`weights`/`attention`
  sections are documented and labelled as *declarative,
  validated-not-applied* constraints (`config.json` stays
  authoritative); an explicit unrecognised `model_type` now fails
  loud instead of being silently swapped. `serde_yaml 0.9` is
  deprecated upstream — accepted, contained (DSL front-end only,
  off the hot path), with a migration TODO.
- **Command-line interface (CLI-0 → CLI-7, complete).** A
  product-grade CLI built as a frontier layer in `src/cli/` — no
  runtime-core, loader or graph-builder change. **CLI-1** — a
  structured error system: stable `E-*` codes, a human
  *what-happened / how-to-fix* format, and a unified exit-code
  scheme (`0` success, `1` system/IO, `2` user input, `3` runtime,
  `101` internal panic), plus a panic boundary that renders a
  caught panic instead of dumping a backtrace. **CLI-2** — a
  logging layer with levels (`--quiet` / `--verbose` / `--debug` /
  `--trace` / `--log-level`), an optional `--log-file`, and a
  per-run `--trace-id`; stdout stays reserved for command results,
  stderr for logs. **CLI-3** — diagnostics: `atenia doctor` (host
  CPU/RAM/CUDA/build), `atenia diagnose --model` (pre-flight model
  check, no generation), `atenia capabilities` (supported families,
  formats, quants), all with `--json`. **CLI-4/5** — `atenia chat`,
  an interactive multi-turn REPL using the model chat template,
  with `/help` `/history` `/reset` `/clear` `/exit`, lazy pipeline
  load, and a streamed token-by-token response. **CLI-6** —
  `atenia download`, a curated downloader (three public, non-gated
  Hugging Face checkpoints: `smollm2-135m`, `tinyllama`,
  `qwen2.5-0.5b`) so first-time users go from `cargo install` to a
  running chat without learning `huggingface-cli`. Sequential
  fetch over `ureq` + rustls with `.partial` + atomic rename and a
  single retry; no resume, no checksum, no arbitrary repo support.
  **CLI-7** — `atenia quickstart`, a first-run UX that prints the
  recommended `doctor` → `download` → `diagnose` → `chat` flow with
  the exact commands; `--download` runs step 2 by reusing the CLI-6
  downloader (no duplication). Plan mode is fully non-interactive
  and scriptable. **517 / 517** `cargo test --lib` at CLI-7 close (the
  lib suite is now **628** with the experimental AQS subsystem); the CLI
  surface is covered by seven integration suites (`cli_errors`,
  `cli_logging`, `cli_diagnostics`, `cli_chat`, `cli_ux`,
  `cli_download`, `cli_quickstart`). Full reference: `docs/CLI.md`.
- **Local validation battery (post-Adapter-Toolkit-v1).** A
  load-and-generate sweep over the 18 checkpoints currently
  present under `models/` on the dev box (RTX 4070 Laptop, 8 GB
  VRAM, default certified mode, 12-token greedy continuation of
  `"The capital of France is "`): **17 PASS / 1 WARN / 0 FAIL**.
  The mandated regressions (TinyLlama Q4_K_M GGUF, Phi-3.5-mini
  Q4_K_M GGUF, gemma-2-2b-it Q4_K_M GGUF) all loaded and
  generated coherent text identical to their HF safetensors
  references. The only WARN was `gpt2-safetensors`, whose
  directory is missing `config.json` (incomplete checkpoint, not
  a load-path defect; GPT-2 is **not** a supported family — no
  GPT-2 adapter is registered). No new failure class surfaced;
  no regression in any previously-validated family.

## Opt-in / experimental (documented profile, not default)

These work and ship, but **off by default** with a known, documented numeric
profile. Operators opt in and accept the profile.

- **Fast mode** (`ATENIA_FAST_MODE=1`) — BF16-Tensor-Core native execution.
  Does **not** satisfy ADR-004 strict on every model by construction; the
  per-checkpoint envelope is documented in
  [ADR-005](./decisions/ADR-005-fast-mode-bf16-tc-envelope.md) (SmolLM2 1.7B
  is the worst-case sentinel of the M4.6 family at 2.33 drift).
- **INT8 W8A16** (`ATENIA_M9_INT8=1`) — measurable speedup
  (−9 % to −14 % on 13B) but misses ADR-004's `< 0.5` gate on all four
  fixture models. Ships **for the drift reason, not the performance reason**.
- **GGUF quantized models** — certified under the **functional** schema
  v2.0.0 (smoke-based, documented drift 0.0–10.19), *not* ADR-004 strict.
  Q4_K_M is aggressive 4-bit quantisation; the drift is intrinsic to the
  format, not an Atenia defect.
- **AQS — Atenia Quantization Search** (AQS-0 → AQS-10, complete). An
  isolated, **CPU-only, opt-in, experimental** research subsystem:
  `QuantizationPolicy` → drift evaluator → end-to-end harness →
  certification report → `3.0.0-draft` manifest → search engine → runner →
  `atenia search` CLI. It never runs in the default path, adds no
  dependency, and produces **draft** output only — it is **not** production
  certification (that remains ADR-004 / ADR-005). On TinyLlama only BF16 is
  ADR-004-certified; AWQ (α=0.25) is the best *useful-lossy* option; GPTQ
  (surrogate and real) did not beat the weight-only plateau. ~93 of the 628
  lib tests cover AQS. Full write-up: [AQS_OVERVIEW.md](./AQS_OVERVIEW.md).
- **MoE — Mixture-of-Experts experimental track** (MOE-0 → MOE-18, closed
  MOE-19). An isolated, **CPU-only, opt-in, experimental** compute + data plane
  in `src/moe/`: detect + fail-loud, classic and packed/fused expert binding,
  sparse execution, graph ops, real layer/stack assembly, validation + smoke
  harnesses, numerical metrics, and HF convention parity with automatic
  selection. Three tiny **real** checkpoints (Qwen1.5-MoE, Qwen2-MoE, Mixtral)
  run end-to-end; numerical parity with HuggingFace is ~1e-10 on the layer-0
  MoE block. The **productive loader still fails loud** on MoE checkpoints —
  MoE is **not** wired into the loader/runtime/Adapter Toolkit/CLI and **no MoE
  family is production-supported**. Full write-up:
  [MOE_OVERVIEW.md](./MOE_OVERVIEW.md).
- **MoE — full transformer + controlled runtime** (MOE-FULL-1 → MOE-FULL-15).
  Building on the experimental track: a full tiny MoE transformer (embeddings →
  attention/RoPE/causal mask → MoE block → lm_head), KV cache + greedy decode,
  expert residency (RAM/NVMe tiers) + LRU cache, GQA, and a **controlled, opt-in
  production path** (`moe::controlled_moe_generate` + `atenia moe-generate`,
  gated by a MoE certification manifest + `ATENIA_ENABLE_MOE=1`). **Three families
  certified vs HF f64 at three levels** — tiny end-to-end (Mixtral 7.451e-08,
  Qwen-MoE 5.960e-08, DeepSeek-MoE MLA attn 9.999e-06), **real-checkpoint** layer-0
  MoE block (~1e-10..1e-11), and **scale topology** (Mixtral-8x7B topology
  1.639e-07, Qwen 16-expert 1.490e-07, DeepSeek 16-routed 7.806e-03; argmax/greedy
  exact). The **dense loader still fails loud**; MoE runs **only** behind the
  explicit opt-in on certified families; unsupported variants (Qwen3 QK-norm,
  DeepSeek Q-LoRA) are refused clearly. **MoE remains experimental** — the
  multi-GB real weights are not certified, and there is no tokenizer text CLI nor
  a dense fail-loud lift. Matrix + verdict: [HANDOFF_MOE_FULL_15.md](./HANDOFF_MOE_FULL_15.md).

## Scaffolding / known limitations

Honest about the rough edges. None of these block the execution-layer thesis,
but they bound what you should rely on.

- **Conservative GPU eligibility.** The tier planner only marks a tensor
  GPU-eligible when `rank >= 2 && name.ends_with("_proj.weight")`. Embeddings,
  tied LM-head inputs, norms, biases, and masks stay in RAM even when VRAM has
  room — because many executor nodes are still CPU-only and call `ensure_cpu`
  before reading. This is deliberate (correctness over utilisation), so GPU SM
  utilisation is low on CLI generation today. Pass 1 of the
  [resolution plan](./ATENIA_RESOLUTION_PLAN.md) addresses the
  *observability* of this first.
- **CPU-only executor nodes.** `RmsNorm`, `Reshape`, `Permute`, `Transpose2D`,
  `IndexSelect`, `Softmax`, `SoftCap`, `BroadcastAdd/Mul`, `Concat`, and
  activations run on CPU. Projection matmuls can hit CUDA but outputs usually
  return to CPU between ops. The tied LM-head path transposes
  `embed_tokens.weight` every forward (CPU-heavy for large-vocab models) — a
  known follow-up.
- **Certified BF16→F32 VRAM slow path (no longer reproduces; not formally
  recertified).** Previously documented as a hard failure
  (`BF16->VRAM slow-path upload failed`) on Mistral 7B and Falcon 3 7B in
  certified/manifest mode. The post-Adapter-Toolkit-v1 local validation
  battery exercised the default certified path on the dev hardware (RTX 4070
  Laptop, 8 GB VRAM) and both `mistral-7b-v0.3` and `falcon3-7b-instruct`
  safetensors loaded and generated coherent text end-to-end. The original
  CUDA error path was not reproduced. This is a single-host observation —
  no F64 numcert was added for these two checkpoints, and "all Mistral /
  all Falcon variants" is not claimed; the underlying helper still returns
  `None` on the swallowed-error path, so a future regression is possible.
  The limitation is downgraded from "fails" to "previously observed,
  currently dormant" pending broader coverage.
- **Decode graph rebuilt per token.** Generation rebuilds a fresh decode graph
  every token (graph rebuild is <1 ms per D68, so this is not the bottleneck,
  but it shapes the GPU-utilisation profile).
- **No production training.** Autograd exists for *graph correctness*, not
  training. No training loops, no optimisers in the runtime. Training is v25+.
- **Single vendor.** NVIDIA CUDA only (sm_70+, Linux + Windows). Intel iGPU
  (v22), AMD ROCm (v23), Apple Metal (v24) are roadmap, not shipped. The core
  never assumes NVIDIA-specific behaviour, and the CUDA-less build is now
  enforced in CI — but multi-vendor execution itself is still not built (a
  CUDA-less binary links and runs the non-GPU surface; it does not provide
  an alternative compute backend).
- **Production hardening — CLI slice done (CLI-0 → CLI-7), engine-internal
  observability pending (v21).** The M12 series plus the CLI-0 → CLI-7 phases
  closed the user-facing error / logging / diagnostics slice: failures
  propagate with a stable `E-*` code and an exit code, logging has explicit
  levels and an optional log file, and `doctor` / `diagnose` report host and
  model state (see the *Command-line interface* entry above). Still pending
  for v21: replay harnesses, the installer/first-run UX, and gating the
  engine-internal `[APX]` / `[ATENIA]` log lines emitted by the runtime core
  on stderr (the CLI log level does not yet reach them — they are printed
  before / independently of the CLI logging layer). Known carried-over issue:
  the adaptive memory-pressure threshold (`0.85`) sits above the OS pagefile
  trigger on RAM-dominated boxes, so the OS pages before the reaction loop
  reacts.

---

## Readiness by audience

- **Systems / Rust / CUDA engineers** — solid ground. The architecture, the
  one-way layering, the kernels, and the test suite are real and inspectable.
- **AI infra engineers** — the tier-aware beyond-VRAM path is the proven
  differentiator on commodity hardware; expect to pin `ATENIA_DISK_TIER_DIR`
  to internal NVMe and to choose `certified` vs `fast` per workload.
- **Researchers / reviewers** — the per-checkpoint F64 certificate is the
  unique, auditable artefact; every number in [numcert/](./numcert/) is
  reproducible offline with a single `cargo test` invocation.
- **End users wanting a turnkey local-LLM tool** — not yet. M12 hardened the
  diagnostics and error surfaces (a prerequisite slice); the turnkey
  installer / first-run UX remains the v21 step.

*Hardware reference for every empirical number above: RTX 4070 Laptop, 8 GB
VRAM, 32 GB DDR5-5600, NVMe SN770, Windows 11.*
