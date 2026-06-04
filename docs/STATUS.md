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
