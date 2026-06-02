# PROPOSAL — NUMERIC-POLICY / MOE-PERF (design + audit only)

**Status: DESIGN ONLY. No code is to be written until this is reviewed.**

The MOE-PROD series (see [HANDOFF_MOE_PROD_SERIES.md](./HANDOFF_MOE_PROD_SERIES.md))
optimised the controlled MoE warm path to **184 s** while keeping the certified
**f64** reference math **bit-exact**. It proved the residual bottleneck is **CPU
f64 graph execution (~94 s)**, which **cannot shrink further without changing the
numerics**. That is a *policy* decision, not an engineering one. This proposal
designs the series that makes — and certifies — that decision.

Two intertwined questions:

1. **NUMERIC-POLICY** — how does Atenia let a workload *choose* its
   correctness/speed trade-off, and how is a non-bit-exact path **certified**
   instead of being bit-equal?
2. **MOE-PERF** — given such a policy, which compute backend (f32 CPU, bf16 CPU,
   GPU f32/TF32/BF16 Tensor Cores, GPU disk-streamed experts) wins for the MoE
   forward, under the 8 GB-VRAM / 24-core constraint?

---

## A. Existing assets to build on (audit)

- **GPU compute**: `cuda/matmul.rs` (CUDA GEMM), `cuda/disk_prefetch.rs` +
  `cuda_matmul_disk_streamed_bf16` (M8.7 — **disk→VRAM bf16 streaming matmul**,
  already used for dense Disk-tier weights), `cuda/bf16_to_f32.rs`,
  `gpu/tier_plan.rs` (VRAM/RAM/Disk placement planner), `gpu/dispatch/hooks.rs`.
- **Numerical certificate**: `moe/numerical.rs` (`max_abs_diff`, `rmse`,
  `argmax_match`, `MOE_NUMERICAL_TOLERANCE`), `nn/llama/numcert.rs` (per-checkpoint
  F64 certificate harness, reproducible offline). The "tolerance vs the f64
  reference" concept **already exists** for the dense path.
- **Tier residency**: `SharedParam{F32,Bf16,Disk,Cuda}` — the `Cuda` arm already
  exists; experts can in principle be placed in VRAM.
- **Reference math**: `moe/dense.rs`, `moe/mla.rs` (f64 `matvec`), the GraphBuilder
  attention/lm_head path (already rayon-parallel).
- **Hardware**: RTX 4070 Laptop — **Ada**, 8 GB VRAM, Tensor Cores supporting
  **FP16, BF16, TF32**; 24-core CPU with AVX2 (the SIMD bf16 decode already uses
  `bf16_decode_bulk`).

The pieces for a fast path mostly exist; what is missing is a **policy** that
selects them and a **certificate** that blesses their output.

---

## B. NUMERIC-POLICY design (the framework)

A per-run **NumericPolicy** with explicit tiers (proposed, names illustrative):

| Tier | Accumulation / units | Guarantee | Use |
|---|---|---|---|
| `Certified` | f64 reference (today) | **bit-exact**, reproducible | audit, golden refs |
| `Strict` | f32 accumulation, CPU/GPU f32 GEMM | `argmax_match` + `max_abs_diff ≤ τ₁`, `rmse ≤ τ₂` | default fast path |
| `Fast` | TF32 / BF16 Tensor Cores | `argmax_match` + looser `τ` over a validation prompt set | interactive |

Design points:

1. **Certificate, not bit-equality.** A non-`Certified` tier is *valid* iff, on a
   fixed validation set, it satisfies: (a) **greedy `argmax_match` = 100 %** for
   every generated token (the decisive property — same tokens out), and
   (b) per-logit-row `max_abs_diff`/`rmse` within published tolerances vs the
   `Certified` f64 run. Reuse `moe/numerical.rs` + the `numcert` harness; emit a
   **per-checkpoint, per-tier certificate artefact** (like the dense F64 cert).
2. **Policy is explicit and logged.** `ATENIA_NUMERIC_POLICY={certified,strict,fast}`
   (+ a Rust API). The chosen tier is recorded in output/telemetry so a result is
   never silently lower-precision.
3. **Fallback ladder.** A tier that fails its certificate (or a kernel that is
   unavailable, e.g. no CUDA) **falls back** to the next stricter tier, ending at
   `Certified`. Never a silent wrong answer — the MOE-PROD discipline carried
   forward.
4. **Tolerances are data, not magic numbers.** Publish `τ` per family/dtype, with
   the justification (bf16 has ~3 decimal digits; the tolerance must bound the
   accumulated error over `d_model`/`d_ff` reductions, not a single op).

Deliverable of the policy sub-series: the `NumericPolicy` type, the tier
selection + fallback, the certificate harness extension, and the published
tolerance table — **no kernel changes yet**.

---

## C. MOE-PERF — candidate compute backends (to evaluate)

Each is a hypothesis to **measure** against the policy certificate, in ROI order
informed by MOE-PROD-8 (generation = ~94 s, dominated by per-position expert FFN
+ attention/lm_head GEMMs; matvec already rayon-parallel but f64).

1. **f32 CPU + SIMD (`Strict`, CPU).** Replace per-element `f32→f64` accumulation
   with AVX2/FMA **f32** accumulation (optionally Kahan/pairwise for stability),
   blocked for cache. Lowest complexity, no GPU dependency. Expected: a few × on
   the matvec; the cheapest credible win. Risk: tolerance — must certify the
   accumulated f32 error stays within `τ` (pairwise/Kahan likely needed for
   `d_ff` in the thousands).

2. **bf16 CPU compute.** Compute the FFN in bf16 (Tensor-Core-free, AVX-512-BF16
   if available — *not* on this CPU). Likely **not** worth it on this box (no
   hardware bf16 GEMM); list for completeness, probably **rejected**.

3. **GPU f32 / TF32 GEMM (`Strict`/`Fast`, GPU).** Offload the expert FFN +
   attention + lm_head GEMMs to `cuda/matmul.rs`. TF32 Tensor Cores give a large
   speedup at ~f32 accuracy. **VRAM budget is the gate** (see §D). Expected: the
   biggest single win.

4. **GPU BF16 Tensor Cores (`Fast`, GPU).** bf16 inputs, f32 accumulate, on Ada
   Tensor Cores — the throughput ceiling. Largest divergence from f64 → loosest
   tolerance; must certify `argmax_match`.

5. **GPU disk-streamed experts.** Experts live on NVMe (bf16 tier from
   MOE-PROD-6). Reuse `cuda_matmul_disk_streamed_bf16` (M8.7) to stream an
   expert's bf16 weights **disk→VRAM→GEMM** without a host f32 materialisation —
   this also removes the MOE-PROD-8 resolve/copy cost. Synergy: the bf16 tier was
   *built for this*. Expected: removes both the CPU GEMM and the host-side
   resolve from the hot path.

---

## D. The 8 GB-VRAM constraint (the hard design problem)

The model is 14.3 B params (2.7 B active/token); it **does not fit** in 8 GB.
A GPU path must therefore be a **streaming / partial-residency** design, which is
exactly what `gpu/tier_plan.rs` + the M8.7 disk-stream path were built for:

- **Resident in VRAM**: the small, every-token tensors — attention q/k/v/o,
  embed/lm_head (or lm_head only), router, the **shared expert** (one per layer).
  Rough budget: attention ~1.6 GB f16, lm_head ~0.3 GB f16, shared experts
  ~1.7 GB f16 → fits with headroom for activations/KV.
- **Streamed from disk (bf16 tier)**: the **routed** experts (the bulk), via
  `cuda_matmul_disk_streamed_bf16`, overlapping NVMe→VRAM copy with compute
  (double-buffer). The MOE-PROD-6 bf16 tier halves the bytes streamed.
- **Open question**: PCIe/NVMe streaming bandwidth vs GEMM time — must measure
  whether streaming hides under compute or becomes the new bottleneck (it may
  just move the wall from CPU-compute to copy-bandwidth). A spike must answer this
  before committing.

---

## E. Proposed phases (design only — do not implement yet)

1. **NP-1 Audit & policy spec** — finalise the `NumericPolicy` tiers, the
   certificate definition (argmax + tolerance), the published `τ` table, the
   fallback ladder, and the telemetry. (doc)
2. **NP-2 Certificate harness** — extend `moe/numerical.rs` + `numcert` to emit a
   per-checkpoint, per-tier MoE certificate from a fixed validation prompt set.
3. **MP-1 CPU f32 spike** — `Strict`-CPU matvec (f32 + pairwise/Kahan); certify;
   measure. Lowest-risk win; may suffice for a big step.
4. **MP-2 GPU TF32 spike** — offload the resident GEMMs (attention/lm_head/shared)
   to TF32; certify; measure VRAM + speed.
5. **MP-3 GPU disk-streamed routed experts** — wire `cuda_matmul_disk_streamed_bf16`
   to the routed-expert FFN; double-buffer; measure streaming-vs-compute overlap.
6. **MP-4 BF16 Tensor Cores** — the throughput ceiling; certify the loosest tier.
7. **MP-5 Policy integration + benchmark matrix** — one table: tier × backend ×
   {wall, VRAM, tokens/s, certificate pass, argmax match}. Pick the default.

Each MP phase is **gated by its certificate** and falls back to `Certified`.

---

## F. Risks & open decisions (for human review)

- **Correctness policy is a product decision.** Shipping a non-bit-exact default
  changes what "correct" means for Atenia. The certificate makes it *defensible*
  (same tokens, bounded error), but the default tier choice is a human call.
- **VRAM streaming may just relocate the bottleneck** (CPU-compute → NVMe/PCIe
  copy). MP-3 must spike-measure before committing to the GPU expert path.
- **Tolerance calibration is subtle.** bf16 over thousands-wide reductions can
  drift; `argmax_match` can fail on near-ties. The `τ` table needs a real
  validation set, not a single prompt.
- **Scope creep.** This touches the engine's numeric core, not just MoE. The
  policy should be designed once, MoE-first, but general.
- **Maintenance.** Multiple compute paths × dtypes × CPU/GPU multiplies the test
  matrix; the certificate harness must be the gate that keeps it honest.

---

## G. Recommendation (for the decision, not an action)

Start with **NP-1/NP-2 (policy + certificate)** and **MP-1 (CPU f32)** — they are
low-risk, need no GPU, and likely deliver a large step while *establishing the
certification machinery* that every later GPU phase depends on. Treat **MP-3 (GPU
disk-streamed experts)** as the high-ceiling bet, but only after an MP-2 spike
confirms the VRAM budget and the streaming-overlap assumption.

**No implementation until reviewed.** This document is the deliverable.
