# Handoff — APX v20 M6 (Tier-Aware GPU Loader, at M6 close)

**Status at handoff**: M6 closed. Atenia loads each parameter
directly into the highest available tier (VRAM → RAM → NVMe)
based on a planner that probes free RAM and free VRAM at load
time. The first measured speedup of an LLM forward on real GPU
hardware ships in this milestone.

After `cargo build --release --bin atenia`, running

```text
atenia generate \
    --prompt "Hello, how are you?" \
    --model models/llama-2-7b-chat \
    --max-tokens 5
```

(*tier-aware is the default since commit `afaa975`; the
historical `ATENIA_TIER_AWARE_LOADER=1` flag is recognised
as a deprecated no-op*) loads Llama 2 7B Chat with the
planner's decision logged to
stderr, uploads ~6.7 GiB of attention/FFN projections directly
to VRAM (no host-side F32 transient), keeps the rest in RAM,
and runs greedy generation through the M6 step 4d mixed-residency
matmul dispatch:

```text
[ATENIA] Tier-aware loader plan:
  VRAM: 60 tensors (6.70 GiB)
  RAM:  263 tensors (9.20 GiB)
  Disk: 0 tensors (0.00 GiB)

> Hello, how are you?

Hello! I'

---
Generated: 5 tokens in 41.1s (0.12 tok/s) [max-tokens reached]
```

The same prompt under the default (CPU-only) path gives
**60.1 s for 5 tokens (0.08 tok/s)** with bit-identical output.
The tier-aware path is **1.46× faster** at this configuration
(8.22 s/tok vs 12.02 s/tok) — and produces the identical text,
proving the GPU residency path is numerically faithful.

**Last M6 commit**: `0e50833` (M6 replan sub-fase 3 — tier-aware
loader wired in pipeline.rs). M6 closes at this commit; M7
(13B-friendly tiers — Disk overflow, adaptive plan, activation
overflow) is the next active milestone.

---

## Headline result

| Path | Build flag | Token avg | Throughput | Speedup |
|---|---|---|---|---|
| **Default CPU** (M5.f.a baseline) | `cargo run --release` | 12.02 s/tok | 0.083 tok/s | 1.0× |
| **Tier-aware loader** | `ATENIA_TIER_AWARE_LOADER=1` | **8.22 s/tok** | **0.122 tok/s** | **1.46×** |

Llama 2 7B Chat, RTX 4070 Laptop (8 GiB VRAM dedicated), 32 GiB
host RAM, NVMe SSD. Generated text is bit-identical between the
two paths.

---

## What is ready

| Sub-phase | Title | Commit |
|---|---|---|
| **M6 step 1** | `cuda_available()` cached behind `OnceLock<bool>` | `92da341` |
| **M6 step 2a** | `cuda_matmul_non_pooled` (dead-code primitive) | `10a9d44` |
| **M6 step 2b** | Lift G5 pool gate, route oversize matmuls to non-pooled | `f3b7e9b` |
| **M6 step 3a** | `bf16_to_f32` CUDA kernel + standalone `examples/test_bf16_upload.rs` | `5421655` |
| **M6 step 3b** | `gpu-trace` diagnostic harness (cfg-gated, default off) | `ba99c2c` |
| **Safety gate** | `gpu/safety/resource_check.rs` — pre-operation RAM/VRAM probe | `164d452` |
| **ATO ordering** | GPU dispatch attempted before Auto-Tiling Optimizer | `45c7b78` |
| **M6 step 4a** | Safe wrapper `bf16_to_f32_on_device` | `395451e` |
| **M6 step 4b** | `WeightStore::upload_layer_bf16_to_vram` (post-load upload, superseded) | `98d831e` |
| **M6 step 4c** | Opt-in caller in `pipeline.rs` (`ATENIA_GPU_RESIDENCY=1`) | `48bf205` |
| **M6 step 4d** | Mixed-residency dispatch (Cpu activation + Cuda weight) in `try_gpu_matmul` | `7cf1264` |
| **Replan sub-fase 0** | `SharedParam::Disk` variant + round-trip test | `36a1eb0` |
| **Replan sub-fase 1** | `gpu/tier_plan.rs` — pure-function residency planner | `1ffbf83` |
| **Replan sub-fase 2** | `WeightMapper::load_into_with_residency_plan` (direct VRAM/Disk load) | `e78cd54` |
| **Replan sub-fase 3** | Tier-aware loader wired in `pipeline.rs` (`ATENIA_TIER_AWARE_LOADER=1`) | `0e50833` |

13 production commits across two design iterations: an initial
"post-load upload" attempt (steps 1–4d) that hit the
construction limits of the 32 GiB box, and a replan that flipped
the architecture to "load each tensor directly into its tier."
The original approach is intentionally still on `main` —
gated behind a separate env var (`ATENIA_GPU_RESIDENCY=1`) that
the replan's gate (`ATENIA_TIER_AWARE_LOADER=1`) supersedes
when both are set.

---

## Architecture — what shipped

### The tier-aware loader (the load-bearing piece)

`gpu::tier_plan::plan(input) -> TierPlan` is a pure function
that converts safetensors header metadata + a free-memory
snapshot into a `HashMap<name, Tier>` where `Tier ∈ {Vram,
Ram, Disk}`. Policy:

- **GPU-eligible filter**: `name.ends_with("_proj.weight") &&
  shape.len() >= 2`. Captures Q/K/V/O attention projections
  and the gate/up/down FFN projections of every Llama-family
  model. Auto-excludes RmsNorm γ, embed_tokens, lm_head,
  biases — those tensors are consumed by ops that force
  `ensure_cpu` and would download from VRAM every step,
  regressing throughput.
- **Greedy bin packing** in input order:
  - VRAM: fill until `free_vram - 1 GiB headroom` is consumed.
  - RAM: fill remaining tensors until `free_ram - 8 GiB
    headroom` is consumed.
  - Disk: any overflow.
- **Cost calculations**:
  - VRAM cost is always F32 (`numel × 4`) because the upload
    kernel `bf16_to_f32_resident_in_vram` produces an F32
    device buffer.
  - RAM/Disk cost is the source dtype size.

The 8 GiB RAM headroom protects against the May 2 BSOD
scenario where a 32 GiB box loading a 24 GiB BF16 model
crossed into pagefile thrashing under sustained GPU activity.

### `WeightMapper::load_into_with_residency_plan`

The new loader entry point. For each safetensors entry,
checks `plan.get(name)` and dispatches:

- **VRAM fast path** (BF16 source + zero `LoadTransform`s):
  raw safetensors bytes → `cudaMemcpy` H→D → GPU upcast kernel
  → `SharedParam::Cuda`. **Zero F32 transient on the host.**
  Validated structurally by `vram_fast_path_count()` /
  `vram_slow_path_count()` atomics (test asserts a no-transforms
  BF16 plan goes 100% through the fast path).
- **VRAM slow path** (F32 source or transforms): F32 working
  buffer materialised, transforms applied, down-converted to
  BF16 bits, uploaded, dropped.
- **RAM**: existing graph-slot write path; pass-2 hoist via
  `extract_from_graph` filtered to non-Vram/Disk param_ids.
- **Disk**: serialise to `disk_tier::write_*_tensor`, register
  `SharedParam::Disk`. RAM transient drops at end of iteration.

Multi-shard models (Llama 7B, 13B) go through
`ShardedSafetensorsReader::load_into_with_residency_plan`,
which calls `collect_tensor_metas()` first to build the
plan from all shards' metadata before iterating shards a
second time to actually load.

### Dispatch — `try_gpu_matmul` mixed-residency

When the executor's MatMul arm reaches a `(a=Cpu activation,
b=Cuda weight, out=Cpu)` configuration (which is what the
tier-aware loader produces for a Vram-tier weight + a CPU
activation flowing from the previous op), `try_gpu_matmul`:

1. Uploads the small activation to VRAM (~20 KB for a
   `[1, 13824]` FFN-down input on Llama 13B).
2. Calls `cuda_matmul_inplace` with the resident weight.
3. Downloads the small output (~20 KB).
4. Increments `GPU_MATMUL_RESIDENT_COUNT`.

The weight upload — the 270 MB H→D copy that the post-load
approach would have to repeat per matmul — is paid **once**
at load and never again.

### Safety gate

`gpu::safety::resource_check::check_before_gpu_operation` is
the pre-flight check that protects against the BSOD scenario:

- < 8 GiB free RAM → `DegradeToCpu`.
- 8–12 GiB → `DegradeToLayers(N)` where `N` is computed from
  the headroom.
- ≥ 12 GiB AND VRAM has room for the request + 1 GiB working
  buffer → `Proceed`.

The tier-aware planner consumes the same probes
(`probe_free_ram_bytes`, `probe_free_vram_bytes`) directly,
making the safety gate effectively a load-time invariant.

---

## D-decisions in M6

- **D70 — Tier as the unit of placement, not the layer.**
  Within a Llama layer, `_proj.weight` tensors benefit from
  VRAM (consumed by MatMul, which has a residency dispatch
  path) while RmsNorm γ and biases regress on VRAM (consumed
  by ops that force `ensure_cpu`). Per-named-tensor decisions,
  not per-layer. Documented in `gpu/tier_plan.rs` module docs.

- **D71 — Greedy bin-packing in input order.**
  The planner does not optimise for "balanced layers." It
  fills VRAM with whatever GPU-eligible tensors come first
  (file order ≈ layer-0-first), accepting partial-layer
  residency. The cost of a true bin-packing algorithm
  (fragmenting decisions across layers) is not justified at
  M6's scope; M7 may revisit if the partial-layer pattern
  produces measurable hot-path penalties.

- **D72 — Direct-VRAM bytes path, not host-side `Vec<u16>`.**
  `bf16_to_f32_resident_in_vram_from_raw_bytes` takes a
  `&[u8]` directly from the safetensors reader's owned buffer
  and `cudaMemcpy`s it to VRAM without ever building a
  Rust-side `Vec<u16>` or `Vec<f32>`. The fast path is the
  M6 contract: peak host RAM during a Vram-tier upload is
  the BF16 source bytes only — no transient.

- **D73 — Replan over patch.**
  After the May 2 BSOD demonstrated that the original
  "post-load upload" approach (M6 steps 4a–4d) had
  load-bearing constructional limits (8 GiB safety floor
  blocks 13B by construction; hardcoded `n_layers = 5`
  caps any model), a deep architectural investigation
  (`INVESTIGATION_M6_DEEP.md`) concluded that the right
  primitive was a load-time tier planner, not a post-load
  migration. The replan ships alongside the original
  implementation rather than replacing it; both are gated
  by their own env vars.

- **D74 — `ATENIA_TIER_AWARE_LOADER=1` opt-in, not default-on.**
  *(Superseded at commit `afaa975`, M8.7 prereq close.)*
  The default smoke must remain bit-exact with M5.f.a. The
  tier-aware loader requires CUDA detection plus disk-cache
  setup plus load-time tier-plan logging — none of which
  should fire on a CPU-only or read-only test run. The
  flag flips during operator validation and stays in the
  smoke script for production runs once stability is
  re-validated per-environment.

  **Update (commit `afaa975`)**: re-validation completed
  across M6 (1.46× on 7B, bit-exact), M7 (13B without BSOD,
  automatic tiers), and M8 (1.31× / 1.36× on 7B / 13B with
  ADR-004 preserved), and the operator confirmed a 21.9
  s/token 13B baseline through the tier-aware path. The
  policy is now inverted: tier-aware is the default;
  `ATENIA_LEGACY_LOADER=1` is the new opt-out.
  `ATENIA_TIER_AWARE_LOADER` is still recognised as a
  no-op with a deprecation warning during a grace period.

- **D75 — Safety gate post-load is structurally correct.**
  Despite the user-facing observation that the safety gate
  blocked all 13B uploads on the original 32 GiB box, the
  gate itself is correctly tuned. The fix was not lowering
  the floor (which would re-enable BSOD); it was changing
  the architecture so the gate ran *before* loading the
  bulk of the model (M6 replan), giving it actionable free
  RAM to budget against.

---

## What is gapped

### Closed gaps

1. **Per-call weight upload overhead** (M5 baseline).
   The 30-100 ms per-matmul `cudaMalloc + memcpy + free` of
   the post-load upload primitive is gone. Resident weights
   pay the upload once at load.

2. **F32 transient duplication** (post-load upload regression).
   The May 2 BSOD root-caused to a `Vec<u16>` BF16 source +
   `Vec<f32>` decoded transient + VRAM destination buffer
   coexisting per layer (~1.35 GiB in flight ×  N layers).
   The replan's direct-bytes path has zero F32 transient on
   the host for the BF16-source-no-transforms case (which is
   100% of Llama 2 weights).

3. **CPU-vs-GPU dispatch for oversize shapes** (M6 step 2b).
   The 64 MiB pool gate that bounced every 13B weight matmul
   to CPU is lifted; oversize shapes route to
   `cuda_matmul_non_pooled` directly.

4. **ATO blocking GPU dispatch** (M6 ordering fix).
   The Auto-Tiling Optimizer's `apx_mode >= 6.6` capture
   used to fire before `try_gpu_matmul` could see the node.
   Reordered: GPU first, then ATO as CPU fallback.

5. **`cuda_available()` per-call subprocess spawn** (M6 step 1).
   `nvidia-smi` is now spawned exactly once per process
   lifetime (`OnceLock<bool>` cache); subsequent calls are
   single atomic loads.

### Open gaps for M7

1. **13B in 32 GiB box.** The current planner does not
   emit `Tier::Disk` assignments for 13B in this hardware
   class because the RAM headroom (24 GiB free post-OS)
   exceeds the 12 GB BF16 model footprint after the 8 GiB
   floor. But VRAM caps residency at ~5 layers; the
   remaining 35 layers stay on RAM and pay the per-matmul
   F32 transient cost the original loader pays. Throughput
   uplift on 13B is therefore modest. M7's "Disk overflow"
   sub-phase ships when the planner can route some Vram-
   ineligible tensors to NVMe, freeing RAM for additional
   layer residency.

2. **Activation residency.** Every non-MatMul op
   (RmsNorm, RoPE, SiLU, Softmax, BroadcastMul,
   BroadcastAdd, LogSoftmax) calls `ensure_cpu` on its
   operand. A `Cuda` operand triggers a D→H download
   per call. The current dispatch design downloads
   the matmul output for every Vram-resident matmul to
   feed the next non-MatMul op. M7's "activation residency"
   sub-phase keeps activations on VRAM between
   compatible ops by porting RmsNorm / RoPE / SiLU to
   GPU-resident kernels. Significant scope.

3. **BF16-resident VRAM path.** The current upload kernel
   produces F32 device buffers (270 MB per FFN-class
   weight). A BF16-resident VRAM path would halve the
   per-layer VRAM cost and let the planner fit ~10
   layers in 8 GiB instead of ~5. Requires a BF16
   matmul kernel (`__nv_bfloat16` operands with F32
   accumulator) and a corresponding `try_gpu_matmul`
   dispatch arm. Numerical validation against ADR-004
   needed across the four-model family.

4. **Adaptive plan refresh.** The planner runs once at
   load time and never adjusts. If the system enters
   memory pressure mid-session (browser tab balloon,
   another process spawning), the M4.7 spill mechanism
   still triggers but does not coordinate with the
   tier plan. M7's "adaptive plan" sub-phase decouples
   the load-time plan from the runtime spill so they
   can negotiate.

5. **Multi-GPU residency.** `gpu/safety/resource_check.rs`
   queries `nvidia-smi -i 0` only. A second dedicated GPU
   (or the iGPU's shared memory) would be reachable with a
   `gpu::backend::Backend` trait extension. Out of M6 scope;
   open as v21 territory.

---

## Validated empirically (M6 close)

- **Bit-exact output** — Llama 2 7B Chat, prompt
  "Hello, how are you?", first 5 tokens identical between
  default and tier-aware paths (`" Hello! I'"`).

- **Stable system** — RAM 99% pressure under tier-aware
  load on the 32 GiB box stays stable; no BSOD; planner's
  headroom budgets respected.

- **Counter-validated zero F32 transient** —
  `vram_fast_path_count` increments by N (= number of
  Vram-tier BF16 tensors with no transforms) and
  `vram_slow_path_count` stays at 0 across both standalone
  and integration tests. Structurally proves the M6
  contract on the host-RAM peak.

- **Five regression suites** (`tinyllama_config_test`,
  `tinyllama_weight_loading_test`, `tinyllama_builder_test`,
  `weight_mapper_test`, `miniflux_safetensors_roundtrip_test`)
  green throughout the milestone — 43 tests covering the
  numerical path, the safetensors I/O, and the graph
  builder.

- **5 tier-plan unit tests** covering the four free-RAM
  bands and a Llama 2 13B realistic shape distribution
  (363 tensors).

- **3 integration tests** for the loader (all-Ram bit-exact,
  mixed-plan Cuda variant, fast-path counter check).

---

## How to resume on M7

1. Read this handoff and `INVESTIGATION_M6_REPLAN.md`.

2. Pick one open gap (Disk overflow, activation residency,
   BF16-resident VRAM, adaptive plan, multi-GPU). The
   four are largely independent; ordering can match the
   end-user pain point currently observed.

3. Each sub-phase follows the M6 protocol that emerged from
   the replan:
   - One activation per commit.
   - Mandatory regression suites green before push.
   - Smoke immediately after with default flags + with the
     new flag, comparing both against the previous baseline.
   - PARAR (stop) on any unexpected anomaly. Do not
     improvise fixes — report and wait. The replan
     experience showed five incremental hotfixes did not
     close a bug whose root was architectural.

4. The relevant assets to re-use:

   - `gpu::tier_plan` — the planner. New tier sub-phases
     extend the `Tier` enum and the policy in
     `is_gpu_eligible` + cost calculations.

   - `cuda::bf16_to_f32::bf16_to_f32_resident_in_vram*` —
     the upload primitives. A BF16-resident kernel would
     ship as a sibling.

   - `gpu::safety::resource_check` — the safety gate.
     New tiers extend `SafetyDecision`.

   - `WeightMapper::load_one_shard_into_with_residency_plan`
     — the per-shard loader primitive. New write paths
     attach to the `match tier` block.

5. Smoke baseline reference for new sub-phases (tier-aware is
   default since `afaa975`; no flag required):

   ```
   atenia generate --prompt "Hello, how are you?" \
       --model models/llama-2-7b-chat --max-tokens 5
   ```

   Expected: 5 tokens in 41±5 s, plan logs 60 Vram /
   263 Ram / 0 Disk on a clean 32 GiB box. Any sub-phase
   that regresses this either improves the plan
   (more Vram, fewer Disk) or introduces a configurable
   knob — never makes the default slower.
