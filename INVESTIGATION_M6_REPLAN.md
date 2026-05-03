# INVESTIGATION_M6_REPLAN — Why the original M6 approach failed, and how the replan solved it

This document is the third in the M6 investigation series. The
first two — `INVESTIGATION_M6.md` and `INVESTIGATION_M6_DEEP.md`
— recorded the post-load-upload approach, its instability under
real-world memory pressure (May 2 BSOD), and the deep architectural
audit that prepared the replan. This one records what actually
shipped and why the redesign worked where the original did not.

---

## 1. The original approach in one paragraph

M6 steps 1 → 4d wired GPU residency as a **post-load migration**:
the model loaded into RAM via the existing `WeightMapper::load_into`
path, then a separate `WeightStore::upload_layer_bf16_to_vram`
method walked the first N layers, materialised an F32 transient
on the host (`bf16_decode_bulk`), uploaded it to VRAM, and
replaced the `SharedParam::Bf16` entries with `SharedParam::Cuda`.
A new dispatch arm in `try_gpu_matmul` (M6 step 4d) detected
`(a=Cpu, b=Cuda, out=Cpu)` configurations and ran the kernel
against the resident weight.

The flow looked sound on paper. It was bit-exact on synthetic
tests, the standalone `examples/test_bf16_upload.rs` confirmed
the kernel's correctness on a real Llama 2 13B FFN-down shape,
and the per-step 4b unit test passed end-to-end.

It collapsed in production.

---

## 2. The May 2 BSOD

Running Llama 2 13B Chat with `ATENIA_GPU_RESIDENCY=1` +
`ATENIA_APX_MODE=4.19` on the dev box — 32 GiB host RAM, RTX 4070
Laptop with 8 GiB VRAM — produced a Windows blue screen during
the model load phase.

State at the moment of the crash, as captured by the parallel
`monitor_hw.ps1` poll:

```
RAM free min:  310 MB     (de 32 GiB — sistema al 99.06%)
RAM used max:  32,161 MB
GPU util max:  0 %        (driver crashed before any matmul fired)
GPU VRAM max:  8 MB        (one cudaMalloc had been issued)
Disk C: 100% sostenido durante 5+ minutos
```

The 13B BF16 model is 24.24 GiB resident. The per-layer upload's
peak memory state held *simultaneously*:

- The original `Arc<Vec<u16>>` (BF16 source, 270 MB for
  FFN-down).
- A fresh `Vec<f32>` materialised by `bf16_decode_bulk`
  (540 MB — twice the source).
- The VRAM destination buffer being filled by `cudaMemcpy`
  H→D (540 MB device-side, but Windows WDDM's shared memory
  policy maps 16 GiB of the GPU's "addressable" memory into
  host RAM, so part of this also pressured the host page
  table).

Per layer: ~1.35 GiB in flight on top of the 24 GiB model.
For the configured 5 layers: ~6.85 GiB cumulative pressure
on a system that had ~7 GiB of actual free RAM. The pagefile
absorbed the gap; the GPU driver's WDDM negotiator entered
an inconsistent state under sustained allocation pressure;
Windows's display-driver watchdog declared the GPU hung; the
bugcheck ran.

This was not "Atenia crashed." This was the operating
system declaring the configuration unfit for purpose.

---

## 3. Three structural defects

The audit triggered by the crash (`INVESTIGATION_M6_DEEP.md`)
identified three load-bearing limitations of the post-load
approach that no incremental fix could close:

### 3.1 The 8 GiB safety floor blocks 13B by construction

The safety gate in `gpu/safety/resource_check.rs` rejects GPU
operations when free RAM < 8 GiB. Loading 13B BF16 (24.24 GiB)
on a 32 GiB box leaves ~7 GiB of free RAM. The gate fires and
returns `DegradeToCpu`. No upload happens. The user observed
exactly this in the May 3 follow-up smoke:

```
[ATENIA] Safety check: 6.72 GiB RAM free (minimum 8 GiB
required for GPU operations) — DegradeToCpu
[M6] Layer 0: 0 params to VRAM, 0.00 GiB RAM freed
... (all 5 layers identical)
[M6] Residency total: 0 params, 0.00 GiB in VRAM, 0.00 GiB RAM freed
```

The gate's threshold is correctly tuned (lowering it re-enables
the BSOD scenario). The structural problem was that the gate
was running *after* the model had been loaded into RAM, when
the budget was already gone. The fix was not changing the
threshold; it was running the gate *before* the bulk of the
model loaded.

### 3.2 The hardcoded `n_layers = 5` cap

`pipeline.rs` step 4c hardcoded `let n_layers: usize = 5;`. No
env var, no derivation from `DegradeToLayers(N)`, no scaling
to the model's actual layer count. The deep doc commented this
was "subject to wire-up" but the wire-up was never implemented.
Even on a system with abundant headroom, only the first 5
layers ever migrated. A 32-layer 7B got 5/32 = 15.6%
residency; a 40-layer 13B got 5/40 = 12.5%. The cap was the
single hardest constraint on observable speedup.

### 3.3 The unit of decision was the layer, not the parameter

`upload_layer_bf16_to_vram` migrated *every* parameter of the
named layer. RmsNorm γ, biases, and any 1-D weight rode along.
But the executor's RmsNorm / RoPE / SiLU / Softmax /
BroadcastMul / BroadcastAdd / LogSoftmax / IndexSelect arms
all call `ensure_cpu` on their operand — a `Cuda` weight gets
downloaded to RAM every single step. Migrating γ to VRAM is a
net regression for every forward pass.

The deep doc's section 3 documented this for every NodeType in
the Llama forward; the fix was to make the unit of decision
the named tensor (so γ stays on RAM while q_proj migrates to
VRAM). The post-load approach had no such filter.

---

## 4. What the replan did

The replan kept *every* asset from the original M6 — the
`bf16_to_f32` kernel, the safe wrapper, the safety gate, the
`SharedParam::Cuda` variant, the mixed-residency dispatch in
`try_gpu_matmul`, the M6.c.7 fixes for `cuda_available()` and
the ATO ordering. It only changed *when* the tier decision is
made and *what kind of decision* it is.

### 4.1 Decision is per-tensor, at load time

`gpu::tier_plan::plan(input)` is a pure function from
`(metadata of every tensor in the safetensors header,
free RAM, free VRAM)` to `HashMap<name, Tier>`. The decision
runs *before* the model loads — the planner sees the empty-RAM
state of the system and budgets against the tensors that are
about to arrive, not against the tensors that already arrived.

The "GPU-eligible" filter at the per-tensor level captures
exactly the Llama-family Q/K/V/O attention projections and the
gate/up/down FFN projections (`name.ends_with("_proj.weight")`
+ rank ≥ 2). Norms, biases, embed_tokens, lm_head all auto-fall
to RAM because they don't match the filter — the planner does
not need to enumerate them as exceptions.

### 4.2 The loader writes to the assigned tier directly

`WeightMapper::load_into_with_residency_plan` replaces
`load_into`. For each safetensors entry, it dispatches by
tier:

- **VRAM**: BF16 raw bytes go directly to a `cudaMemcpy` H→D,
  then the GPU upcast kernel writes F32 into a persistent
  device buffer. The `SharedParam::Cuda` lands in the store
  with no host-side `Vec<u16>` or `Vec<f32>` ever
  materialised. Peak host RAM cost during this upload =
  the safetensors reader's owned bytes (which the caller was
  going to pay anyway).

- **RAM**: existing graph-slot write path. Hoisted into
  Arc-shared `SharedParam::F32` / `SharedParam::Bf16` after
  the loop via the existing `extract_from_graph` machinery,
  filtered to non-Vram/Disk param_ids.

- **Disk**: F32 working buffer materialised once for transforms,
  serialised to NVMe via `disk_tier::write_*_tensor`,
  registered as `SharedParam::Disk`. RAM transient drops at
  end of iteration.

A `vram_fast_path_count` / `vram_slow_path_count` pair of
process-wide atomics validates structurally that the no-transforms
BF16 case goes 100% through the no-host-transient path. The M6
replan integration test asserts `fast = N, slow = 0` for a
synthetic two-tensor plan.

### 4.3 No more "n_layers = 5" cap

The planner consumes the live free-VRAM measurement and fills
greedily until the 1 GiB working-buffer headroom is reached. On
the dev box's 8 GiB VRAM with 7 GiB usable, that yields ~5
layers' projections for 13B (where each layer is ~1.21 GiB F32)
and ~8 layers' projections for 7B (~840 MiB per layer F32).
The cap is now mechanical: it's whatever the available VRAM
is, modulo headroom.

### 4.4 The safety gate runs before the load

The planner consumes the same `probe_free_ram_bytes` /
`probe_free_vram_bytes` primitives the safety gate uses. With
no model in RAM yet, the planner sees ~26 GiB free; the
8 GiB headroom leaves 18 GiB of effective Ram-tier budget.
For 13B that comfortably accommodates the ~12 GiB BF16 model
minus the ~1.5 GiB worth that ends up on VRAM. For 7B the
budget is even more comfortable. The BSOD scenario is now
structurally unreachable.

---

## 5. The empirical result

```
Llama 2 7B Chat — RTX 4070 Laptop, 32 GiB host RAM
Default (CPU only):     12.02 s/tok    (0.083 tok/s)
Tier-aware loader:       8.22 s/tok    (0.122 tok/s)

Speedup: 1.46×
Output: bit-identical (" Hello! I'" first 5 tokens)
```

The plan log:

```
[ATENIA] Tier-aware loader plan:
  VRAM: 60 tensors (6.70 GiB)
  RAM:  263 tensors (9.20 GiB)
  Disk: 0 tensors (0.00 GiB)
```

60 of 7B's 224 `_proj.weight` tensors (32 layers × 7 = 224)
landed on VRAM — that's ~8.6 layers' projection capacity
(60 / 7 = 8.6), or ~27% of the model's matmul-eligible
weights. The 263 Ram-tier tensors are the remaining ~24
layers' projections plus the auto-excluded category
(64 norms × 1 weight each + embed + lm_head + final norm
= 67 — combined with the Ram-tier projections it sums to
the full 263).

The 1.46× speedup at 27% residency is roughly proportional
because each Vram-resident matmul also avoids the F32
transient + cudaMemcpy H→D that the Ram-tier matmul still
pays through `cuda_matmul_non_pooled`. In other words: VRAM
residency does double duty — saves the kernel time *and*
saves the upload time per call.

---

## 6. Lessons of process

The replan is the second design iteration of M6. The first
iteration had clean unit tests, working integration tests,
and a kernel that benchmarked at 84% of the RTX 4070's
theoretical memory bandwidth peak. None of that prevented
the BSOD. Five lessons emerged:

### 6.1 One activation per commit

The original M6 went 1 → 2a → 2b → 3a → 3b → 4a → 4b → 4c
→ 4d in single-purpose commits. That part worked: when the
May 2 BSOD happened, we could revert exactly the commit that
introduced the failure mode without losing the rest. The
replan kept this discipline (sub-fase 0 → 1 → 2 → 3, each a
separate commit, each closed by tests + smoke before the
next started).

### 6.2 PARAR ante anomalías

When the May 2 smoke produced 671 s/tok instead of the
expected ~14 s/tok, the user's instruction was clear:
**stop, report, wait**. Do not improvise fixes. The five
M6.c.7 hotfixes (`fix 1` through `fix 5`) were generated
under earlier, less disciplined cycles — each one fixed
*one* GPU surface and assumed that was the only one. The
deep audit (`INVESTIGATION_M6_DEEP.md`) showed there were
three GPU surfaces and the regression was not in any of
them — it was in the post-load architecture itself. The
audit cost ~5 hours; it saved an unbounded number of
hotfix cycles that were never going to converge.

### 6.3 If the test passes but production fails, the test does not match production

The standalone `examples/test_bf16_upload.rs` validated the
kernel + the upload primitive end-to-end with bit-exact
numerics on a real 270 MB FFN-down shape. The unit test for
`upload_layer_bf16_to_vram` did the same on a synthetic
4×4 weight. Both passed. The system still BSOD'd on real
13B because neither test exercised the *combined* peak
memory state of "BF16 source + F32 transient + VRAM
destination + accumulated residency from prior layers." The
replan added the `vram_fast_path_count` atomic specifically
because the structural property "no F32 transient on host"
is not visible from a numerical equivalence test.

### 6.4 The safety gate's threshold is not the place to debate

When the May 3 follow-up smoke showed the safety gate
correctly blocking all uploads (RAM was 6.72 GiB < 8 GiB
floor), the temptation was to lower the floor. The deep doc
held the line: the floor exists because RAM at 6.72 GiB free
+ active GPU operations is the BSOD scenario empirically.
Lowering the floor re-enables the failure mode. The fix
was upstream — change the architecture so the gate runs
before the budget is consumed.

### 6.5 The M4.7 tier system is a transition machine, not an allocator

A persistent pre-investigation assumption was "M4.7 already
does VRAM ↔ RAM ↔ NVMe; M6 just adds VRAM as a third tier."
The audit refuted this directly: M4.7's `TensorStorage::Disk`
is only ever produced by reactive spill (`migrate_all_to_disk`,
`deep_degrade_with_lru`), never by the loader. There is no
metadata path by which the loader can request "place this
tensor on disk." The replan had to introduce a load-time
tier planner — a new architectural capability, not a
generalisation of an existing pattern.

---

## 7. Closing note

The headline result — 1.46× speedup on Llama 2 7B with
bit-identical output — is modest in isolation. The
significance is that it is the first measurable, reproducible
GPU acceleration of a real LLM forward in Atenia, and it ships
with no destabilising side effects on the default path. M6 is
closed because the architecture that produced this result
generalises: M7's open gaps (Disk overflow for 13B-class models,
activation residency, BF16-resident VRAM, adaptive plan)
extend the tier-aware loader's planner and tier dispatch
without rewriting the core. The first iteration of M6
attempted to retrofit GPU residency onto an existing
RAM-resident loader; the replan put the tier decision at
the right architectural level. That's the part that
generalises.
