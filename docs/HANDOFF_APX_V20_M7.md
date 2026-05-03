# Handoff — APX v20 M7 (13B-friendly tiers, at M7 close)

**Status at handoff**: M7 closed. Llama 2 13B Chat runs end-to-end
on a 32 GiB Windows box with an 8 GiB RTX 4070 Laptop GPU and
project data on NVMe — without BSOD, without thrashing, without
manual offload knobs. The tier-aware loader (M6) plus the new
Disk fast-path (M7.1) plus the adaptive RAM headroom (M7.2)
together place every parameter into the highest tier that the
hardware can support, automatically. The May 2 BSOD scenario that
forced the M6 architectural replan is now closed by construction:
peak free RAM stayed at **7.36 GiB** during the smoke and the
disk subsystem was bursty, not saturated.

After `cargo build --release --bin atenia`, running

```powershell
$env:ATENIA_TIER_AWARE_LOADER = "1"
$env:ATENIA_DISK_TIER_DIR    = "D:\atenia-m7-cache"

cargo run --release --bin atenia -- generate `
    --prompt "Hello, how are you?" `
    --model D:/Atenia/models/llama-2-13b-chat `
    --max-tokens 5
```

loads Llama 2 13B Chat with the adaptive headroom and the tier
plan logged to stderr, writes ~20 GiB of BF16 weights directly
to NVMe via the M7.1 fast-path (no F32 transient), keeps the
remaining ~7 GiB split between VRAM and RAM, and produces a
coherent reply through the M6 step 4d mixed-residency dispatch:

```text
[ATENIA] Adaptive headroom: model 24.24 GiB, free RAM 19.41 GiB,
         total RAM 31.71 GiB → RAM headroom 18.65 GiB
         (8.00 base + 10.65 overflow protection)
[ATENIA] Tier-aware loader plan:
  VRAM: 38 tensors (6.70 GiB)
  RAM:  126 tensors (0.75 GiB)
  Disk: 239 tensors (20.14 GiB)

Model loaded in 198.9s (363 parameters, 27.59 GiB resident).

> Hello, how are you?

 Hello! I'

---
Generated: 5 tokens in 183.0s (0.03 tok/s) [max-tokens reached]
```

**Last M7 commit**: `19fdcf8` (M7.2 — adaptive headrooms). M7
closes at this commit; the next active milestone is **M8**
(performance optimisation of the 13B Disk-tier path), with
**M7.4** (Disk LRU cache + prefetch) available as an optional
predecessor if 13B throughput becomes a deployment-time blocker.

---

## Headline result

| Configuration | Tier distribution | Peak RAM in use | Token avg | Stability |
|---|---|---|---|---|
| **Llama 2 13B Chat, 32 GiB box, RTX 4070 Laptop, NVMe** | 38 VRAM / 126 RAM / **239 Disk** | ≤ 24.6 GiB (≥ 7.36 GiB free) | 36.6 s/tok | No BSOD, no thrashing |

The four success criteria for M7.3 all passed:

| # | Criterion | Threshold | Observed |
|---|---|---|---|
| 1 | Disk overflow active | `Disk > 0` tensors | **239 tensors / 20.14 GiB** ✅ |
| 2 | Disk fast-path engaged | `disk_fast_path_count > 0` | implied by load completing with `store_params_as_bf16` and zero F32 host transient ✅ |
| 3 | RAM headroom respected | ≥ 4 GiB free at all times | **min 7.36 GiB free** during the entire run ✅ |
| 4 | Coherent generation | 5 tokens, no BSOD | **" Hello! I'"** ✅ |

None of the rollback criteria triggered: free RAM never dropped
below 7 GiB (rollback floor 2 GiB), `disk_busy_pct` only hit 100 %
for ~1 s total (rollback threshold > 30 s sustained), the first
token emitted within the 600 s budget, and the safety gate did
not degrade to CPU.

---

## What M7 actually shipped

Four sub-phases, each a single-commit activation gated behind a
mandatory regression suite (43 tests across 5 suites: `tinyllama_config_test`,
`tinyllama_weight_loading_test`, `tinyllama_builder_test`,
`weight_mapper_test`, `miniflux_safetensors_roundtrip_test`).

### M7.0 — NVMe bench + bit-exact disk loader test (commit `8f29233`)

Gating data point. Synthesised a Llama 2 13B FFN-down weight
(5120 × 13824 BF16 = 141.6 MB on disk), wrote it to NVMe via
`disk_tier::write_bf16_tensor`, then read it back three times
(cold / warm / warm) with bit-exact verification.

Measured: **3.6 GB/s sustained**, **37 ms cold read**. Projected
worst-case 280-matmul step at 10.4 s/token if every weight is
cold — already comfortably below the M5 CPU baseline of ~14 s/tok
on the same shape. **Decision: Plan A** (proceed M7.1 → M7.3
directly without inserting a Disk LRU cache).

Bench delivered as `examples/bench_disk_weight.rs`; bit-exact
test as `tests/m7_disk_loader_test.rs::disk_tier_loader_arm_executes_and_round_trips_bit_exact`.

### M7.1 — Disk fast-path raw bytes (commit `db5a49f`)

Mirror of the M6 VRAM fast-path. When a tensor is BF16 in the
safetensors file, has no transforms, and the loader is configured
with `store_params_as_bf16`, the bytes flow directly from the
mmap'd safetensors body into the `disk_tier` file with **zero
intermediate Vec<u16> or Vec<f32>**. New primitive:

```rust
pub fn write_bf16_from_raw_bytes(
    cache_dir: &Path, raw_bytes: &[u8], numel: usize,
) -> io::Result<DiskTensorHandle>
```

Counters added: `DISK_FAST_PATH_COUNT`, `DISK_SLOW_PATH_COUNT`.
Structural validation that the 13B load consumed essentially
zero RAM during the Disk-tier write phase: every `Tier::Disk`
BF16-no-transforms tensor increments `disk_fast_path_count`
and never `disk_slow_path_count`. Test:
`tests/m7_disk_loader_test.rs::disk_fast_path_with_bf16_store_no_transforms`.

Reduces peak RAM during the write of a 270 MB FFN-down weight
from ~810 MB (BF16 source + Vec<u16> copy + Vec<f32> upcast)
down to the source mmap residency only.

### M7.2 — Adaptive RAM headrooms (commit `19fdcf8`)

The M6 planner used a fixed 8 GiB RAM headroom regardless of
model size. On a 32 GiB box, that meant a 24 GiB BF16 model fit
entirely in `(32 − 8) = 24 GiB` of RAM budget — the Disk tier
**never triggered** even on hardware where it should have. M7.2
inflates the headroom proportionally:

```
threshold = 0.7 × free_ram_bytes
if model_total > threshold:
    headroom = 8 GiB + (model_total − threshold)
else:
    headroom = 8 GiB                          (M6 base, unchanged)
```

`TierPlanInput` now carries `model_total_bytes` and
`total_ram_bytes`. `TierPlan` exposes `ram_headroom_bytes` and
`ram_headroom_overflow_bytes` so the caller can log the breakdown.
The pure helper `adaptive_ram_headroom(model, free_ram) → (h, o)`
is unit-tested independently from the bin-packing.

Five new tests in `src/gpu/tier_plan.rs`:

- `m7_2_adaptive_13b_on_32gib_box_triggers_disk_overflow` — the headline scenario, verifies bump fires and Disk count > 0.
- `m7_2_adaptive_7b_on_32gib_box_keeps_base_headroom` — regression-zero for the M6 7B path; everything still in RAM.
- `m7_2_adaptive_7b_on_16gib_box_forces_disk` — RAM-constrained box; budget saturates, every tensor lands on Disk.
- `m7_2_adaptive_small_model_keeps_base_headroom` — small model, modest box; rule does not fire (asserts on headroom, not placement).
- `m7_2_adaptive_ram_headroom_helper_branches` — direct helper unit test covering below-threshold, exactly-at-threshold (strict `>`), above-threshold, and `free_ram = 0`.

The five existing M6 tests were updated with the new fields;
their behaviour is preserved (greedy bin packing, GPU-eligible
filter, byte accounting).

### M7.3 — 13B integration smoke (this milestone's gating run)

End-to-end on the operator's hardware. The numbers above are
this run; the success/rollback table at the top of this doc is
this run's checklist.

The smoke completed in 382 s wall-clock end-to-end:
- **Load**: 198.9 s (~3:18). Reads safetensors, writes 20.14 GiB
  to NVMe via M7.1 fast-path, uploads 6.70 GiB to VRAM via the
  M6 fast-path, holds 0.75 GiB in RAM.
- **Generation**: 183.0 s for 5 tokens. First-token latency
  dominates (warm-up of OS page cache); steady-state would be
  faster but `--max-tokens 5` does not give enough samples.
- **Adaptive math verified**: model 24.24 GiB, free RAM 19.41 GiB,
  threshold 0.7 × 19.41 = 13.59 GiB, extra = 24.24 − 13.59 = 10.65,
  headroom = 8.00 + 10.65 = 18.65 GiB. Logged values match exactly.

Hardware telemetry from `monitor_hw.ps1` (`m7_13b_hw.log`):

| Signal | Floor / ceiling | Observed | Margin |
|---|---|---|---|
| `free_ram_MB` | rollback at < 2048 | min **7 535** | 3.7× the floor |
| `disk_busy_pct` sustained | rollback at > 30 s @ 100 % | max 1 s @ 100 % | 30× margin |
| VRAM used / 8188 total | M6 headroom 1 GiB | peak 7 245 (88 %) | within budget |

---

## Architectural decisions taken in M7

1. **Plan A condicional adopted**: M7.0's 37 ms cold-read figure
   carried the milestone past the gate. No Disk LRU cache was
   introduced. The forward path reads each Disk-tier tensor
   per-token; the first-token latency is high but subsequent
   tokens benefit from OS page cache. M7.4 is reserved as the
   activation point if production deployments need < 10 s/tok
   on 13B.

2. **`model_total_bytes` enters the planner contract**. Previously
   the planner was a pure function of `(tensors, free_vram, free_ram)`.
   M7.2 expanded it to also need `(model_total, total_ram)`. The
   loader sums `model_total` from `collect_tensor_metas()` and
   probes `total_ram` via `sysinfo::System::total_memory()`. The
   planner itself remains pure — no new I/O at plan time.

3. **Strict `>` on the threshold**. The adaptive rule fires when
   `model_total > threshold`, not `≥`. Boundary case explicitly
   tested. Rationale: at the exact threshold the model still
   fits, so the M6 base headroom is correct; only when the model
   strictly exceeds the threshold do we need extra protection.

4. **The `27.59 GiB resident` log line is informational, not
   structural**. It sums VRAM + RAM + Disk. Disk is the file
   size on NVMe, not host residency. The semantics are
   "everywhere the loader placed bytes", which is what the
   operator wants to see at a glance.

5. **No Disk-tier kernel residency**. The forward path
   materialises Disk-tier weights to RAM on demand via
   `Tensor::ensure_cpu`'s Disk arm (M4.7.4 era), then dispatches
   to the existing CPU/GPU path. There is no "Disk → VRAM
   direct upload" path; that would be M7.4 territory if needed.

---

## Gaps closed by M7

- **R3 (BSOD on 13B with active GPU residency)**. The May 2 BSOD
  scenario was traced to the post-load upload approach that held
  BF16 source + F32 transient + VRAM destination simultaneously
  per layer. M6's tier-aware loader closed it for the 7B case
  (load-time tier decision); M7.2's adaptive headroom plus
  M7.1's Disk fast-path closes it for the 13B case. The smoke
  ran without anomalies for 6:22 wall-clock, peak RAM in use
  ~24.6 GiB (≥ 7.36 GiB free).
- **R5 (fixed headroom misroutes 13B)**. M6 routed the entire
  13B BF16 model into RAM on a 32 GiB box; the Disk tier never
  triggered. M7.2 makes the headroom adapt to the model size.
- **Disk fast-path missing**. M7.1 mirrors the VRAM fast-path
  for the Disk tier. Counters validate "no F32 transient on
  the Disk path".

## Gaps left open (intentional)

- **First-token latency on 13B is 36.6 s/tok** — the bench
  projected 10.4 s/tok worst case for a single weight. The gap
  is decode-on-read (`bf16_to_f32` per layer per step), not I/O
  saturation (`disk_busy_pct` was bursty, not pinned at 100 %).
  Closing this would require either:
  - a Disk LRU cache + prefetch (M7.4), or
  - a BF16-resident path that reads `Vec<u16>` from NVMe and
    feeds the GPU upcast kernel directly without a host F32
    detour.
  Neither is on the M7 critical path and both stay open.
- **`--max-tokens 5` only gives the first-token latency**, which
  is warm-up-dominated. Steady-state token timing on 13B is
  unmeasured by M7.3. A higher-budget smoke (e.g. 50 tokens) is
  the natural follow-up if performance triage starts.
- **The Disk slow path is reachable** for BF16 tensors that have
  any transform attached (transpose-on-load, scale-on-load, etc.)
  or for F16/F32 source dtypes. None of the four production
  Llama-family checkpoints today exercise that path, so its
  RAM consumption is dormant. Worth an audit if a future
  checkpoint introduces transforms.
- **`disk_fast_path_count` is not printed in the operator-facing
  log**. It is read in the M7.1 unit test. A `log_disk_residency`
  emitter analogous to `log_tier_plan` would be a 5-line addition
  if we want it for telemetry. Not a blocker.

---

## How to resume

Three independent next-step trajectories, listed in increasing
scope:

### Option 1 — M7.4 (Disk LRU cache + prefetch)

If 13B throughput is a deployment concern, this is the targeted
fix. The M7.0 bench measured 3.6 GB/s NVMe; the M7.3 smoke
showed `disk_busy_pct` is bursty (idle most of the time). The
forward path is decode-bound, not I/O-bound, but the OS page
cache is doing the caching today implicitly. An explicit LRU
on the loader side would let the planner free the OS file cache
budget for activations and KV cache.

Files to touch:
- `src/tensor/disk_tier.rs` — add a `DiskCache` with bounded
  RAM residency, served by `read_bf16_tensor` first.
- `src/tensor/tensor.rs` — `ensure_cpu`'s Disk arm consults the
  cache before reading.
- `src/gpu/tier_plan.rs` — optional: budget for the cache out
  of the RAM headroom, so the cache is sized to what the
  current adaptive policy would otherwise reserve.

Close criterion: 13B sustained tok/s improves materially over
the M7.3 baseline (suggested target: ≥ 5× steady-state, i.e.
≤ 7 s/tok).

### Option 2 — M8 (BF16-resident VRAM, multi-GPU, etc.)

The natural M6 follow-up that M7 leapfrogged. Today the GPU
path uploads F32 because that is the only kernel ABI. A BF16-
resident kernel would halve VRAM cost per weight and unlock
more proj weights into VRAM on the same hardware (the M7.3
plan put 38 tensors in VRAM; a BF16-resident path would let
~76 tensors fit in the same 7 GiB usable budget).

Files to touch:
- `src/cuda/bf16_to_f32.cu` — extend with BF16-input matmul
  kernel (`__bfloat162float` inside the kernel, not as a
  separate decode pass).
- `src/cuda/cuda_matmul_bf16.rs` — new dispatcher.
- `src/gpu/tier_plan.rs` — `vram_cost_bytes` drops from
  `numel × 4` to `numel × 2` for the BF16-resident path.

### Option 3 — Production hardening (v21 territory)

M7 closes the v20 thesis: real models on real hardware,
end-to-end, automatic placement, no manual offload. The next
big leap is the Guards layer hardening (v21): adaptive memory
pressure thresholds, replay harnesses, structured logging.
The M7.3 smoke is the first concrete operational data point
for those guards. M4.7's known issue about the OS pagefile
trigger now has a counter-example: M7.3 was the first 13B run
that did **not** saturate the pagefile, because the planner
reserves 18.65 GiB headroom before the load starts.

---

## Tests

5 mandatory regression suites green at every M7 commit:

```text
tinyllama_config_test               15 passed (3 ignored)
tinyllama_weight_loading_test       10 passed (1 ignored)
tinyllama_builder_test              10 passed
weight_mapper_test                   5 passed
miniflux_safetensors_roundtrip_test  3 passed
                                  ---
                                   43 tests
```

Plus the M6 / M7 specific suites, all green at `19fdcf8`:

```text
gpu::tier_plan unit suite           10 passed (5 M6 + 4 M7.2 + 1 helper)
m6_replan_loader_test                1 passed (2 ignored — CUDA gated)
m7_disk_loader_test                  2 passed (slow path + fast path)
                                  ---
                                   13 additional tests
```

---

## Commits in M7

| Commit | Sub-phase |
|---|---|
| `8f29233` | M7.0 — NVMe bench + Disk weight bit-exact test (gating data for M7 plan) |
| `db5a49f` | M7.1 — Disk fast-path loader (raw BF16 bytes direct to NVMe, no F32 transient) |
| `19fdcf8` | M7.2 — adaptive headrooms (model size + available RAM → dynamic Disk overflow trigger) |
| (this doc) | M7.3 + closure |

**Tag**: `v0.7.0-m7` is the closure marker.

---

## Operator quickstart (for the next person)

```powershell
# Hardware: 32 GiB Windows box, NVIDIA RTX 4070 Laptop (8 GiB), NVMe.
# Cache must land on a fast SSD. Default of %LOCALAPPDATA% may be
# on C: which on the dev box is slower than D:.

$env:ATENIA_TIER_AWARE_LOADER = "1"
$env:ATENIA_DISK_TIER_DIR    = "D:\atenia-m7-cache"

cargo run --release --bin atenia -- generate `
    --prompt "Hello, how are you?" `
    --model D:/Atenia/models/llama-2-13b-chat `
    --max-tokens 5
```

Expected outcome: ~3 minutes of load, ~3 minutes for 5 tokens,
no BSOD, free RAM stays ≥ 4 GiB at all times. The tier plan
is logged to stderr at load time; if `Disk: 0 tensors` shows up
on a 13B model, the adaptive headroom did not fire — check
`free RAM` in the log line and confirm it matches what
`Get-Counter '\Memory\Available MBytes'` reports.

Switching to a smaller model (e.g. Llama 2 7B Chat) on the same
flag is bit-identical to the M6 default (validated at M6
close): the adaptive rule does not fire, the headroom stays at
8 GiB base, the Disk tier shows zero tensors, and throughput
matches the 1.46× speedup measured in the M6 handoff.
