# APX v13

APX v13 – Hybrid Execution Engine (H.E.E.)

APX v13 introduces Atenia Engine’s Hybrid Execution Engine: a fully
hardware-agnostic runtime layer capable of dynamically deciding where tensors
live, where kernels execute, and how memory is managed across CPU, GPU, RAM,
VRAM, and SSD.

This version unifies:

- hardware profiling
- tensor placement
- hybrid memory movement
- execution planning
- offloading
- checkpointing
- self-training and explainability

All components are deterministic, test-driven, vendor-independent, and designed
for stability under real runtime constraints.

# APX v13 – Hardware Profiler Scaffold

This directory contains the scaffold for **APX v13**, focused on a canonical hardware profiling layer for the engine. The goal of v13 is to provide a clean, well-typed foundation to describe and inspect the hardware where the engine runs (CPU, GPU, RAM, SSD), without committing yet to any concrete probing logic or external crates.

> Status: **13.0.0 – Scaffold + Types** (no logic, no external dependencies)

## Scope of this version (13.0.0)

In this initial subversion we only introduce:

- **Canonical types** to represent hardware capabilities and pressure.
- **A placeholder profiler struct** that owns a `GlobalHardwareSnapshot`.
- **Empty probe modules** for future implementations.
- **A minimal compilation test** to ensure the v13 API is wired into the crate.

No runtime logic, OS calls, or external libraries are implemented yet.

## Modules

### `mod.rs`

Exports the public v13 API surface inside the crate:

- `pub mod types;`
- `pub mod hardware_profiler;`
- `pub mod probe_cpu;`
- `pub mod probe_ram;`
- `pub mod probe_gpu;`
- `pub mod probe_ssd;`

The probe modules are empty placeholders for now.

### `types.rs`

Defines the canonical types used by the hardware profiler:

- **`BackendKind`**: enumerates GPU / accelerator backends:
  - `Cpu`, `Cuda`, `Rocm`, `Metal`, `OneApi`, `Vulkan`, `Unknown`.
- **`CpuCaps`**: basic CPU capabilities.
  - `physical_cores`, `logical_cores`, `simd` (vector of feature strings).
- **`GpuCaps`**: per-GPU capabilities and rough estimates.
  - `id`, `backend`, `vram_total_bytes`, `vram_free_bytes`,
    `bandwidth_gbps_est`, `compute_score_est`.
- **`RamCaps`**: total and free RAM.
  - `total_bytes`, `free_bytes`.
- **`SsdCaps`**: basic storage / cache directory and estimated performance.
  - `cache_dir`, `read_mb_s_est`, `write_mb_s_est`, `latency_ms_est`.
- **`ReliabilityStats`**: coarse-grained reliability metrics per device.
  - `ok_count`, `fail_count`, `last_error`, `last_error_epoch_ms`.
- **`PressureSnapshot`**: high-level pressure indicators.
  - `ram_pressure`, `vram_pressure`.
- **`GlobalHardwareSnapshot`**: the main struct that aggregates everything.
  - `timestamp_epoch_ms`, `cpu`, `gpus`, `ram`, `ssd`,
    `reliability_by_device`, `pressure`.

These types are intended to be stable, high-level primitives on top of which
future v13 subversions will build actual probing and scoring logic.

### `hardware_profiler.rs`

Defines a minimal owner for the global hardware snapshot:

- **`GlobalHardwareProfiler`**
  - Holds a single field: `snapshot: GlobalHardwareSnapshot`.
  - Marked with `#[allow(dead_code)]` for now, because there are no methods
    or call sites yet in 13.0.0.

Future subversions will add constructors, probing functions, and high-level
APIs to interact with this profiler.

### `probe_cpu.rs`, `probe_ram.rs`, `probe_gpu.rs`, `probe_ssd.rs`

These files are **intentionally empty** in 13.0.0.

They are placeholders for future implementations that will:

- Inspect CPU topology and SIMD capabilities.
- Inspect RAM capacity, NUMA, and pressure.
- Inspect GPUs, VRAM, bandwidth, and approximate compute score.
- Inspect SSD / storage behavior relevant to engine caching.

For now, they only exist so the module structure is stable and ready.

## Tests

### `tests/apx13_scaffold_test.rs`

This file contains a **minimal compilation test** for the v13 scaffold:

- It imports `BackendKind` from `atenia_engine::v13::types`.
- It constructs a `BackendKind::Unknown` value.
- It matches on the enum to ensure pattern matching compiles and links.

The purpose of this test is not to validate behavior, but to:

- Ensure that the `v13` module is exported from `src/lib.rs`.
- Ensure that the `types` module compiles and is usable from tests.
- Guard against regressions where v13 types are renamed or removed.

You can run this specific test target with:

```bash
cargo test --test apx13_scaffold_test
```

## Future subversions (high level roadmap)

The following is an informal, high-level roadmap for next APX v13 subversions:

- **13.1.x – Basic OS-backed probes**
  - Implement minimal CPU, RAM, GPU, and SSD probes using standard Rust and
    widely available system APIs.
  - Populate `GlobalHardwareSnapshot` with real data.

- **13.2.x – Scoring and pressure models**
  - Introduce scoring functions ("how good is this device for APX?").
  - Add better pressure estimation and historical snapshots.

- **13.3.x+ – Cross-platform refinements and integration**
  - Improve coverage across OSes and GPU backends.
  - Integrate with higher APX versions that make scheduling / placement
    decisions using this hardware information.

This README should be kept up to date as new subversions of v13 are
implemented.

## APX 13.1.0 – Tensor Placement Engine v1 (Decision Model Only)

Version **13.1.0** introduces the first version of the **Tensor Placement
Engine**, a *pure decision model* that chooses where a tensor **should** live,
without actually moving memory or touching any backend.

Key constraints for 13.1.0:

- No real tensors are moved.
- No GPU / RAM / SSD operations are performed.
- No new crates are added.
- No backend-specific assumptions (no CUDA-only logic, etc.).
- All decisions are based only on abstract information from
  `GlobalHardwareSnapshot` (APX 13.0.0).

### `placement_types.rs`

Defines the core types used by the placement engine:

- **`PlacementTarget`**
  - Possible targets: `Cpu`, `Gpu`, `Ram`, `Vram`, `Ssd`.
- **`PlacementDecision`**
  - `target: PlacementTarget` – where the tensor is recommended to live.
  - `reason: String` – human-readable explanation for debugging and logs.
- **`TensorProfile`**
  - `num_elements: u64` – number of elements in the tensor.
  - `element_size_bytes: u32` – size of each element in bytes.
  - `estimated_compute_cost: Option<f32>` – abstract compute cost units.
  - Method `total_size_bytes()` returns `num_elements * element_size_bytes`.

These types are backend-agnostic and are meant to be stable inputs and
outputs for placement policies.

### `tensor_placement.rs`

Implements the **v1 decision engine**:

- **`TensorPlacementEngine`**
  - Stateless type that exposes:

```rust
pub fn decide(
    _tensor: &TensorProfile,
    hw: &GlobalHardwareSnapshot,
) -> PlacementDecision
```

Current heuristic (v1):

1. Start with a safe default decision:
   - `target = PlacementTarget::Cpu`
   - `reason = "Default CPU fallback"`.
2. If there is **no GPU available** (`hw.gpus.is_empty()`):
   - Stay on CPU with `reason = "No GPU detected"`.
3. Read memory pressure signals (best-effort):
   - `vram_pressure = hw.pressure.vram_pressure.unwrap_or(1.0)`.
   - `ram_pressure = hw.pressure.ram_pressure.unwrap_or(0.0)`.
4. Apply a simple, deterministic policy:
   - If `vram_pressure < 0.80` → choose `PlacementTarget::Vram` with
     `reason = "Low VRAM pressure"`.
   - Else, if `ram_pressure < 0.80` → choose `PlacementTarget::Ram` with
     `reason = "High VRAM pressure, RAM available"`.
   - Else → fallback to `PlacementTarget::Cpu` with
     `reason = "High memory pressure, CPU fallback"`.

This version focuses purely on logic and debuggability; no hardware calls,
no side effects, no tensor movements.

### Tests: `tests/tensor_placement_v1_test.rs`

The test file `tests/tensor_placement_v1_test.rs` validates the behavior of
the v1 placement engine using a fully synthetic
`GlobalHardwareSnapshot` (no real hardware probing):

- Helper: `mock_hw_snapshot(has_gpu, vram_pressure, ram_pressure)`
  - Builds a `GlobalHardwareSnapshot` with:
    - Minimal `CpuCaps`, `RamCaps`, `SsdCaps`.
    - A single fake GPU (`GpuCaps`) when `has_gpu == true`.
    - Pressure information filled from the function arguments.

Coverage tests:

- `placement_cpu_when_no_gpu`
  - `has_gpu = false` → expects `PlacementTarget::Cpu`.
- `placement_vram_when_low_pressure`
  - `has_gpu = true`, `vram_pressure = Some(0.3)`, `ram_pressure = Some(0.1)`
    → expects `PlacementTarget::Vram`.
- `placement_ram_when_vram_high`
  - `has_gpu = true`, `vram_pressure = Some(0.95)`, `ram_pressure = Some(0.2)`
    → expects `PlacementTarget::Ram`.

How to run only these tests:

```bash
cargo test --test tensor_placement_v1_test
```

These tests are fully deterministic, do not depend on real hardware, and
validate only the decision model, not memory movement or backend behavior.

## APX 13.1.1 – Cost + Reliability-aware Multi-GPU Placement

Version **13.1.1** extends the tensor placement engine with a
**cost- and reliability-aware decision model** that supports multiple GPUs.
The engine still does not move any real tensors or allocate memory; it only
decides *where* a tensor should live.

Key additions in 13.1.1:

- The decision can now specify an optional `device_id` (e.g. `"gpu0"`).
- GPU selection takes into account historical reliability statistics.
- The model considers tensor size, estimated compute cost, and memory
  pressure (RAM/VRAM).
- Deterministic selection in presence of multiple GPUs.

### Updated `PlacementDecision`

In `placement_types.rs`, `PlacementDecision` was extended to:

```rust
pub struct PlacementDecision {
    pub target: PlacementTarget,
    pub device_id: Option<String>, // e.g. "gpu0" when target is Vram/Gpu
    pub reason: String,            // human-readable, for debugging/logs
}
```

- `device_id` is `None` for CPU/RAM/SSD targets.
- For GPU/VRAM targets, `device_id` identifies the chosen GPU.

### Reliability and multi-GPU helpers

In `tensor_placement.rs`, several internal helpers were introduced:

- `clamp01(x: f32) -> f32`
  - Clamps a floating-point value to the range `[0.0, 1.0]`.
- `reliability_score(stats: Option<&ReliabilityStats>) -> f32`
  - If there are no stats → score = `1.0`.
  - Else, `score = ok / (ok + fail)` with safe handling when the denominator
    is zero.
  - The score is clamped to `[0.0, 1.0]`.
  - If `last_error_epoch_ms` is present, apply a fixed penalty
    (`score *= 0.85`), without depending on real time.
- `best_gpu(hw: &GlobalHardwareSnapshot) -> Option<(&GpuCaps, f32)>`
  - Iterates over all GPUs and selects the best one according to:
    1. Highest reliability score.
    2. If tied, highest `compute_score_est`.
    3. If still tied, lexicographically smallest `id`.

These helpers keep the behavior deterministic and vendor-agnostic.

### Extended placement heuristic

`TensorPlacementEngine::decide` was upgraded to use:

- `tensor.total_size_bytes()` for tensor size decisions.
- `tensor.estimated_compute_cost` when available.
- Memory pressure from `hw.pressure.ram_pressure` and
  `hw.pressure.vram_pressure`.
- Multi-GPU selection via `best_gpu` and reliability scores.

The main constants controlling the policy are:

```rust
const VRAM_SAFE: f32 = 0.80;
const RAM_SAFE: f32 = 0.85;
const TENSOR_HUGE_BYTES: u64 = 512 * 1024 * 1024;
const COMPUTE_HEAVY: f32 = 100.0; // abstract units
```

High-level behavior:

1. If there are **no GPUs**, the decision is CPU with reason
   `"No GPU detected"`.
2. Choose the best GPU using the reliability-aware `best_gpu` helper.
3. If VRAM pressure is safe, the tensor is not huge, and the tensor is
   compute-heavy (or the GPU is reasonably reliable), place it in VRAM on
   the selected GPU and set `device_id` accordingly.
4. If the tensor is huge and RAM pressure is acceptable, prefer RAM even if
   VRAM is moderately free.
5. If VRAM is under pressure but RAM is fine, place the tensor in RAM.
6. If memory pressure is high everywhere, fall back to CPU with a clear
   reason.

All `reason` strings are human-readable and in English, for example:

- `"Selected GPU gpu1 (reliability=0.90) and VRAM pressure is safe"`
- `"Tensor is huge and RAM pressure is acceptable; placing tensor in RAM"`
- `"VRAM pressure is high; placing tensor in RAM"`
- `"System memory pressure is high; falling back to CPU"`

### Tests: `tests/tensor_placement_v1_1_test.rs`

The file `tests/tensor_placement_v1_1_test.rs` validates the new behavior
with deterministic, synthetic scenarios and no real hardware:

- Utility helpers:
  - `make_gpu(id, compute_score_est)` – constructs `GpuCaps` with a given id
    and compute score.
  - `make_stats(ok, fail, has_recent_error)` – constructs `ReliabilityStats`
    with controlled counters and an optional recent error.
  - `mock_hw_snapshot_multi_gpu(...)` – builds a `GlobalHardwareSnapshot`
    with multiple GPUs, pressures, and reliability map.
  - `make_tensor(...)` – constructs `TensorProfile` instances used in tests.

Covered scenarios:

- **Best GPU by reliability**
  - Two GPUs with different reliability; expects Vram placement on the more
    reliable GPU and the corresponding `device_id`.
- **Tie-breaker by compute score**
  - Same reliability, different `compute_score_est`; expects the GPU with
    higher compute score to be selected.
- **Huge tensor prefers RAM when pressure allows**
  - Tensor size above `TENSOR_HUGE_BYTES` and RAM pressure low;
    expects `PlacementTarget::Ram`.
- **Compute-heavy prefers GPU when safe**
  - High `estimated_compute_cost` and safe VRAM pressure with a reliable
    GPU; expects `PlacementTarget::Vram` and the correct `device_id`.
- **High pressure forces CPU**
  - Both VRAM and RAM under high pressure; expects CPU fallback.

You can run only these tests with:

```bash
cargo test --test tensor_placement_v1_1_test -- --nocapture
```

As with previous versions, these tests are deterministic, do not depend on
real hardware, and validate only the decision logic.

## APX 13.1.2 – SSD-aware Last-resort Placement (Policy Only)

Version **13.1.2** completes the v1 tensor placement engine by explicitly
introducing **SSD** as a last-resort target. This remains strictly
policy-only: it does not write to SSD, does not implement any cache, and does
not move real tensors.

Key ideas in 13.1.2:

- SSD is only considered when the system is under very high memory pressure
  or when a huge, cold tensor would put RAM at risk.
- Decisions are deterministic and use simple, well-named constants.
- The model still does not allocate memory or touch real hardware.

### New SSD-related constants

In `tensor_placement.rs`, the following constants were added:

```rust
const VRAM_CRITICAL: f32 = 0.95;
const RAM_CRITICAL: f32 = 0.95;
const RAM_VERY_HIGH: f32 = 0.90;
const SSD_LAST_RESORT_ALLOWED: bool = true;
```

- `VRAM_CRITICAL` / `RAM_CRITICAL` represent thresholds where pressure is
  considered "critical".
- `RAM_VERY_HIGH` is a softer threshold used when RAM is already close to
  being problematic but not yet fully critical.
- `SSD_LAST_RESORT_ALLOWED` is a boolean flag that keeps the SSD behavior
  centralized and easy to tweak in future subversions.

### Extended SSD placement rules

The existing decision logic is extended with two SSD-related rules.

1. **Both VRAM and RAM critical + huge tensor**

   When:

   - `vram_pressure >= VRAM_CRITICAL`
   - `ram_pressure >= RAM_CRITICAL`
   - `tensor.total_size_bytes() > TENSOR_HUGE_BYTES`

   then the engine chooses SSD as last resort:

   - `target = PlacementTarget::Ssd`
   - `device_id = None`
   - `reason = "VRAM and RAM are under critical pressure; placing huge tensor on SSD as last resort"`

2. **Huge, not compute-heavy tensor with very high RAM pressure**

   When RAM is already very high and the tensor is huge but not
   compute-heavy, the engine prefers SSD over RAM to reduce OOM risk:

   - Conditions (simplified):
     - `tensor.total_size_bytes() > TENSOR_HUGE_BYTES`
     - `ram_pressure > RAM_VERY_HIGH`
     - `estimated_compute_cost` is low or `None`
   - Decision:
     - `target = PlacementTarget::Ssd`
     - `device_id = None`
     - `reason = "RAM pressure is very high and tensor is huge but not compute-heavy; placing tensor on SSD as last resort"`

In all other cases, the previously defined rules continue to apply:

- Compute-heavy tensors with safe VRAM still prefer GPU/VRAM.
- Huge tensors with acceptable RAM pressure but without extreme conditions
  prefer RAM.
- CPU remains the fallback when system pressure is high and SSD is not
  warranted by the policy.

### Tests: `tests/tensor_placement_v1_2_test.rs`

The file `tests/tensor_placement_v1_2_test.rs` validates the SSD-related
policy with deterministic, synthetic scenarios:

- Helper functions:
  - `make_gpu(id, compute_score_est)` – creates a single `GpuCaps` instance.
  - `mock_hw_snapshot_single_gpu(vram_pressure, ram_pressure)` – builds a
    minimal `GlobalHardwareSnapshot` with one GPU and specified pressure.
  - `make_tensor(...)` and `huge_tensor_elements_for_4b()` – construct
    tensors of controlled size, including a "huge" tensor above
    `TENSOR_HUGE_BYTES`.

Covered scenarios:

- **SSD when VRAM and RAM critical and tensor huge**
  - `vram_pressure = 0.98`, `ram_pressure = 0.97`, huge tensor.
  - Expects `PlacementTarget::Ssd`, `device_id = None`, and a non-empty
    English `reason`.
- **CPU when critical but tensor small**
  - Same critical pressures but small tensor.
  - Expects `PlacementTarget::Cpu`, `device_id = None`, and a non-empty
    `reason`.
- **RAM when VRAM critical but RAM safe**
  - `vram_pressure` critical, `ram_pressure` low/safe.
  - Expects `PlacementTarget::Ram`, `device_id = None`, and a non-empty
    `reason`.
- **GPU/VRAM still wins when compute-heavy and VRAM safe**
  - Compute-heavy tensor, `vram_pressure` safe, `ram_pressure` high but not
    critical.
  - Expects `PlacementTarget::Vram`, `device_id = Some("gpu0")`, and a
    non-empty `reason`.

You can run only these tests with:

```bash
cargo test --test tensor_placement_v1_2_test -- --nocapture
```

As with the previous versions, these SSD-related tests are deterministic,
CI-safe, and do not touch any real hardware or storage.

## APX 13.2.0 – Hybrid Memory System (Scaffold + Logical Moves)

Version **13.2.0** introduces a vendor-agnostic **3-tier hybrid memory
abstraction** together with a logical migration pipeline. This subversion does
not move any real tensor bytes; it only tracks logical residency and plans
state transitions.

### Core memory types

Defined in `memory_types.rs`:

- **`MemoryTier`**
  - Variants: `Vram`, `Ram`, `Ssd`, `Cpu`.
  - `Cpu` is allowed as a safe fallback for compute-only / unmanaged tensors.
- **`TensorId`**
  - Wrapper around `String` identifying a tensor logically.
- **`MemoryFootprint`**
  - `bytes: u64` – logical size of a tensor in bytes.
- **`TierStatus`**
  - `total_bytes`, `free_bytes`, `pressure` – optional telemetry per tier.
- **`MemorySnapshot`**
  - Aggregates `TierStatus` for `vram`, `ram`, and `ssd`.
- **`MoveError`**
  - `Unsupported(String)`, `IoError(String)`, `BackendUnavailable(String)` –
    used to report logical and IO-related issues in a backend-agnostic way.
- **`MovePlan`**
  - `from: MemoryTier`, `to: MemoryTier`, `reason: String` – describes a
    single planned logical transition between tiers.
- **`TensorResidence`**
  - `id: TensorId`, `tier: MemoryTier`, `footprint: MemoryFootprint` – tracks
    the current logical location of a tensor.

All strings and reasons are in English to keep logs and errors consistent.

### SSD cache manager

Defined in `ssd_cache.rs`:

- **`SsdCache`**
  - Holds a cache directory path.
  - Methods:
    - `new(dir: &str) -> Self` – constructs a cache handle.
    - `ensure_dir(&self) -> Result<(), MoveError>` – calls `create_dir_all`
      and maps any filesystem errors to `MoveError::IoError` with a clear
      message; also rejects empty paths as `Unsupported`.
    - `dir(&self) -> &str` – returns the underlying directory path.

This remains a pure directory manager: it does not read or write any tensor
buffers in 13.2.0.

### HybridMemoryManager – logical state machine

Defined in `hybrid_memory.rs`:

- **`HybridMemoryManager`**
  - Fields:
    - `tensors: HashMap<String, TensorResidence>` – logical registry keyed by
      tensor id string.
    - `cache: SsdCache` – SSD cache directory manager.
  - Methods:
    - `new(cache_dir: &str) -> Self`
      - Initializes an empty registry and an `SsdCache` for the given
        directory.
    - `register_tensor(&mut self, id: &str, bytes: u64, initial: MemoryTier)`
      - Inserts a `TensorResidence` with the given id, footprint, and tier.
    - `get_tier(&self, id: &str) -> Option<MemoryTier>`
      - Returns the current tier for a registered tensor, or `None` if the
        tensor is unknown.
    - `plan_move(&self, id: &str, target: MemoryTier, snapshot: &MemorySnapshot)
       -> Result<MovePlan, MoveError>`
      - If the tensor is not registered →
        `Err(MoveError::Unsupported("Tensor not registered".to_string()))`.
      - If `from == target` → returns a `MovePlan` with reason
        `"Already in target tier"`.
      - If `target == MemoryTier::Ssd` → calls `cache.ensure_dir()` before
        allowing the plan.
      - Otherwise, returns a generic plan that describes the logical move.
    - `apply_move(&mut self, id: &str, plan: &MovePlan) -> Result<(), MoveError>`
      - Updates only the logical `tier` of the tensor; **no real data
        movement** is performed.

This design prepares a clean API for 13.2.1+, where real buffers and backend
integration can be layered on top of the same logical interface.

### Tests: `tests/hybrid_memory_test.rs`

The test suite `hybrid_memory_test.rs` validates the hybrid memory scaffold
using only deterministic logic and filesystem operations for the cache
directory:

- Helper:
  - `empty_snapshot()` – builds a `MemorySnapshot` with all fields set to
    `None`, representing a best-effort, neutral view of the tiers.

Covered scenarios:

- `register_and_query_tier`
  - Registers a tensor in `MemoryTier::Vram` and verifies that
    `get_tier` returns `Some(Vram)`.
- `plan_and_apply_move_ram`
  - Registers a tensor in `Vram`, plans a move to `Ram`, applies the move,
    and verifies the updated tier.
- `plan_to_ssd_ensures_cache_dir`
  - Plans a move to `Ssd` for a registered tensor and checks that the cache
    directory (e.g. `./.atenia_cache_test`) is created on disk; performs
    best-effort cleanup before and after the test.
- `already_in_target_tier_is_ok`
  - Plans a move to the same tier and verifies that the reason is exactly
    `"Already in target tier"`.
- `unknown_tensor_returns_error`
  - Calls `plan_move` on an unknown tensor id and expects a
    `MoveError::Unsupported("Tensor not registered".to_string())`.

You can run only these tests with:

```bash
cargo test --test hybrid_memory_test -- --nocapture
```

As with the rest of v13, these tests are deterministic, CI-safe, and do not
depend on GPUs, real tensor buffers, or vendor-specific APIs.

## APX 13.2.1 – Backed Moves v1 (RAM Buffers + SSD Blobs)

Version **13.2.1** upgrades the hybrid memory system from purely logical
placement to **backed moves** between RAM and SSD for tensor data. VRAM and
CPU remain logical tiers for now; no real GPU transfers are performed.

Key properties:

- RAM tier holds real bytes in-process (`Vec<u8>`).
- SSD tier persists bytes as blob files under the cache directory.
- Moves between RAM and SSD migrate the actual bytes.
- No new crates, no async, and no GPU involvement yet.

### Extended storage model

In `memory_types.rs`, the following backing enum was introduced:

```rust
pub enum StorageBacking {
    None,
    Ram(Vec<u8>),
    SsdFile { path: String },
}
```

`TensorResidence` now tracks both the logical tier and the backing:

```rust
pub struct TensorResidence {
    pub id: TensorId,
    pub tier: MemoryTier,
    pub footprint: MemoryFootprint,
    pub backing: StorageBacking,
}
```

`MemoryFootprint` gained a helper to validate consistency between the
declared footprint and the actual data length:

```rust
impl MemoryFootprint {
    pub fn validate_len(&self, len: usize) -> Result<(), MoveError> { ... }
}
```

If there is a mismatch, it returns `MoveError::Unsupported` with an explicit
English error message.

### SSD blob utilities

In `ssd_cache.rs`, the SSD cache was extended with blob helpers:

- `blob_path(&self, tensor_id: &str) -> String`
  - Produces a sanitized path like `"{dir}/tensor_<id>.bin"`, where the id is
    restricted to `[a-zA-Z0-9_-]` (other characters are replaced by `_`).
- `write_blob(&self, path: &str, data: &[u8]) -> Result<(), MoveError>`
- `read_blob(&self, path: &str) -> Result<Vec<u8>, MoveError>`
- `delete_blob(&self, path: &str) -> Result<(), MoveError>`
  - Missing files are treated as success (best-effort cleanup).

All IO errors are mapped into `MoveError::IoError("...")` with clear
messages and no panics.

### HybridMemoryManager with real backing

`HybridMemoryManager` in `hybrid_memory.rs` was extended to understand and
operate on real RAM/SSD data:

- Registration:
  - `register_tensor_with_data(&mut self, id: &str, data: Vec<u8>, initial: MemoryTier)
     -> Result<(), MoveError>`
    - Sets the footprint based on `data.len()`.
    - If `initial == MemoryTier::Ram` → stores `StorageBacking::Ram(data)`.
    - If `initial == MemoryTier::Ssd` → `ensure_dir`, writes a blob, and
      stores `StorageBacking::SsdFile { path }`.
    - If `initial == Cpu` or `Vram` → stores `StorageBacking::None`.
  - `register_tensor(&mut self, id: &str, bytes: u64, initial: MemoryTier)`
    - Keeps backward compatibility: registers footprint with
      `StorageBacking::None`.

- Planning moves:
  - `plan_move` now rejects moves to SSD when there is no backing data:
    - Returns `MoveError::Unsupported("Cannot move tensor to SSD without backing data")`.
  - Continues to return `"Already in target tier"` when `from == to`.

- Applying moves (`apply_move`):
  - **RAM → SSD**
    - For `(MemoryTier::Ram, StorageBacking::Ram(data), MemoryTier::Ssd)`:
      - Ensures the cache directory exists.
      - Computes a blob path via `blob_path`.
      - Writes the bytes with `write_blob`.
      - Clears the in-memory `Vec<u8>` and sets backing to
        `StorageBacking::SsdFile { path }`.
      - Updates the logical tier to `MemoryTier::Ssd`.
  - **SSD → RAM**
    - For `(MemoryTier::Ssd, StorageBacking::SsdFile { path }, MemoryTier::Ram)`:
      - Reads bytes with `read_blob`.
      - Validates size via `footprint.validate_len(bytes.len())`.
      - Deletes the blob with `delete_blob` (best-effort).
      - Sets backing to `StorageBacking::Ram(bytes)` and tier to `MemoryTier::Ram`.
  - **CPU / VRAM moves**
    - If `backing == StorageBacking::None`, moves are logical tier updates.
    - If backing is `Ram` or `SsdFile` and the target is `Cpu`/`Vram`, the
      backing is preserved and only the tier is updated. Real VRAM transfers
      will be added in future subversions.
  - Other combinations (e.g. Ram→Ram, Ssd→Ssd) are treated as simple tier
    updates.

For tests, `HybridMemoryManager` exposes two small helpers:

- `backing_for_test(&self, id: &str) -> Option<&StorageBacking>` – to assert
  which backing is currently stored.
- `set_footprint_bytes_for_test(&mut self, id: &str, bytes: u64)` – used only
  to simulate invalid states (e.g. length mismatches) in negative tests.

### Tests: `tests/hybrid_memory_backed_moves_test.rs`

This test file verifies real data movement between RAM and SSD using a
dedicated cache directory `./.atenia_cache_test_backed` and best-effort
cleanup:

- **`ram_to_ssd_moves_bytes`**
  - Registers a tensor in RAM with `register_tensor_with_data` and a known
    byte pattern.
  - Plans a move to SSD and applies it.
  - Asserts that the tier is `MemoryTier::Ssd`.
  - Verifies that the SSD blob exists and that `read_blob` returns exactly the
    same bytes.

- **`ssd_to_ram_moves_bytes_and_deletes_file`**
  - Registers a tensor initially in SSD with `register_tensor_with_data`.
  - Confirms the blob file exists.
  - Plans a move to RAM and applies it.
  - Asserts that the tier is `MemoryTier::Ram` and that
    `backing_for_test` reports `StorageBacking::Ram` with the original bytes.
  - Checks that the blob file has been deleted.

- **`cannot_move_none_backing_to_ssd`**
  - Registers a tensor with `register_tensor` (footprint only, no backing
    data).
  - Attempts to plan a move to SSD and expects
    `MoveError::Unsupported("Cannot move tensor to SSD without backing data")`.

- **`length_mismatch_is_error`**
  - Registers a tensor with real SSD backing using `register_tensor_with_data`.
  - Uses `set_footprint_bytes_for_test` to intentionally shrink the
    footprint and create an inconsistent state.
  - Plans a move to RAM and applies it.
  - Expects `MoveError::Unsupported` with a message containing
    `"Byte length mismatch"`, confirming that `validate_len` is enforced.

You can run only these tests with:

```bash
cargo test --test hybrid_memory_backed_moves_test -- --nocapture
```

Together with 13.2.0, this subversion provides a fully tested, deterministic
hybrid memory scaffold where RAM and SSD tiers are backed by real bytes,
while VRAM remains a logical target for future work.

## APX 13.2.2 – VRAM Backing Adapter (Best-effort, Fallback-first)

Version **13.2.2** extends the hybrid memory system with a
**VRAM backing adapter**. The goal is to provide a *best-effort*,
vendor-agnostic interface to move bytes between RAM/SSD and a logical VRAM
backend, while keeping behavior safe and deterministic on machines with
**no GPU** by always degrading cleanly to RAM.

Key properties:

- No direct dependency on CUDA, ROCm, or any vendor-specific API.
- No new crates; the adapter is a pure trait that can be implemented by
  higher layers.
- If VRAM is not available, the system **never panics** and falls back to
  RAM placement.
- All error strings remain in English and are surfaced as `MoveError`
  variants.

### VRAM adapter interface

Defined in `vram_adapter.rs`:

- **`VramAdapter` trait**
  - A minimal interface describing a VRAM-like backend:
    - `is_available(&self) -> bool`
      - Returns `true` if the adapter is ready to send/receive bytes.
    - `upload(&self, id: &str, data: &[u8]) -> Result<(), MoveError>`
      - Uploads a byte slice under a given logical id.
    - `download(&self, id: &str) -> Result<Vec<u8>, MoveError>`
      - Retrieves the bytes previously stored under that id.
    - `free(&self, id: &str) -> Result<(), MoveError>`
      - Releases any VRAM-side resources associated with the id.
  - All methods are synchronous and return `MoveError` on failure; no
    panics, no `unwrap`, no assumptions about hardware.

- **`NullVramAdapter`**
  - Default, **safe** implementation used by `HybridMemoryManager::new`.
  - Behavior:
    - `is_available() -> false`.
    - `upload`, `download`, and `free` all return
      `MoveError::BackendUnavailable("VRAM adapter not available".to_string())`.
  - This keeps the system fully functional on machines without GPUs: any
    attempt to use VRAM is treated as best-effort and degraded to RAM.

Higher layers (or tests) can provide their own `VramAdapter` implementations
without changing the core hybrid memory logic.

### Extended storage model: `VramHandle`

In `memory_types.rs`, the `StorageBacking` enum was extended with a new
variant to represent VRAM-backed tensors:

```rust
pub enum StorageBacking {
    None,
    Ram(Vec<u8>),
    SsdFile { path: String },
    VramHandle { key: String },
}
```

- `VramHandle { key }` stores a logical handle used by `VramAdapter`.
- The `key` is typically the tensor id string; the adapter is free to
  interpret it as needed.
- Existing RAM and SSD behavior from 13.2.1 remains unchanged.

### HybridMemoryManager with VRAM backing

In `hybrid_memory.rs`, `HybridMemoryManager` was extended to own a VRAM
adapter and to handle backed moves involving VRAM:

- **Structure**
  - Fields now include:
    - `tensors: HashMap<String, TensorResidence>` – unchanged.
    - `cache: SsdCache` – unchanged.
    - `vram: Box<dyn VramAdapter + Send + Sync>` – new field.
  - Constructors:
    - `new(cache_dir: &str) -> Self`
      - Uses `NullVramAdapter` internally; safe default when no GPU is
        available.
    - `new_with_vram(cache_dir: &str, vram: Box<dyn VramAdapter + Send + Sync>) -> Self`
      - Allows tests or higher layers to inject a concrete VRAM adapter.

- **Planning moves with VRAM-aware degradation**
  - `plan_move` now checks whether VRAM is actually available:
    - If the target tier is `MemoryTier::Vram` and `vram.is_available()` is
      `false`, the plan is degraded to RAM:
      - `to = MemoryTier::Ram`.
      - `reason = "VRAM unavailable; degrading placement to RAM"`.
  - This keeps behavior deterministic on machines without GPUs while still
    exercising the same code paths.

- **Applying moves involving VRAM**

  The `apply_move` method gained several new match arms for VRAM-backed
  transitions, all respecting the best-effort and fallback-first design:

  - **RAM → VRAM**
    - Pattern: `(MemoryTier::Ram, StorageBacking::Ram(data), MemoryTier::Vram)`.
    - If `vram.is_available() == false`, the function returns `Ok(())` and
      leaves the tensor in RAM (the plan should already have degraded to
      RAM, but the code is defensive).
    - If VRAM is available:
      - Calls `vram.upload(id, data)`.
      - On success, clears the `Vec<u8>` and updates the backing to
        `StorageBacking::VramHandle { key: id.to_string() }`.
      - Updates the logical tier to `MemoryTier::Vram`.

  - **VRAM → RAM**
    - Pattern: `(MemoryTier::Vram, StorageBacking::VramHandle { key }, MemoryTier::Ram)`.
    - If `vram.is_available() == false`, returns
      `MoveError::BackendUnavailable("VRAM not available for download".to_string())`.
    - Otherwise:
      - Calls `vram.download(key)` to retrieve bytes.
      - Validates the length via `footprint.validate_len(bytes.len())`.
      - Calls `vram.free(key)` as a best-effort cleanup (ignoring errors).
      - Sets the backing to `StorageBacking::Ram(bytes)` and tier to
        `MemoryTier::Ram`.

  - **SSD → VRAM**
    - Pattern: `(MemoryTier::Ssd, StorageBacking::SsdFile { path }, MemoryTier::Vram)`.
    - If VRAM is not available, returns `Ok(())` and leaves the tensor as-is
      (planning should have degraded earlier).
    - Otherwise:
      - Reads bytes from SSD with `cache.read_blob(path)`.
      - Validates length with `footprint.validate_len(bytes.len())`.
      - Uploads to VRAM via `vram.upload(id, &bytes)`.
      - Attempts to delete the SSD blob with `cache.delete_blob(path)` as
        best-effort cleanup.
      - Updates backing to `StorageBacking::VramHandle { key: id.to_string() }`
        and tier to `MemoryTier::Vram`.

  - **VRAM → SSD**
    - Pattern: `(MemoryTier::Vram, StorageBacking::VramHandle { key }, MemoryTier::Ssd)`.
    - If VRAM is not available, returns
      `MoveError::BackendUnavailable("VRAM not available for download".to_string())`.
    - Otherwise:
      - Downloads bytes via `vram.download(key)`.
      - Validates length with `footprint.validate_len(bytes.len())`.
      - Ensures the SSD cache directory exists.
      - Computes a blob path and writes it with `cache.write_blob`.
      - Calls `vram.free(key)` as best-effort cleanup.
      - Updates backing to `StorageBacking::SsdFile { path }` and tier to
        `MemoryTier::Ssd`.

  - **Logical CPU/VRAM moves without backing**
    - When backing is `StorageBacking::None`, moves involving `Cpu`/`Vram`
      remain purely logical tier updates, as in 13.2.0–13.2.1.

All new branches continue to use `MoveError` for error reporting and do not
introduce any `unwrap` or `expect` calls.

### Tests: `tests/hybrid_memory_vram_adapter_test.rs`

This test file validates the VRAM adapter integration using a **fake VRAM
adapter** and per-test cache directories. It remains fully deterministic and
does not require a real GPU.

- **`FakeVramAdapter`**
  - Implements `VramAdapter` using an in-memory
    `Mutex<HashMap<String, Vec<u8>>>`.
  - `is_available()` always returns `true`.
  - `upload`, `download`, and `free` map mutex poisoning to
    `MoveError::BackendUnavailable` but otherwise operate purely in memory.

Covered scenarios:

- **`ram_to_vram_and_back_roundtrip`**
  - Uses a dedicated cache directory (e.g.
    `./.atenia_cache_test_vram_ram_roundtrip`) with best-effort cleanup.
  - Registers a tensor in RAM with a known byte pattern.
  - Plans and applies a move to VRAM using `FakeVramAdapter`.
  - Asserts that the tier becomes `MemoryTier::Vram` and that the backing is
    `StorageBacking::VramHandle`.
  - Then plans and applies a move back to RAM.
  - Verifies that the tier returns to `MemoryTier::Ram` and the backing is
    `StorageBacking::Ram` with exactly the original bytes.

- **`ssd_to_vram_via_fake_adapter`**
  - Uses its own cache directory
    (`./.atenia_cache_test_vram_ssd_roundtrip`).
  - Registers a tensor initially in SSD using `register_tensor_with_data`.
  - Plans and applies a move SSD → VRAM.
  - Asserts that the tier is `MemoryTier::Vram` with `VramHandle` backing.
  - Then plans and applies a move VRAM → RAM.
  - Confirms that the final backing is `StorageBacking::Ram` and that the
    bytes match the original payload.

- **`vram_unavailable_degrades_to_ram`**
  - Uses the default constructor `HybridMemoryManager::new`, which is wired
    to `NullVramAdapter`.
  - Registers a tensor in RAM and builds a neutral `MemorySnapshot`.
  - Calls `plan_move` with target `MemoryTier::Vram`.
  - Asserts that the resulting plan has `to = MemoryTier::Ram` and a reason
    containing `"VRAM unavailable"`.
  - Applies the plan and verifies that the tensor remains in RAM with RAM
    backing and unchanged bytes.

All tests use best-effort directory cleanup (`remove_dir_all`) and avoid
any reliance on real GPUs or platform-specific features, keeping the suite
CI-safe and reproducible.

## APX 13.2.3 – Compression Scaffold (On-demand, Optional)

Version **13.2.3** introduces a **compression scaffold** on top of the
hybrid memory system. The goal is to prepare hooks for future
compression-aware policies **without** changing default behavior:

- Compression is **optional** and **off by default**.
- There are **no new crates** (no `flate2`, `zstd`, etc.).
- The implementation is deterministic and fully test-driven.
- All non-code text (errors, comments) remains in English.

### Compression types

In `memory_types.rs`, two new types describe compression state:

- **`CompressionKind`**
  - `None` – no compression.
  - `Rle` – a simple run-length encoding used as a toy, deterministic codec.
- **`CompressionMeta`**
  - `kind: CompressionKind` – which compression was applied.
  - `original_bytes: u64` – uncompressed logical byte length.

The SSD backing variant was extended to carry optional metadata:

```rust
pub enum StorageBacking {
    None,
    Ram(Vec<u8>),
    SsdFile { path: String, compression: Option<CompressionMeta> },
    VramHandle { key: String },
}
```

- `compression: Some(meta)` indicates that bytes on disk may be compressed
  and provides the information required to decompress.
- `compression: None` is reserved for legacy or manually written blobs.

### RLE codec (`compression.rs`)

A new module `compression.rs` implements a minimal **run-length encoding
(RLE)** codec:

- **`rle_compress(input: &[u8]) -> (Vec<u8>, CompressionMeta)`**
  - Encodes the input as `(count: u8, value: u8)` pairs.
  - Returns the compressed bytes together with:
    - `CompressionMeta { kind: CompressionKind::Rle, original_bytes }`.
- **`rle_decompress(input: &[u8], meta: &CompressionMeta) -> Result<Vec<u8>, MoveError>`**
  - Decodes the RLE stream back to raw bytes.
  - On malformed input (odd length, zero count, length mismatch), it
    returns `MoveError::Unsupported("Invalid RLE stream ...".to_string())`.

This codec is intentionally simple and deterministic; it does not attempt to
optimize ratios or performance.

### Compression-aware SSD cache (`ssd_cache.rs`)

`SsdCache` was extended to understand compression:

- **`write_blob(&self, path: &str, data: &[u8], compression: CompressionKind)
   -> Result<CompressionMeta, MoveError>`**
  - If `compression == CompressionKind::None`:
    - Writes `data` as-is to disk.
    - Returns `CompressionMeta { kind: None, original_bytes }`.
  - If `compression == CompressionKind::Rle`:
    - Calls `rle_compress` and writes the compressed bytes.
    - Returns the associated `CompressionMeta`.
  - Any filesystem error is mapped to `MoveError::IoError` with a clear
    message.

- **`read_blob(&self, path: &str) -> Result<Vec<u8>, MoveError>`**
  - Unchanged: reads raw bytes from disk.

- **`read_blob_with_meta(&self, path: &str, meta: &CompressionMeta)
   -> Result<Vec<u8>, MoveError>`**
  - Reads bytes via `read_blob` and then:
    - If `meta.kind == CompressionKind::None`:
      - Verifies that the file length matches `meta.original_bytes`.
      - Returns bytes or `MoveError::Unsupported("Invalid SSD blob: length mismatch for uncompressed data".to_string())`.
    - If `meta.kind == CompressionKind::Rle`:
      - Calls `rle_decompress` with the metadata.

This layer remains synchronous and uses only the standard library.

### HybridMemoryManager integration (default: no compression)

`HybridMemoryManager` now stores and uses compression metadata when moving
tensors between RAM and SSD, while preserving previous behavior by default:

- **Registration in SSD**
  - When `register_tensor_with_data` is called with `initial == MemoryTier::Ssd`:
    - Ensures the cache directory exists.
    - Computes a blob path.
    - Calls `write_blob(path, data, CompressionKind::None)`.
    - Stores `StorageBacking::SsdFile { path, compression: Some(meta) }`.

- **RAM → SSD moves**
  - In `apply_move`, for `(MemoryTier::Ram, StorageBacking::Ram(data), MemoryTier::Ssd)`:
    - Ensures the cache directory exists.
    - Chooses `CompressionKind::None` (explicit TODO left for future policy).
    - Writes the blob via `write_blob` and clears the RAM `Vec<u8>`.
    - Stores `SsdFile { path, compression: Some(meta) }`.

- **SSD → RAM moves**
  - For `(MemoryTier::Ssd, StorageBacking::SsdFile { path, compression }, MemoryTier::Ram)`:
    - If `compression == Some(meta)` → uses `read_blob_with_meta(path, meta)`.
    - If `compression == None` → falls back to `read_blob(path)`.
    - Validates the resulting length via `footprint.validate_len`.
    - Deletes the blob and restores `StorageBacking::Ram(bytes)`.

- **VRAM ↔ SSD moves**
  - VRAM→SSD paths continue to write **uncompressed** blobs using
    `CompressionKind::None`, storing the returned `CompressionMeta`.
  - SSD→VRAM paths currently read raw bytes via `read_blob` only; compression
    awareness can be extended here in future subversions.

Throughout these changes, the **default external behavior remains
unchanged**: the policy always writes uncompressed data unless tests or
future code explicitly request `CompressionKind::Rle`.

### Tests: `tests/hybrid_memory_compression_test.rs`

This test suite validates the compression scaffold and ensures that default
behavior remains stable and deterministic. All tests use dedicated cache
directories and best-effort cleanup to avoid interference.

Covered scenarios:

- **`rle_roundtrip_integrity`**
  - Builds a byte buffer with repeated patterns.
  - Calls `rle_compress` and then `rle_decompress` with the returned
    `CompressionMeta`.
  - Asserts that the decompressed bytes are exactly equal to the input.

- **`ssd_roundtrip_with_compression_meta_none`**
  - Uses `HybridMemoryManager::new` with a dedicated cache directory.
  - Registers a tensor in RAM with `register_tensor_with_data`.
  - Plans and applies a RAM→SSD move (default `CompressionKind::None`).
  - Asserts that the backing is `SsdFile { compression: Some(meta) }` with
    `meta.kind == CompressionKind::None` and `meta.original_bytes` matching
    the original length.
  - Then plans and applies an SSD→RAM move and verifies that the resulting
    RAM backing bytes are identical to the original tensor data.

- **`ssd_roundtrip_with_rle_compression`**
  - Works directly with `SsdCache` on a separate cache directory.
  - Uses `write_blob(path, data, CompressionKind::Rle)` to write compressed
    bytes and obtain a `CompressionMeta`.
  - Reads back via `read_blob_with_meta(path, &meta)`, expecting transparent
    decompression and exact byte equality with the original data.

- **`invalid_rle_stream_returns_error`**
  - Constructs a `CompressionMeta` with `kind = Rle` and an expected length.
  - Calls `rle_decompress` with:
    - An odd-length input buffer.
    - A buffer with a zero `count` value.
  - In both cases, expects `MoveError::Unsupported` with a message
    containing `"Invalid RLE stream"`.

These tests ensure that:

- The RLE codec is deterministic and well-guarded against malformed input.
- The SSD cache correctly records and uses compression metadata.
- `HybridMemoryManager` can perform SSD roundtrips while keeping the default
  behavior effectively **uncompressed** from the caller's perspective.

## APX 13.3.0 – Hybrid Execution Planner (Decision-only Scaffold)

Version **13.3.0** introduces the first version of a
**Hybrid Execution Planner**. This planner decides *where* a kernel should
run (CPU, GPU, or CPU fallback) based on:

- Kernel characteristics.
- Current tensor placements (memory tiers).
- A hardware memory snapshot.

This subversion is **decision-only**:

- It does **not** execute kernels.
- It does **not** launch threads or touch GPU runtimes.
- It is deterministic, vendor-agnostic, and testable without a real GPU.

### Kernel model (`kernel_model.rs`)

The kernel model captures a small, explicit description of what a kernel
looks like from the scheduler's point of view:

- **`KernelKind`**
  - `ComputeHeavy` – arithmetic/compute-dominated kernels.
  - `MemoryBound` – bandwidth/latency-dominated kernels.
  - `Small` – tiny kernels where launch overhead dominates.
  - `Serial` – inherently sequential work that does not parallelize well.

- **`KernelProfile`**
  - Fields:
    - `name: String` – debug-friendly name.
    - `kind: KernelKind` – qualitative classification.
    - `estimated_flops: u64` – rough compute estimate.
    - `estimated_bytes: u64` – rough memory traffic estimate.
  - Method:
    - `pub fn is_gpu_friendly(&self) -> bool`
      - Returns `true` for `ComputeHeavy` and `MemoryBound`.
      - Returns `false` for `Small` and `Serial`.

This keeps the planner's interface simple while leaving room for future
extensions (e.g. more detailed performance estimates) without changing the
core decision API.

### Execution target and plan (`execution_planner.rs`)

The planner emits a minimal, debuggable execution plan:

- **`ExecutionTarget`**
  - `Cpu` – run on CPU under normal conditions.
  - `Gpu` – run on GPU when conditions are favorable.
  - `CpuFallback` – CPU execution chosen as a protective fallback.

- **`ExecutionPlan`**
  - `target: ExecutionTarget` – the selected execution target.
  - `reason: String` – a human-readable English explanation.

All reasons are short English sentences, suitable for logs and debugging.

### HybridExecutionPlanner – decision rules

The core planner is implemented as:

```rust
pub struct HybridExecutionPlanner;

impl HybridExecutionPlanner {
    pub fn plan(
        kernel: &KernelProfile,
        tensor_tiers: &[MemoryTier],
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> ExecutionPlan { ... }
}
```

The decision policy is deliberately simple and fully deterministic, using the
following ordered rules:

1. **GPU not available**
   - If `gpu_available == false` →

     - `target = ExecutionTarget::CpuFallback`
     - `reason = "GPU not available"`.
2. **Kernel not GPU-friendly**
   - If `kernel.is_gpu_friendly() == false` →

     - `target = ExecutionTarget::Cpu`
     - `reason = "Kernel not suitable for GPU execution"`.
3. **Any tensor resides on SSD**
   - If any entry in `tensor_tiers` is `MemoryTier::Ssd` →

     - `target = ExecutionTarget::Cpu`
     - `reason = "Tensor resides on SSD"`.
4. **VRAM pressure too high**
   - Reads `snapshot.vram.pressure.unwrap_or(0.0)`.
   - If `vram_pressure > 0.90` →

     - `target = ExecutionTarget::CpuFallback`
     - `reason = "VRAM pressure too high"`.
5. **Otherwise: prefer GPU**
   - In all remaining cases →

     - `target = ExecutionTarget::Gpu`
     - `reason = "GPU execution preferred"`.

There are no side effects: the planner does not launch kernels, does not
touch GPU runtimes, and does not modify any global state.

### Tests: `tests/hybrid_execution_planner_test.rs`

The planner is validated by a small, deterministic test suite that does not
require a real GPU. It uses synthetic `KernelProfile` values, `MemoryTier`
arrays, and `MemorySnapshot` instances built from `TierStatus`.

Covered scenarios:

- **`cpu_when_gpu_not_available`**
  - `gpu_available = false` for a GPU-friendly kernel and RAM-resident
    tensors.
  - Expects `ExecutionTarget::CpuFallback` and a reason containing
    `"GPU not available"`.
- **`cpu_when_kernel_small`**
  - Kernel kind set to `KernelKind::Small` with GPU available and tensors in
    RAM.
  - Expects `ExecutionTarget::Cpu` and a reason containing
    `"Kernel not suitable for GPU execution"`.
- **`cpu_when_tensor_on_ssd`**
  - GPU-friendly kernel with at least one tensor tier equal to
    `MemoryTier::Ssd`.
  - Expects `ExecutionTarget::Cpu` and a reason containing
    `"Tensor resides on SSD"`.
- **`cpu_fallback_when_vram_pressure_high`**
  - GPU-friendly kernel with VRAM pressure set to `Some(0.95)`.
  - Expects `ExecutionTarget::CpuFallback` and a reason containing
    `"VRAM pressure too high"`.
- **`gpu_when_all_conditions_good`**
  - GPU-friendly kernel, all tensors in RAM, `gpu_available = true`, and
    VRAM pressure low (e.g. `Some(0.2)`).
  - Expects `ExecutionTarget::Gpu` and a reason containing
    `"GPU execution preferred"`.

## APX 13.3.1 – Execution Decision Trace & Explainability

Version **13.3.1** augments the Hybrid Execution Planner with a structured
**decision trace**, without changing any decision rules. The goal is to make
every decision explainable and debuggable while keeping the planner
side-effect free.

### Decision rules (`DecisionRule`)

In `execution_planner.rs`, each rule is now identified by a small enum:

- **`DecisionRule`**
  - `GpuNotAvailable` – rule 1, GPU is globally unavailable.
  - `KernelNotGpuFriendly` – rule 2, kernel not suitable for GPU.
  - `TensorOnSsd` – rule 3, at least one tensor resides on SSD.
  - `HighVramPressure` – rule 4, VRAM pressure above a safe threshold.
  - `GpuPreferred` – rule 5, default GPU preference when all checks pass.

These identifiers map 1:1 to the existing rules and preserve their order.

### Decision trace model (`execution_trace.rs`)

A new module `execution_trace.rs` defines a structured trace for a single
planner decision:

- **`ExecutionDecisionTrace`**
  - `kernel_name: String` – name taken from `KernelProfile::name`.
  - `evaluated_rules: Vec<DecisionRule>` – rules checked in order.
  - `winning_rule: DecisionRule` – the rule that produced the final decision.
  - `target: ExecutionTarget` – the decided execution target.
  - `reason: String` – English explanation, mirrors `ExecutionPlan::reason`.

The trace is created once per decision and is immutable after construction.

### Extended ExecutionPlan (non-breaking)

`ExecutionPlan` now includes an optional trace field:

```rust
pub struct ExecutionPlan {
    pub target: ExecutionTarget,
    pub reason: String,
    pub trace: Option<ExecutionDecisionTrace>,
}
```

- Existing callers can continue to read only `target` and `reason`.
- The planner populates `trace` with a full `ExecutionDecisionTrace`.

### Instrumented planner (semantics unchanged)

`HybridExecutionPlanner::plan` still evaluates the same rules in the same
order, but now records which rules were considered:

1. Pushes `DecisionRule::GpuNotAvailable` and checks `gpu_available`.
2. Pushes `DecisionRule::KernelNotGpuFriendly` and checks
   `kernel.is_gpu_friendly()`.
3. Pushes `DecisionRule::TensorOnSsd` and checks for any `MemoryTier::Ssd`.
4. Pushes `DecisionRule::HighVramPressure` and checks
   `snapshot.vram.pressure.unwrap_or(0.0) > 0.90`.
5. Finally pushes `DecisionRule::GpuPreferred` when falling through to the
   default GPU path.

When a rule triggers, the planner captures:

- `evaluated_rules` – all rules checked up to and including the winner.
- `winning_rule` – which rule decided the outcome.
- `target` and `reason` – identical to 13.3.0 behavior.

The planner still performs **no real execution**, no threading, and no GPU
backend calls.

### Tests: `tests/hybrid_execution_trace_test.rs`

This test file validates that the trace is populated correctly, independent
of any real hardware. It uses synthetic kernels, tensor tiers, and memory
snapshots.

Covered scenarios:

- **`trace_records_rule_order`**
  - Scenario: GPU not available.
  - Asserts that `evaluated_rules == [GpuNotAvailable]` and
    `winning_rule == GpuNotAvailable`.

- **`trace_records_multiple_rules_until_match`**
  - Scenario: small kernel with GPU available and tensors in RAM.
  - Asserts that `evaluated_rules == [GpuNotAvailable, KernelNotGpuFriendly]`
    and `winning_rule == KernelNotGpuFriendly`.

- **`trace_gpu_preferred_case`**
  - Scenario: GPU-friendly kernel, all tensors in RAM, low VRAM pressure,
    GPU available.
  - Asserts that `evaluated_rules` contains all rules in order and that
    `winning_rule == GpuPreferred` with `target == ExecutionTarget::Gpu`.

- **`trace_reason_matches_plan`**
  - Scenario: GPU-preferred case.
  - Asserts that `trace.reason == plan.reason` and
    `trace.target == plan.target`.

All tests are deterministic, do not require a GPU, and do not perform any
real execution. They only exercise the decision and tracing logic.

## APX 13.4.0 – Asynchronous Hybrid Streams (Scaffold + Mock Executor)

Version **13.4.0** introduces a first-class **stream model** and a
deterministic, single-threaded executor that simulates asynchronous execution
across CPU, GPU, and SSD prefetch streams. This is a **scaffold only**:

- No real GPU stream APIs.
- No real asynchronous file I/O.
- No async runtimes or new crates.
- Purely deterministic, CI-safe behavior.

The goal is to establish the contract for hybrid streams and to provide a
human-readable timeline that can be reused by later subversions (e.g. 13.5
and 13.10).

### Stream model (`streams.rs`)

The `streams.rs` module defines the core types used by the executor:

- **`StreamKind`**
  - `Cpu` – CPU execution stream.
  - `Gpu` – GPU execution stream.
  - `SsdPrefetch` – background SSD prefetch stream.

- **`TaskKind`**
  - `Compute { name: String }` – compute kernels (e.g. matmul, conv).
  - `Transfer { name: String }` – data transfer tasks (e.g. H2D, D2H).
  - `Io { name: String }` – I/O-related work (e.g. SSD prefetch).

- **`StreamTask`**
  - `pub id: u64` – monotonically assigned identifier.
  - `pub stream: StreamKind` – logical stream for this task.
  - `pub kind: TaskKind` – high-level classification and debug name.
  - `pub estimated_cost: u64` – abstract cost units (not used by the
    scheduler yet, but reserved for future policies).

- **`StreamConfig`**
  - `pub advanced_streams_supported: bool` – toggles between advanced
    round-robin scheduling and a conservative fallback mode.

All non-code text (names, comments) remains in English to keep logs and
debugging consistent.

### Deterministic AsyncExecutor (`async_executor.rs`)

The `AsyncExecutor` is a mock, single-threaded executor that maintains
per-stream queues and a public timeline of events:

```rust
pub struct AsyncExecutor {
    cfg: StreamConfig,
    cpu_q: VecDeque<StreamTask>,
    gpu_q: VecDeque<StreamTask>,
    ssd_q: VecDeque<StreamTask>,
    pub timeline: Vec<String>,
    next_id: u64,
}
```

Key behavior:

- **Construction**
  - `pub fn new(cfg: StreamConfig) -> Self`
  - Initializes empty queues, an empty `timeline`, and `next_id = 1`.

- **Submission**
  - `pub fn submit(&mut self, stream: StreamKind, kind: TaskKind, cost: u64) -> u64`
  - Assigns a monotonically increasing `id` using `next_id` and
    `saturating_add(1)`.
  - Enqueues the resulting `StreamTask` into the corresponding queue
    (`cpu_q`, `gpu_q`, or `ssd_q`).
  - Appends an `ENQUEUE` line to the `timeline`, for example:
    - `"ENQUEUE stream=Cpu id=1 kind=Compute name=matmul cost=10"`.
  - Returns the assigned `id` to the caller.

- **Execution entry point**
  - `pub fn run_to_completion(&mut self)`
  - If `cfg.advanced_streams_supported == true` → runs `run_advanced()`.
  - Otherwise → runs `run_fallback()`.

#### Advanced streams mode (round-robin)

When `advanced_streams_supported` is `true`, the executor simulates
overlapping streams using a deterministic round-robin policy:

- Iterates synchronously with an index `i`.
- At each step, chooses a preferred stream based on `i % 3`:
  - `0` → `Cpu`
  - `1` → `Gpu`
  - `2` → `SsdPrefetch`
- If the chosen stream's queue is non-empty, it pops **one** task and
  records a `RUN` line, for example:
  - `"RUN stream=Gpu id=2 kind=Compute name=conv cost=20"`.
- The loop terminates when all three queues are empty.

Per-stream ordering remains **FIFO** because each queue is a `VecDeque` and
tasks are always taken from the front.

#### Fallback mode (serialize to CPU)

When `advanced_streams_supported` is `false`, the executor behaves in a
conservative, serialized way:

1. Drains **CPU** tasks as-is, producing only `RUN` lines with
   `stream=Cpu`.
2. Then drains **GPU** tasks:
   - For each GPU task, records a `FALLBACK` line:
     - `"FALLBACK stream=Gpu id=.. -> Cpu"`.
   - Immediately records a `RUN` line treating the task as CPU work.
3. Finally drains **SSD prefetch** tasks in the same pattern:
   - `"FALLBACK stream=SsdPrefetch id=.. -> Cpu"`.
   - Followed by a `RUN` line with `stream=Cpu`.

This mode preserves per-stream FIFO semantics within GPU and SSD queues
while making all actual execution logically CPU-only, matching a
"no advanced streams" backend.

### Tests: `tests/hybrid_streams_scaffold_test.rs`

The test suite validates the behavior of the `AsyncExecutor` in both
advanced and fallback modes, and confirms that task IDs are monotonic.

Covered scenarios:

- **`advanced_streams_round_robin_order`**
  - Uses `StreamConfig { advanced_streams_supported: true }`.
  - Submits tasks to `Cpu`, `Gpu`, and `SsdPrefetch` (two CPU tasks, one GPU,
    one SSD).
  - Runs `run_to_completion()` and extracts only `RUN` lines from
    `timeline`.
  - Asserts that the first three `RUN` entries correspond to
    `Cpu`, `Gpu`, and `SsdPrefetch` respectively, demonstrating the
    round-robin policy.
  - Confirms per-stream FIFO ordering by checking that `cpu_task1` appears
    before `cpu_task2` in the timeline.

- **`fallback_serializes_all_to_cpu`**
  - Uses `StreamConfig { advanced_streams_supported: false }`.
  - Submits tasks to `Gpu` and `SsdPrefetch`.
  - Runs `run_to_completion()` and iterates over `timeline` entries.
  - Asserts that:
    - There are `FALLBACK` entries for both `stream=Gpu` and
      `stream=SsdPrefetch`.
    - All `RUN` entries use `stream=Cpu`, confirming that execution is
      serialized onto the CPU stream.

- **`ids_are_monotonic`**
  - Uses `advanced_streams_supported: true`.
  - Submits three tasks across different streams and captures the returned
    IDs.
  - Asserts that `id1 < id2` and `id2 < id3`, verifying monotonic ID
    assignment regardless of stream.

These tests are deterministic, single-threaded, and do not rely on any real
GPU or I/O backend. They validate only the mock scheduling and timeline
semantics defined for APX 13.4.0.

## APX 13.4.1 – Planner-driven Stream Assignment

Version **13.4.1** connects the **Hybrid Execution Planner** with the
**AsyncExecutor** and stream model introduced in 13.4.0. It introduces a
router that translates high-level planner decisions into concrete stream
tasks without executing any real kernels.

### StreamRouter API (`stream_router.rs`)

The core entry point is a small, stateless router:

- **`StreamRouter`**
  - Empty struct used as a namespace for routing functions.

- **`RoutedBundle`**
  - `plan_target: ExecutionTarget` – the target chosen by
    `HybridExecutionPlanner` (CPU, GPU, or CPU fallback).
  - `submitted_task_ids: Vec<u64>` – IDs of tasks submitted to the
    `AsyncExecutor`, in the exact order they were enqueued.
  - `reason: String` – the planner's human-readable explanation.

The main function is:

```rust
pub struct StreamRouter;

pub struct RoutedBundle {
    pub plan_target: ExecutionTarget,
    pub submitted_task_ids: Vec<u64>,
    pub reason: String,
}

impl StreamRouter {
    pub fn route_kernel(
        exec: &mut AsyncExecutor,
        kernel: &KernelProfile,
        tensor_tiers: &[MemoryTier],
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> RoutedBundle { ... }
}
```

### Routing rules

`route_kernel` performs the following steps in order:

1. **Call the planner**
   - Invokes `HybridExecutionPlanner::plan(kernel, tensor_tiers, snapshot, gpu_available)`.
   - Captures `plan.target` and `plan.reason`.

2. **Inject SSD prefetch when needed**
   - If any entry in `tensor_tiers` is `MemoryTier::Ssd`:
     - Submits an I/O task to the `SsdPrefetch` stream:
       - `StreamKind::SsdPrefetch`.
       - `TaskKind::Io { name: format!("prefetch:{}", kernel.name) }`.
       - `cost = 1` (small constant).
     - Pushes the returned task ID as the **first** element of
       `submitted_task_ids`.

3. **Route compute task according to planner target**
   - Chooses the stream from the planner's decision:
     - If `plan.target == ExecutionTarget::Gpu` → `StreamKind::Gpu`.
     - If `plan.target == ExecutionTarget::Cpu` or
       `ExecutionTarget::CpuFallback` → `StreamKind::Cpu`.
   - Submits a compute task:
     - `TaskKind::Compute { name: kernel.name.clone() }`.
     - `cost = 1`.
   - Appends the compute task ID to `submitted_task_ids`.

4. **Return bundle**
   - Returns `RoutedBundle` with:
     - `plan_target = plan.target`.
     - `submitted_task_ids` in enqueue order (prefetch first, then compute).
     - `reason = plan.reason`.

The actual execution remains under `AsyncExecutor::run_to_completion`, which
preserves deterministic semantics and fallback behavior.

### Tests: `tests/hybrid_stream_router_test.rs`

This test suite validates the integration between planner, router, and
mock executor. It relies only on synthetic kernels, tensor tiers, and memory
snapshots.

Covered scenarios:

- **`routes_gpu_compute_to_gpu_stream`**
  - GPU-friendly kernel (`ComputeHeavy`), all tensors in RAM, low VRAM
    pressure, `gpu_available = true`.
  - Asserts that `plan_target == ExecutionTarget::Gpu`.
  - After `run_to_completion`, the `timeline` contains both an
    `ENQUEUE stream=Gpu` and a `RUN stream=Gpu` entry.
  - Confirms that no `SsdPrefetch` entries are present.

- **`injects_prefetch_when_tensor_on_ssd`**
  - GPU-friendly kernel with tensor tiers including `MemoryTier::Ssd`.
  - Asserts that `submitted_task_ids.len() == 2` (prefetch + compute).
  - Verifies that `ENQUEUE` entries appear in the expected order:
    - First an `ENQUEUE` for `stream=SsdPrefetch`.
    - Then an `ENQUEUE` whose `name` matches the kernel compute task.
  - After `run_to_completion`, asserts that the `timeline` contains at least
    one `RUN stream=SsdPrefetch` and one `RUN` for the compute task.

- **`routes_to_cpu_when_gpu_not_available`**
  - GPU-friendly kernel, tensors in RAM, but `gpu_available = false`.
  - Confirms that `plan_target == ExecutionTarget::CpuFallback`.
  - After execution, all `RUN` entries use `stream=Cpu`, reflecting the
    planner's fallback decision.

- **`fallback_mode_serializes_gpu_task`**
  - Uses an `AsyncExecutor` with `advanced_streams_supported = false`.
  - Planner picks `ExecutionTarget::Gpu` for a GPU-friendly kernel.
  - Router enqueues compute on the GPU stream.
  - Executor fallback mode then:
    - Records `FALLBACK stream=Gpu id=.. -> Cpu` entries.
    - Runs all tasks as `stream=Cpu` in the `RUN` entries.

Together, these tests confirm that planner decisions are faithfully
translated into stream tasks, SSD prefetch is injected deterministically
when required, and executor fallback semantics remain intact, all without
performing any real computation.

## APX 13.4.2 – Memory-aware Routing (Prefetch → Move → Compute)

Version **13.4.2** extends the stream router to become **memory-aware** by
integrating `HybridMemoryManager`. The router now:

- Looks up the current memory tier of tensors.
- Plans and applies memory moves via `HybridMemoryManager`.
- Enqueues prefetch, transfer, and compute tasks in a deterministic order.
- Safely degrades to CPU when required memory tiers are not available.

### Extended StreamRouter API (`stream_router.rs`)

In addition to the existing `route_kernel(...)`, a new entrypoint is
introduced:

```rust
pub fn route_kernel_with_memory(
    exec: &mut AsyncExecutor,
    mem: &mut HybridMemoryManager,
    kernel: &KernelProfile,
    tensor_ids: &[&str],
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) -> RoutedBundle
```

This function:

- Derives tensor tiers by calling `mem.get_tier(id)` for each `tensor_id`.
- Calls `HybridExecutionPlanner::plan` using the derived tiers and snapshot.
- Computes the **required memory tier** for compute:
  - `ExecutionTarget::Gpu` → `MemoryTier::Vram`.
  - `ExecutionTarget::Cpu`/`CpuFallback` → `MemoryTier::Ram`.
- Prepares memory (prefetch + moves) and then enqueues the final compute
  task.

### Memory preparation rules

For each tensor, the router ensures that data is placed in the tier required
by the planned execution target.

1. **SSD tensors when compute requires RAM/VRAM**

   - If `current_tier == MemoryTier::Ssd` and `required_tier != Ssd`:

     1. **Prefetch task**
        - Enqueues an I/O task on the SSD prefetch stream:
          - `StreamKind::SsdPrefetch`.
          - `TaskKind::Io { name: format!("prefetch:{}", tensor_id) }`.
          - `cost = 1`.

     2. **Plan and apply move**
        - Calls `mem.plan_move(id, required_tier, snapshot)`.
        - If planning fails or returns a different target (e.g. VRAM
          unavailable), the router **degrades to CPU** (see below).
        - Otherwise, calls `mem.apply_move(id, &plan)`.

     3. **Transfer task**
        - On successful move, enqueues a transfer task:
          - If `required_tier == Ram`:
            - `stream = StreamKind::Cpu`.
            - `TaskKind::Transfer { name: format!("move:ssd->ram:{}", id) }`.
          - If `required_tier == Vram`:
            - `stream = StreamKind::Gpu`.
            - `TaskKind::Transfer { name: format!("move:ssd->vram:{}", id) }`.
          - `cost = 1`.

2. **RAM → VRAM for GPU execution**

   - If `current_tier == Ram`, `required_tier == Vram` and
     `plan.target == ExecutionTarget::Gpu`:
     - Calls `mem.plan_move(id, MemoryTier::Vram, snapshot)`.
     - If `plan_move` returns a degraded target (e.g. VRAM unavailable) or
       if `apply_move` fails, the router degrades to CPU.
     - Otherwise, enqueues:
       - `stream = StreamKind::Gpu`.
       - `TaskKind::Transfer { name: format!("move:ram->vram:{}", id) }`.
       - `cost = 1`.

3. **VRAM → RAM for CPU execution**

   - If `current_tier == Vram`, `required_tier == Ram` and
     `plan.target != ExecutionTarget::Gpu`:
     - Calls `mem.plan_move(id, MemoryTier::Ram, snapshot)` and
       `mem.apply_move` similarly.
     - Enqueues a CPU transfer task:
       - `stream = StreamKind::Cpu`.
       - `TaskKind::Transfer { name: format!("move:vram->ram:{}", id) }`.

### Safe degradation semantics

If any `plan_move` or `apply_move` fails, or if the move cannot reach the
required tier (for example when VRAM is unavailable), the router does **not
panic**. Instead:

- It marks the bundle as degraded:
  - `plan_target = ExecutionTarget::CpuFallback`.
  - `reason` extended with an English explanation, e.g.:
    - `"...; degraded to CPU because VRAM is unavailable"`.
    - `"...; degraded to CPU because memory move failed"`.
- The final compute task is always enqueued to the CPU stream.

Tensor tiers are still updated by `HybridMemoryManager` based on whatever
moves succeeded.

### Final compute task

After all memory preparation steps (prefetch + moves/transfers), the router
enqueues a single compute task:

- If not degraded and `plan.target == ExecutionTarget::Gpu`:
  - `StreamKind::Gpu`, `TaskKind::Compute { name: kernel.name.clone() }`.
- Otherwise (CPU or fallback):
  - `StreamKind::Cpu`, `TaskKind::Compute { name: kernel.name.clone() }`.

This guarantees a deterministic ENQUEUE ordering per routing call:

1. All SSD prefetch tasks.
2. All transfer tasks.
3. The final compute task.

### Tests: `tests/hybrid_stream_router_memory_test.rs`

This test file validates memory-aware routing using real
`HybridMemoryManager` moves and a fake VRAM adapter for the VRAM case.

Covered scenarios:

- **`ssd_tensor_is_moved_to_ram_before_cpu_compute`**
  - Registers a tensor `t1` initially in SSD using `register_tensor_with_data`.
  - Uses a CPU-targeted kernel (small or with `gpu_available = false`).
  - Calls `route_kernel_with_memory` with `tensor_ids = ["t1"]`.
  - Asserts that ENQUEUE entries follow:
    1. `ENQUEUE stream=SsdPrefetch ... "prefetch:t1"`.
    2. `ENQUEUE ... "move:ssd->ram:t1"`.
    3. `ENQUEUE stream=Cpu ... name=cpu_kernel`.
  - After execution, confirms `mem.get_tier("t1") == Some(MemoryTier::Ram)`
    and that `RUN` entries include prefetch, transfer, and compute.

- **`ram_tensor_is_moved_to_vram_before_gpu_compute_using_fake_vram`**
  - Uses `HybridMemoryManager::new_with_vram` with a `FakeVramAdapter`.
  - Registers a tensor `t2` in RAM.
  - Uses a GPU-friendly kernel with `gpu_available = true` and low VRAM
    pressure.
  - Calls `route_kernel_with_memory` and asserts:
    - `plan_target == ExecutionTarget::Gpu`.
    - ENQUEUEs:
      - First: `ENQUEUE stream=Gpu ... "move:ram->vram:t2"`.
      - Second: `ENQUEUE stream=Gpu ... name=gpu_kernel`.
  - After execution, confirms `mem.get_tier("t2") == Some(MemoryTier::Vram)`
    and observes GPU `RUN` entries for both transfer and compute.

- **`vram_unavailable_degrades_to_cpu`**
  - Uses `HybridMemoryManager::new` (backed by `NullVramAdapter`, VRAM
    unavailable).
  - Registers a tensor `t3` in RAM and uses a GPU-friendly kernel.
  - Calls `route_kernel_with_memory` with `gpu_available = true`.
  - Asserts that:
    - `plan_target == ExecutionTarget::CpuFallback`.
    - `reason` contains words like `"degraded"` or `"fallback"`.
    - The tensor remains in RAM.
    - Both ENQUEUE and RUN entries for the compute task target the CPU
      stream.

All tests are deterministic and use real `HybridMemoryManager` moves (RAM
↔ SSD and RAM/SSD ↔ VRAM via adapters) while keeping the overall system
side-effect free beyond local cache directories.

## APX 13.5.0 – Smart Offloading Engine (Policy Scaffold)

Version **13.5.0** introduces a **SmartOffloadEngine** that inspects memory
pressures and proposes offload actions for specific tensors. This version is
policy-only and deterministic:

- No background monitoring loops.
- No hysteresis or cooldowns.
- No threading or hardware polling.

The engine generates an `OffloadPlan` with human-readable reasons and can
optionally apply the plan via `HybridMemoryManager`.

### Offload model (`offload_engine.rs`)

The offload model consists of:

- **`OffloadAction`**
  - `MoveToRam { tensor_id: String }`
  - `MoveToSsd { tensor_id: String }`

- **`OffloadPlan`**
  - `actions: Vec<OffloadAction>` – planned actions for the given tensors.
  - `reason: String` – English explanation describing why the plan was
    generated (e.g. high VRAM/RAM pressure).

- **`SmartOffloadEngine`**
  - `pub vram_threshold: f32` – threshold for `snapshot.vram.pressure`.
  - `pub ram_threshold: f32` – threshold for `snapshot.ram.pressure`.

The default constructor provides a simple policy:

```rust
impl SmartOffloadEngine {
    pub fn default() -> Self {
        SmartOffloadEngine {
            vram_threshold: 0.95,
            ram_threshold: 0.95,
        }
    }
}
```

### Planning API

The planning method computes an offload plan for a specific set of tensors:

```rust
pub fn plan(
    &self,
    snapshot: &MemorySnapshot,
    tensor_ids: &[&str],
    mem: &HybridMemoryManager,
) -> OffloadPlan
```

Key rules (simple v1, deterministic):

1. **Both VRAM and RAM pressure high**
   - If `snapshot.vram.pressure >= vram_threshold` **and**
     `snapshot.ram.pressure >= ram_threshold`:
     - For each `tensor_id` whose current tier (via `mem.get_tier`) is
       `MemoryTier::Vram` or `MemoryTier::Ram`, add:
       - `OffloadAction::MoveToSsd { tensor_id }`.
     - The plan `reason` is set to:
       - `"VRAM and RAM pressure high"`.
     - This prefers SSD offloading for both VRAM and RAM tensors to relieve
       pressure on both tiers. For VRAM tensors, the two-step
       VRAM → RAM → SSD move is handled later in `apply`.

2. **Only VRAM pressure high**
   - If VRAM pressure exceeds the threshold but RAM pressure does not:
     - For each tensor currently in `MemoryTier::Vram` among `tensor_ids`,
       add:
       - `OffloadAction::MoveToRam { tensor_id }`.
     - `reason` is:
       - `"VRAM pressure high"` if at least one action is planned.
       - `"VRAM pressure high but no offloadable tensors in VRAM"` if no
         provided tensors reside in VRAM.

3. **Only RAM pressure high**
   - If RAM pressure exceeds the threshold but VRAM pressure does not:
     - For each tensor currently in `MemoryTier::Ram` among `tensor_ids`,
       add:
       - `OffloadAction::MoveToSsd { tensor_id }`.
     - `reason` is:
       - `"RAM pressure high"` if at least one action is planned.
       - `"RAM pressure high but no offloadable tensors in RAM"` otherwise.

4. **No offloading needed**
   - If neither pressure exceeds its threshold and no actions are added,
     the plan contains an empty `actions` vector and:
     - `reason = "No offloading needed"`.

The planner only considers the provided `tensor_ids` and avoids duplicate
actions per tensor.

### Apply API (safe moves)

The offload engine can apply a previously computed plan using
`HybridMemoryManager`:

```rust
pub fn apply(
    &self,
    snapshot: &MemorySnapshot,
    plan: &OffloadPlan,
    mem: &mut HybridMemoryManager,
) -> Result<(), MoveError>
```

For each action in `plan.actions`:

- **`MoveToRam { tensor_id }`**
  - Plans a move to RAM and applies it:
    - `let move_plan = mem.plan_move(id, MemoryTier::Ram, snapshot)?;`
    - `mem.apply_move(id, &move_plan)?;`

- **`MoveToSsd { tensor_id }`**
  - If the tensor is in VRAM, first moves it to RAM, then to SSD:
    - If `mem.get_tier(id) == Some(MemoryTier::Vram)`:
      - Plan/apply `Vram → Ram`.
    - Plan/apply `target = MemoryTier::Ssd`.

All failures are propagated as `Result::Err(MoveError)`; there are **no
panics**. The details of how bytes move between tiers are delegated entirely
to `HybridMemoryManager`.

### Tests: `tests/smart_offload_engine_test.rs`

The test suite validates both the planning policy and the application of
offload plans using a real `HybridMemoryManager` and a fake VRAM adapter.
All tests use dedicated cache directories under `./.atenia_cache_test_offload*`.

Covered scenarios:

- **`vram_high_moves_vram_tensors_to_ram`**
  - Uses `HybridMemoryManager::new_with_vram` with a `FakeVramAdapter`.
  - Registers tensor `t1` in RAM with data, then moves it to VRAM.
  - Builds a snapshot with `vram.pressure = 0.99`, `ram.pressure = 0.1`.
  - Expects `plan.actions` to contain a single
    `OffloadAction::MoveToRam { tensor_id: "t1" }` and a reason containing
    `"VRAM pressure high"`.
  - After `apply`, asserts that `mem.get_tier("t1") == Some(MemoryTier::Ram)`.

- **`ram_high_moves_ram_tensors_to_ssd`**
  - Registers tensor `t2` in RAM with real data.
  - Builds a snapshot with `ram.pressure = 0.99` and low VRAM pressure.
  - Expects the plan to contain `MoveToSsd { tensor_id: "t2" }`.
  - After `apply`, confirms that `mem.get_tier("t2") == Some(MemoryTier::Ssd)`.

- **`both_high_prefers_ssd`**
  - Uses a VRAM-enabled `HybridMemoryManager`.
  - Registers `t3` in RAM, then moves it to VRAM; registers `t4` in RAM.
  - Snapshot has both VRAM and RAM pressures above threshold.
  - Expects `plan.actions` to contain `MoveToSsd` for **both** `t3` and `t4`
    and a reason containing `"VRAM and RAM pressure high"`.
  - After `apply`, both tensors should have tier `MemoryTier::Ssd`, with
    `t3` having taken a two-step path via RAM.

- **`no_offload_when_pressures_low`**
  - Registers `t5` in RAM with data.
  - Snapshot with low VRAM and RAM pressures.
  - Expects `plan.actions.is_empty()` and a reason containing
    `"No offloading needed"`.

These tests ensure that the SmartOffloadEngine produces deterministic,
explainable offload plans and that applying those plans leads to real,
observable tier changes via `HybridMemoryManager`.

## APX 13.5.1 – Smart Offloading Engine (Hysteresis + Cooldown)

Version **13.5.1** extends the SmartOffloadEngine with **hysteresis bands**
and **per-tensor cooldown** to reduce oscillations. The goals are:

- Avoid repeated offload/"unoffload" decisions while pressure hovers around
  a single threshold.
- Prevent immediately re-offloading the same tensor across consecutive
  planning cycles.
- Keep behavior deterministic, using a logical tick instead of wall-clock
  time.

The 13.5.0 behavior remains available via the legacy `plan` API, which uses
only "high" thresholds and no cooldown. The new hysteresis behavior is
opt-in via a separate `plan_with_tick` method.

### Extended SmartOffloadEngine configuration (`offload_engine.rs`)

The engine now exposes bands instead of single thresholds and includes a
cooldown horizon:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SmartOffloadEngine {
    pub vram_high: f32,
    pub vram_low: f32,
    pub ram_high: f32,
    pub ram_low: f32,
    pub cooldown_ticks: u64,
    last_moved: HashMap<String, u64>,
}
```

- `vram_high` / `ram_high` – upper thresholds where offload decisions are
  allowed.
- `vram_low` / `ram_low` – lower thresholds below which no offloading is
  needed.
- `cooldown_ticks` – minimal logical ticks between consecutive moves
  involving the same tensor.
- `last_moved` – internal map tracking the last tick at which a tensor was
  subject to an offload action.

The default constructor defines a simple band and cooldown:

```rust
impl SmartOffloadEngine {
    pub fn default() -> Self {
        SmartOffloadEngine {
            vram_high: 0.95,
            vram_low: 0.85,
            ram_high: 0.95,
            ram_low: 0.85,
            cooldown_ticks: 5,
            last_moved: HashMap::new(),
        }
    }
}
```

The existing `plan` method remains as a **13.5.0 compatibility shim**. It
uses the `*_high` thresholds only and ignores hysteresis and cooldown:

```rust
pub fn plan(
    &self,
    snapshot: &MemorySnapshot,
    tensor_ids: &[&str],
    mem: &HybridMemoryManager,
) -> OffloadPlan
```

### New hysteresis-aware planning API

The new entrypoint introduces a logical tick and uses both bands and
cooldown:

```rust
pub fn plan_with_tick(
    &mut self,
    snapshot: &MemorySnapshot,
    tensor_ids: &[&str],
    mem: &HybridMemoryManager,
    tick: u64,
) -> OffloadPlan
```

- `tick` is a **logical time** value that the caller advances monotonically
  between planning cycles (e.g. per scheduler iteration). No real clock is
  used.
- The method is `&mut self` because it updates the internal `last_moved`
  map whenever it schedules an offload action.

#### Hysteresis rules

The engine derives four boolean conditions from the snapshot:

- `vram_high = snapshot.vram.pressure >= vram_high`.
- `vram_low  = snapshot.vram.pressure <= vram_low`.
- `ram_high  = snapshot.ram.pressure >= ram_high`.
- `ram_low   = snapshot.ram.pressure <= ram_low`.

Decisions are then made according to these states:

1. **Both VRAM and RAM high**

   - If `vram_high && ram_high`:
     - For each `tensor_id` whose tier (via `mem.get_tier`) is `Vram` or
       `Ram`, attempt to schedule:
       - `OffloadAction::MoveToSsd { tensor_id }`.
     - This preserves the 13.5.0 policy: when both tiers are under heavy
       pressure, preferentially move tensors to SSD.
     - The plan reason starts with:
       - `"VRAM and RAM pressure high"`.

2. **VRAM-only high**

   - If `vram_high && !ram_high`:
     - For each tensor currently in `MemoryTier::Vram` among `tensor_ids`:
       - Attempt `OffloadAction::MoveToRam { tensor_id }`.
     - The reason is either:
       - `"VRAM pressure high"` (when actions are present), or
       - `"VRAM pressure high but no offloadable tensors in VRAM"`.

3. **RAM-only high**

   - If `ram_high && !vram_high`:
     - For each tensor currently in `MemoryTier::Ram` among `tensor_ids`:
       - Attempt `OffloadAction::MoveToSsd { tensor_id }`.
     - The reason is either:
       - `"RAM pressure high"` (when actions are present), or
       - `"RAM pressure high but no offloadable tensors in RAM"`.

4. **Stable band / below-low pressure**

   - When neither VRAM nor RAM are above their high thresholds, the engine
     does **not** introduce new offloads:
     - If `vram_low && ram_low`:
       - `reason = "No offloading needed"`.
     - Otherwise (between low and high for one or both tiers):
       - `reason = "No new offloading due to hysteresis band"`.

In all cases, the engine only considers the explicitly provided
`tensor_ids` and de-duplicates actions per tensor.

#### Cooldown rules

Before scheduling an action for a given `tensor_id`, the engine checks a
cooldown condition:

```rust
let can_schedule = |id: &str,
                    last_moved: &HashMap<String, u64>,
                    cooldown_ticks: u64| -> bool { ... };
```

For each candidate tensor:

- If `last_moved` does not contain the id → scheduling is allowed.
- If `tick <= last_tick` (non-monotonic or equal) → scheduling is denied.
- If `tick - last_tick < cooldown_ticks` → scheduling is denied (still in
  cooldown window).
- Otherwise → scheduling is allowed.

When an action is finally added to the `actions` vector, the engine updates

```rust
self.last_moved.insert(tensor_id.clone(), tick);
```

If any tensor is skipped due to cooldown, the plan reason is extended with
the suffix:

- `"; some tensors skipped due to cooldown"`.

This ensures that repeated planning calls with unchanged high pressures do
not keep scheduling moves for the same tensors on every tick.

### Tests: `tests/smart_offload_engine_hysteresis_test.rs`

This test file focuses on planning behavior only (no calls to `apply` are
required). It builds artificial `MemorySnapshot` instances and uses a
dedicated cache directory `./.atenia_cache_test_offload_hys`.

Covered scenarios:

- **`stable_band_does_not_trigger`**
  - Tensor resides in `MemoryTier::Vram`.
  - VRAM pressure is between `vram_low` and `vram_high`.
  - `plan_with_tick` returns an empty `actions` vector.

- **`triggers_on_high`**
  - Tensor resides in `MemoryTier::Vram`.
  - VRAM pressure is above `vram_high`.
  - `plan_with_tick` yields a single
    `OffloadAction::MoveToRam { tensor_id }` and a reason containing
    `"VRAM pressure high"`.

- **`cooldown_skips_repeated_moves`**
  - Tensor resides in `MemoryTier::Vram` with high VRAM pressure.
  - First call at `tick = 10` produces one move action.
  - Second call at `tick = 12` (within the default cooldown window of 5
    ticks) produces **no** actions and a reason that mentions cooldown or
    the absence of new offloading.
  - Third call at `tick = 16` (after cooldown) again produces a move
    action.

- **`below_low_turns_off_pressure`**
  - Tensor resides in `MemoryTier::Vram`.
  - VRAM pressure is below `vram_low`.
  - `plan_with_tick` returns no actions and a reason indicating that no
    offloading is needed or that no new offloading is planned.

Together, these tests confirm that:

- No new actions are scheduled in the hysteresis stable band.
- Cooldown is enforced per tensor via logical ticks.
- The behavior is fully deterministic and independent of wall-clock time.

## APX 13.5.2 – Smart Offloading Engine (Priority Offloading)

Version **13.5.2** adds **priority-based selection** and a **per-tick
budget** on top of the hysteresis + cooldown behavior introduced in
13.5.1. The goals are:

- Prefer offloading larger tensors first (more memory freed per action).
- Keep decisions deterministic via a simple scoring rule and tie-breaker.
- Limit the number of actions per planning cycle to avoid aggressive
  thrashing under high pressure.

The `apply` API remains unchanged; this version focuses purely on
planning.

### Extended SmartOffloadEngine configuration (priority + budget)

The engine configuration gains a new field:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SmartOffloadEngine {
    pub vram_high: f32,
    pub vram_low: f32,
    pub ram_high: f32,
    pub ram_low: f32,
    pub cooldown_ticks: u64,
    pub max_actions_per_tick: usize,
    last_moved: HashMap<String, u64>,
}
```

- `max_actions_per_tick` – hard limit on how many offload actions may be
  planned in a single `plan_with_tick` call.

The default constructor sets:

```rust
impl SmartOffloadEngine {
    pub fn default() -> Self {
        SmartOffloadEngine {
            vram_high: 0.95,
            vram_low: 0.85,
            ram_high: 0.95,
            ram_low: 0.85,
            cooldown_ticks: 5,
            max_actions_per_tick: 4,
            last_moved: HashMap::new(),
        }
    }
}
```

The legacy `plan` method (13.5.0 semantics) remains available and does not
use `max_actions_per_tick`. Priority and budgeting only affect
`plan_with_tick`.

### Tensor scoring

To implement priority ordering, the engine introduces a simple scoring
function:

```rust
fn score_tensor(&self, mem: &HybridMemoryManager, id: &str) -> u64 {
    match mem.tensor_len_bytes(id) {
        Some(len) => len as u64,
        None => 0,
    }
}
```

The score is currently just the tensor length in bytes obtained from
`HybridMemoryManager`. If the size is unknown, the score is zero.

The corresponding metadata accessor is added to `HybridMemoryManager`:

```rust
pub fn tensor_len_bytes(&self, id: &str) -> Option<usize> {
    self.tensors
        .get(id)
        .map(|r| r.footprint.bytes as usize)
}
```

This reuses the existing `MemoryFootprint` without changing any
registration APIs.

### Priority integration in `plan_with_tick`

The hysteresis and cooldown rules from 13.5.1 are preserved. What changes
in 13.5.2 is **how eligible tensors are chosen** under high pressure.

Internally, `plan_with_tick` now builds a sorted list of candidate tensors
and applies a budget:

1. **Candidate selection**

   - For the relevant pressure condition (VRAM+RAM high, VRAM-only high,
     RAM-only high), the engine:
     - Iterates over the provided `tensor_ids`.
     - Looks up each tensor's `MemoryTier` via `mem.get_tier(id)`.
     - Applies the same predicates as in 13.5.1:
       - Both high: tiers `Vram` or `Ram`.
       - VRAM-only high: tier `Vram`.
       - RAM-only high: tier `Ram`.
     - Applies cooldown using the existing `can_schedule` helper.
     - Avoids duplicates by tracking ids in a `HashSet<String>`.
     - Computes a score for each candidate using `score_tensor`.

2. **Sorting and tie-breaking**

   - Candidates are sorted by:
     1. Score descending (larger tensors first).
     2. Tensor id ascending (lexicographically) when scores are equal.

3. **Budget enforcement**

   - The engine walks the sorted list and adds actions until it reaches
     `max_actions_per_tick` or runs out of candidates.
   - For each selected tensor, it:
     - Creates an appropriate `OffloadAction` (`MoveToRam` or `MoveToSsd`).
     - Updates `last_moved[id] = tick` to maintain cooldown semantics.

4. **Reason string**

   - Base reasons remain aligned with previous versions:
     - Both high: `"VRAM and RAM pressure high"`.
     - VRAM-only high: `"VRAM pressure high"` or
       `"VRAM pressure high but no offloadable tensors in VRAM"`.
     - RAM-only high: `"RAM pressure high"` or
       `"RAM pressure high but no offloadable tensors in RAM"`.
   - If cooldown skipped any tensors, the reason is extended with:
     - `"; some tensors skipped due to cooldown"`.
   - When there were any candidates (even if some were not selected due to
     the budget), the reason also includes a deterministic summary:

     - `"; Priority offloading enabled; selected N/M"`, where `N` is the
       number of actions planned and `M` is the total number of eligible
       candidates.

The stable-band / below-low behavior from 13.5.1 is unchanged: when
pressures are not above their high thresholds, `plan_with_tick` does not
introduce new offloads and returns either `"No offloading needed"` or
`"No new offloading due to hysteresis band"`.

### Tests: `tests/smart_offload_engine_priority_test.rs`

This test suite validates the priority-based behavior using a dedicated
cache directory `./.atenia_cache_test_offload_pri` and synthetic
`MemorySnapshot` instances (no real GPU required).

Covered scenarios:

- **`selects_largest_tensors_first_under_ram_pressure`**
  - Registers three RAM tensors with sizes 10, 100, and 50 bytes.
  - Sets `ram.pressure` above `ram_high`.
  - Sets `max_actions_per_tick = 2`.
  - Expects the plan to produce two `MoveToSsd` actions ordered as:
    - First: the 100-byte tensor.
    - Second: the 50-byte tensor.

- **`tie_breaks_by_id_when_same_size`**
  - Registers `tA` and `tB` with identical sizes.
  - Sets `max_actions_per_tick = 1`.
  - Expects the engine to select `tA` first, using the tensor id as a
    deterministic tie-breaker.

- **`respects_cooldown_even_if_large`**
  - Registers a small and a large RAM tensor.
  - At `tick = 10`, with high RAM pressure and `max_actions_per_tick = 1`,
    the engine selects the large tensor.
  - At `tick = 12`, still under high pressure but within the cooldown
    window, the large tensor is skipped and the small tensor is selected
    instead.

- **`reason_includes_priority_summary`**
  - Confirms that the `reason` string produced by `plan_with_tick` includes
    both:
    - The phrase `"Priority offloading enabled"`.
    - A `"selected"` summary indicating how many tensors were chosen.

Together, these tests confirm that:

- Larger tensors are offloaded first under pressure.
- Ties are broken deterministically by tensor id.
- The per-tick budget is enforced.
- Cooldown and hysteresis from earlier versions remain intact.

## APX 13.6.0 – Reconfigurable Graph (Dynamic Placement Scaffold)

Version **13.6.0** introduces a lightweight **ReconfigurableGraph** that
models a sequence of operation nodes and computes per-node placement for a
given memory snapshot. This version focuses on planning only:

- No op execution.
- No memory moves.
- No interaction with streams or the async executor.

Instead, it delegates per-node target decisions to the existing
`HybridExecutionPlanner` and aggregates results into a
`GraphPlacementPlan`.

### Graph model (`reconfigurable_graph.rs`)

The graph is intentionally simple and sequence-oriented:

- **`NodeId`**
  - Type alias for `u64` used to identify nodes within a graph.

- **`GraphNode`**

  ```rust
  pub struct GraphNode {
      pub id: NodeId,
      pub kernel: KernelProfile,
      pub tensor_tiers: Vec<MemoryTier>,
  }
  ```

  - `id` – unique identifier assigned by the graph when the node is added.
  - `kernel` – the `KernelProfile` describing the operation.
  - `tensor_tiers` – current memory tiers (e.g. RAM/VRAM/SSD) for the
    tensors referenced by this node.

- **`ReconfigurableGraph`**

  ```rust
  pub struct ReconfigurableGraph {
      nodes: Vec<GraphNode>,
      next_id: NodeId,
  }
  ```

  - `nodes` – ordered list of graph nodes.
  - `next_id` – monotonically increasing counter used to assign `NodeId`s.

The core API is:

```rust
impl ReconfigurableGraph {
    pub fn new() -> Self { ... }

    pub fn add_node(&mut self, kernel: KernelProfile, tensor_tiers: Vec<MemoryTier>) -> NodeId { ... }

    pub fn nodes(&self) -> &[GraphNode] { ... }
}
```

- `new` – creates an empty graph with `next_id = 0`.
- `add_node` – appends a node and returns its assigned `NodeId`.
- `nodes` – read-only access to the internal node list.

### Placement plan model

To represent the result of planning, the module defines:

- **`NodePlacement`**

  ```rust
  pub struct NodePlacement {
      pub node_id: NodeId,
      pub target: ExecutionTarget,
      pub reason: String,
  }
  ```

  - `node_id` – the id of the node this placement refers to.
  - `target` – `ExecutionTarget::Cpu`, `Gpu`, or `CpuFallback` as decided by
    the planner.
  - `reason` – human-readable explanation for the chosen target.

- **`GraphPlacementPlan`**

  ```rust
  pub struct GraphPlacementPlan {
      pub placements: Vec<NodePlacement>,
      pub snapshot_summary: String,
  }
  ```

  - `placements` – per-node placement decisions in the order of the graph.
  - `snapshot_summary` – English summary of the snapshot used for planning
    (e.g. memory pressures).

### Replanning API (`plan_for_snapshot`)

The main entrypoint for planning is:

```rust
pub fn plan_for_snapshot(
    &self,
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) -> GraphPlacementPlan
```

Behavior:

- Derives a snapshot summary using VRAM and RAM pressures:

  ```rust
  let vram_p = snapshot.vram.pressure.unwrap_or(0.0);
  let ram_p = snapshot.ram.pressure.unwrap_or(0.0);
  let snapshot_summary = format!(
      "Snapshot pressures: vram={:.4}, ram={:.4}",
      vram_p, ram_p,
  );
  ```

- For each `GraphNode` in `nodes`:
  - Calls `HybridExecutionPlanner::plan` with:
    - `kernel = &node.kernel`.
    - `tensor_tiers = &node.tensor_tiers`.
    - `snapshot` – the provided memory snapshot.
    - `gpu_available` – whether a GPU is available.
  - Creates a `NodePlacement` using the resulting `ExecutionTarget` and
    planner `reason`.

- Returns a `GraphPlacementPlan` with:
  - `placements` – all node placements.
  - `snapshot_summary` – the formatted pressures.

No caching is performed; repeated calls with different snapshots re-run
the planner per node without modifying the graph.

### Tests: `tests/reconfigurable_graph_test.rs`

This test file validates that the graph can be **replanned** under
different snapshots and that the resulting placements and summary are
deterministic and explainable.

Covered scenarios:

- **`graph_can_replan_with_different_snapshots`**
  - Builds a graph with two nodes:
    - Node 1: `KernelKind::ComputeHeavy`, tensor tiers in `MemoryTier::Ram`.
    - Node 2: `KernelKind::Small`, tensor tiers in `MemoryTier::Ram`.
  - Snapshot A: low VRAM pressure, GPU available.
    - Expects node 1 to target `ExecutionTarget::Gpu` and node 2 to target
      `ExecutionTarget::Cpu` (heavy work is offloaded to GPU when safe).
  - Snapshot B: high VRAM pressure (≥ 0.95), GPU still available.
    - Expects node 1 to target `CpuFallback` (or `Cpu` depending on the
      planner's rules for high VRAM pressure) while node 2 remains `Cpu`.
  - Asserts that the heavy node's target differs between A and B and that
    planner reasons are non-empty.

- **`tensor_on_ssd_forces_cpu_in_plan`**
  - Adds a node with `tensor_tiers` containing `MemoryTier::Ssd`.
  - Uses low-pressure snapshot with `gpu_available = true`.
  - Expects the planner to choose `ExecutionTarget::Cpu` and the reason to
    mention SSD in some form.

- **`snapshot_summary_contains_pressures`**
  - Uses a snapshot with specific pressures (e.g. `vram=0.42`, `ram=0.84`).
  - Verifies that `snapshot_summary` includes both the VRAM and RAM values
    and the corresponding labels.

Together, these tests confirm that the ReconfigurableGraph:

- Can be replanned across snapshots without mutating the graph.
- Properly delegates decisions to `HybridExecutionPlanner`.
- Produces deterministic per-node placement plans with human-readable
  reasons and a clear summary of the snapshot context.

## APX 13.6.1 – Executable Graph (Planner → Router → Streams)

Version **13.6.1** makes the `ReconfigurableGraph` **executable** by
connecting it to the stream router, async executor, and hybrid memory
manager. This version still does not run real kernels or spawn threads, but
it:

- Translates graph nodes into routed stream tasks.
- Applies real memory moves via `HybridMemoryManager` as part of routing.
- Records a global, deterministic timeline in `AsyncExecutor`.

### GraphNode tensor identifiers

To allow routing by tensor id, `GraphNode` is extended with explicit tensor
identifiers:

```rust
pub struct GraphNode {
    pub id: NodeId,
    pub kernel: KernelProfile,
    pub tensor_ids: Vec<String>,
    pub tensor_tiers: Vec<MemoryTier>,
}
```

Two construction APIs are available:

- `add_node(kernel: KernelProfile, tensor_tiers: Vec<MemoryTier>)` – legacy
  helper that auto-generates synthetic tensor ids based on node id and
  index (e.g. `"t0_0"`, `"t0_1"`).
- `add_node_with_tensors(kernel, tensor_ids, tensor_tiers)` – new API that
  accepts explicit ids and tiers.

The implementation pairs ids and tiers using the minimum of their lengths
to avoid out-of-bounds access, keeping behavior deterministic and safe.

### GraphExecutor (`graph_executor.rs`)

The new `GraphExecutor` type encapsulates graph-level execution planning
over streams:

```rust
pub struct GraphExecutor {
    pub cfg: StreamConfig,
}

impl GraphExecutor {
    pub fn new(cfg: StreamConfig) -> Self { ... }

    pub fn enqueue_graph(
        &self,
        graph: &ReconfigurableGraph,
        exec: &mut AsyncExecutor,
        mem: &mut HybridMemoryManager,
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> Vec<RoutedBundle> { ... }
}
```

- `cfg` – the same `StreamConfig` used by `AsyncExecutor` to pick advanced
  vs fallback behavior.
- `enqueue_graph` – walks the graph and enqueues all tasks into the
  executor.

### Enqueue semantics

`enqueue_graph` performs the following steps:

1. Iterates `graph.nodes()` in order. This preserves **global node order**
   for all enqueued tasks.

2. For each node, builds a list of tensor id references:

   ```rust
   let tensor_id_refs: Vec<&str> = node.tensor_ids.iter().map(|s| s.as_str()).collect();
   ```

3. Calls the existing memory-aware router:

   ```rust
   let bundle = StreamRouter::route_kernel_with_memory(
       exec,
       mem,
       &node.kernel,
       &tensor_id_refs,
       snapshot,
       gpu_available,
   );
   ```

   - This reuses all 13.4.2 logic for:
     - Injecting SSD prefetch tasks.
     - Planning/applying memory moves via `HybridMemoryManager`.
     - Enqueuing transfer and compute tasks.
     - Handling safe degradation (e.g. VRAM unavailable).

4. Collects the returned `RoutedBundle` for each node and returns a `Vec` of
   bundles to the caller.

As a result, the `AsyncExecutor.timeline` records a single, global
ENQUEUE/RUN sequence across all graph nodes while preserving per-node
ordering (prefetch → move → compute).

### Tests: `tests/executable_graph_test.rs`

This integration test suite validates the interaction between:

- `ReconfigurableGraph` (13.6.0),
- `StreamRouter::route_kernel_with_memory` (13.4.2),
- `AsyncExecutor` (13.4.0), and
- `HybridMemoryManager` (13.2.x).

Covered scenarios:

- **`graph_execution_enqueues_tasks_in_node_order`**
  - Registers `t1` as SSD-backed data and `t2` as RAM.
  - Builds a graph with two nodes:
    - Node 1 uses `t1` (SSD, compute-heavy kernel).
    - Node 2 uses `t2` (RAM, small kernel).
  - Uses low-pressure snapshot with `gpu_available = false` and
    `advanced_streams_supported = true`.
  - Calls `GraphExecutor::enqueue_graph` and inspects `AsyncExecutor.timeline`:
    - Asserts that all `ENQUEUE` entries for node 1 (prefetch + move +
      compute) appear **before** the compute `ENQUEUE` for node 2.

- **`graph_execution_updates_memory_tiers`**
  - Similar setup with `t1` on SSD and `t2` in RAM.
  - After `enqueue_graph`:
    - `mem.get_tier("t1") == Some(MemoryTier::Ram)` (SSD → RAM for CPU
      compute).
    - `mem.get_tier("t2") == Some(MemoryTier::Ram)`.
  - After `exec.run_to_completion()`:
    - Ensures that `RUN ... name=node1` and `RUN ... name=node2` entries
      are present in the timeline.

- **`gpu_plan_enqueues_gpu_compute_for_compute_heavy_node`**
  - Uses `HybridMemoryManager::new_with_vram` + a `FakeVramAdapter` to allow
    RAM ↔ VRAM moves without real hardware.
  - Registers a single tensor `t` in RAM and a compute-heavy kernel.
  - Under low pressure with `gpu_available = true`, calls `enqueue_graph`.
  - Assertions:
    - `mem.get_tier("t") == Some(MemoryTier::Vram)` after planning
      (RAM → VRAM).
    - Timeline contains:
      - `ENQUEUE stream=Gpu ... move:ram->vram:t`.
      - `ENQUEUE stream=Gpu kind=Compute name=gpu_node`.

Together, these tests confirm that:

- Graph execution is globally deterministic and respects node order.
- Memory tiers are updated via `HybridMemoryManager` as part of routing.
- GPU-capable nodes enqueue the expected GPU transfer + compute tasks when
  the planner chooses GPU execution.
- No real kernels or threads are involved; all behavior remains
  test-friendly and deterministic.

## APX 13.6.2 – Batch Loop (Graph + Offload Between Batches)

Version **13.6.2** introduces a **batch loop runner** that orchestrates
graph execution and smart offloading across a sequence of logical ticks.
This version still performs no real kernel work, but it:

- Executes the graph for each tick using `GraphExecutor` and
  `AsyncExecutor`.
- Runs `SmartOffloadEngine` between ticks to offload tensors based on
  memory pressures.
- Produces a deterministic, human-readable timeline describing both graph
  execution and offload decisions.

### BatchLoopRunner (`batch_loop.rs`)

The core type is:

```rust
pub struct BatchLoopRunner {
    pub offload: SmartOffloadEngine,
    pub graph_exec: GraphExecutor,
}
```

- `offload` – the smart offload engine configured with hysteresis,
  cooldown, priority scoring, and per-tick budget from 13.5.x.
- `graph_exec` – the `GraphExecutor` introduced in 13.6.1.

Construction:

```rust
impl BatchLoopRunner {
    pub fn new(offload: SmartOffloadEngine, graph_exec: GraphExecutor) -> Self { ... }
```

The main loop API is:

```rust
pub fn run_ticks(
    &mut self,
    graph: &ReconfigurableGraph,
    exec: &mut AsyncExecutor,
    mem: &mut HybridMemoryManager,
    snapshots: &[MemorySnapshot],
    gpu_available: bool,
) -> Vec<String>
```

- `snapshots` – ordered list of `MemorySnapshot` instances, one per tick.
- `gpu_available` – shared GPU availability flag for all ticks.
- Returns a clone of `exec.timeline` after running all ticks (or stopping
  early on an offload error).

### Per-tick behavior

For each snapshot at index `i` (`tick = i as u64`), the runner:

1. **Marks tick start**

   - Appends to `exec.timeline`:
     - `"TICK_START tick=<i>"`.

2. **Executes the graph for this tick**

   - Calls `graph_exec.enqueue_graph(graph, exec, mem, &snapshots[i], gpu_available)`.
   - Then calls `exec.run_to_completion()` to drain all enqueued tasks
     (prefetch, transfers, compute) according to the configured
     `StreamConfig`.

3. **Collects tensor ids for offloading**

   - Iterates `graph.nodes()` and builds a unique set of tensor ids across
     all nodes to pass into offload planning.

4. **Plans offloads using `SmartOffloadEngine`**

   - Builds a stable `Vec<&str>` of all unique tensor ids.
   - Calls:

     ```rust
     let plan = offload.plan_with_tick(&snapshots[i], &tensor_ids_refs, mem, tick);
     ```

   - Records:

     ```text
     OFFLOAD_PLAN tick=<i> actions=<N> reason=<...>
     ```

     where `N` is the number of planned offload actions.

5. **Applies offload plan**

   - If `N == 0`:
     - Appends `"OFFLOAD_APPLY tick=<i> skipped"`.
   - Otherwise:
     - Calls `offload.apply(&snapshots[i], &plan, mem)`.
       - On success:
         - Appends `"OFFLOAD_APPLY tick=<i> ok"`.
       - On error (`MoveError`):
         - Appends `"OFFLOAD_APPLY tick=<i> error=<...>"` and terminates
           the loop early.

6. **Marks tick end**

   - Appends `"TICK_END tick=<i>"` to the timeline.

All decisions remain purely logical and deterministic; there is no use of
wall-clock time or background threads.

### Tests: `tests/batch_loop_offload_test.rs`

This test file validates that the batch loop integrates graph execution
and smart offloading correctly between ticks, using a dedicated cache
directory `./.atenia_cache_test_batch_loop*`.

Covered scenarios:

- **`offload_runs_between_ticks_and_moves_ram_to_ssd_when_ram_high`**
  - Registers a tensor `t1` in RAM with real bytes.
  - Builds a graph with a single CPU node referencing `t1`.
  - Uses two snapshots:
    - Tick 0: low RAM pressure.
    - Tick 1: high RAM pressure (0.99).
  - Configures `SmartOffloadEngine` with `max_actions_per_tick >= 1`.
  - After `run_ticks`, asserts that:
    - `t1` has moved from RAM to SSD via offloading.
    - Timeline includes `TICK_START tick=0`, `TICK_END tick=0`,
      `OFFLOAD_PLAN tick=1 actions=1`, and `"OFFLOAD_APPLY tick=1 ok"`.

- **`hysteresis_stable_band_does_not_thrash_between_ticks`**
  - Registers `t2` in RAM and builds a graph node referencing it.
  - Uses snapshots with RAM pressure in the hysteresis **stable band**
    (e.g. 0.90, between low and high thresholds) for both tick 0 and 1.
  - After `run_ticks`, confirms that:
    - `t2` remains in RAM.
    - `OFFLOAD_PLAN tick=0` and `OFFLOAD_PLAN tick=1` both report
      `actions=0`.

- **`priority_budget_limits_actions_per_tick`**
  - Registers three RAM tensors with different sizes: `tA` (10 bytes),
    `tB` (100 bytes), `tC` (50 bytes).
  - Builds a graph with a single node referencing all three tensors.
  - Uses a single snapshot with high RAM pressure (0.99).
  - Configures `SmartOffloadEngine` with `max_actions_per_tick = 2`.
  - After `run_ticks`, asserts that:
    - `tB` and `tC` (the largest tensors) have been moved to SSD.
    - `tA` remains in RAM.

Together, these tests confirm that:

- Graph execution and offloading are coordinated per tick.
- Offload planning and application happen **between** graph executions.
- Hysteresis prevents unnecessary offload thrashing across ticks.
- Priority scoring and per-tick budget are respected in a multi-tensor
  scenario.

## APX 13.6.3 – Replan-on-Next-Tick (Placement Reacts to Tier Changes)

Version **13.6.3** extends the batch loop runner with **per-tick placement
introspection**. The goal is to demonstrate that:

- **Between-tick offloading** (RAM/VRAM → SSD) performed by
  `SmartOffloadEngine` on tick **N**
- **changes the planner's decision** on tick **N+1**, because tensor tiers
  have changed.

This version remains fully deterministic and uses only logical planners,
mock executors, and fake VRAM adapters. No real kernels, threads, or devices
are involved.

### Per-tick placement capture: `run_ticks_with_plans`

`batch_loop.rs` now provides an extended API alongside `run_ticks`:

```rust
pub struct TickResult {
    pub tick: u64,
    pub plan: GraphPlacementPlan,
}

impl BatchLoopRunner {
    pub fn run_ticks_with_plans(
        &mut self,
        graph: &ReconfigurableGraph,
        exec: &mut AsyncExecutor,
        mem: &mut HybridMemoryManager,
        snapshots: &[MemorySnapshot],
        gpu_available: bool,
    ) -> (Vec<String>, Vec<TickResult>) { ... }
}
```

Key points:

- **Signature**:
  - Accepts the same arguments as `run_ticks` plus the shared
    `HybridMemoryManager` and `ReconfigurableGraph`.
  - Returns a pair:
    - `Vec<String>` – a clone of the final `AsyncExecutor.timeline`.
    - `Vec<TickResult>` – per-tick placement plans.
- **Non-breaking**:
  - `run_ticks` is unchanged; existing callers can continue to use it.
  - `run_ticks_with_plans` is an additive, more introspectable API.

Per tick (for each snapshot index `i`, `tick = i as u64`), the runner:

1. **Computes a placement plan before execution**

   - Uses current memory tiers from `HybridMemoryManager` instead of the
     static tiers stored in the graph nodes:

   ```rust
   let plan = graph.plan_for_snapshot_with_mem(mem, snapshot, gpu_available);
   results.push(TickResult { tick, plan });
   ```

   - This ensures that any tier changes from previous ticks (e.g. RAM → SSD
     offload) affect the planner's decision on the next tick.

2. **Executes the graph and offloads exactly as in 13.6.2**

   - The rest of the tick is identical to `run_ticks`:
     - Append `"TICK_START tick=<i>"`.
     - Call `graph_exec.enqueue_graph(...)` and `exec.run_to_completion()`.
     - Collect tensor ids, plan offloads with `SmartOffloadEngine`.
     - Apply the offload plan and log:
       - `OFFLOAD_PLAN tick=<i> actions=<N> reason=<...>`.
       - `OFFLOAD_APPLY tick=<i> ok` / `error=<...>` / `skipped`.
     - Append `"TICK_END tick=<i>"`.

3. **Stops early on offload error**

   - As in 13.6.2, if `offload.apply` returns a `MoveError`, the loop stops
     and the partially collected `TickResult`s and timeline are returned.
   - This preserves deterministic error behavior.

### Dynamic planning from live memory: `plan_for_snapshot_with_mem`

`reconfigurable_graph.rs` introduces an extended planning helper that derives
tensor tiers from the **live** `HybridMemoryManager` instead of the static
tiers stored on each node:

```rust
impl ReconfigurableGraph {
    pub fn plan_for_snapshot_with_mem(
        &self,
        mem: &HybridMemoryManager,
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> GraphPlacementPlan { ... }
}
```

Behavior:

- Builds a human-readable `snapshot_summary` similar to
  `plan_for_snapshot`, using `snapshot.vram.pressure` and
  `snapshot.ram.pressure`.
- For each `GraphNode`:
  - Iterates `node.tensor_ids` and calls `mem.get_tier(id)`.
  - Falls back to `MemoryTier::Ram` if the id is unknown.
  - Calls `HybridExecutionPlanner::plan(&node.kernel, &tiers, snapshot, gpu_available)`.
  - Records `ExecutionTarget` and `reason` into `NodePlacement`.
- Returns a `GraphPlacementPlan` containing all placements and the
  snapshot summary.

The existing `plan_for_snapshot(snapshot, gpu_available)` API remains
unchanged and uses the static `tensor_tiers` recorded on each node. The new
method is additive and is used exclusively by `run_ticks_with_plans`.

### Tests: `tests/replan_next_tick_test.rs`

This test file validates that **between-tick offloading changes the planner's
decision on a later tick**, and that this change is observable via
`run_ticks_with_plans`.

Covered scenario:

- **`offload_changes_next_tick_placement`**

  - Uses `HybridMemoryManager::new_with_vram` and a `FakeVramAdapter` so
    that RAM ↔ VRAM moves are fully in-memory and deterministic.
  - Registers a single tensor `t1` in RAM with a small, real byte buffer.
  - Builds a `ReconfigurableGraph` with one compute-heavy node that uses
    `t1`.
  - Configures `SmartOffloadEngine` with:
    - `cooldown_ticks = 0` to allow back-to-back decisions.
    - `max_actions_per_tick >= 1` to allow offloading on the high-pressure
      tick.
  - Uses three `MemorySnapshot`s:
    - **Tick 0**: low VRAM and RAM pressure (e.g. 0.10 / 0.10).
    - **Tick 1**: high VRAM and RAM pressure (e.g. 0.99 / 0.99),
      triggering offloads to SSD.
    - **Tick 2**: low pressure again, but tiers may have changed due to
      tick 1 offload.
  - Calls:

    ```rust
    let (timeline, tick_results) = runner.run_ticks_with_plans(
        &graph,
        &mut exec,
        &mut mem,
        &snapshots,
        true, // gpu_available
    );
    ```

  - Asserts that:
    - `tick_results[0].plan.placements[0].target == ExecutionTarget::Gpu`:
      - At tick 0, `t1` is in RAM, GPU is available, and pressure is low, so
        the planner prefers GPU.
    - `tick_results[2].plan.placements[0].target` is `Cpu` or
      `CpuFallback`, and its `reason` string mentions `ssd`:
      - By tick 2, `SmartOffloadEngine` has offloaded `t1` to SSD under
        high pressure at tick 1.
      - The planner now "sees" SSD residency via `plan_for_snapshot_with_mem`
        and therefore chooses a CPU path.
    - The timeline includes `OFFLOAD_PLAN` and `OFFLOAD_APPLY` entries for
      tick 1 with `actions > 0` and `ok`, confirming that offload actually
      ran between tick 0 and tick 2.

  - This establishes a clear, testable story:
    - **Tick 0**: GPU-friendly placement (tensor in RAM).
    - **Tick 1**: high pressure → offload `t1` to SSD.
    - **Tick 2**: planner detects SSD residency and falls back to CPU.

Together, these additions show that APX 13.6.x is not only capable of
executing graphs and offloading tensors between ticks, but also of
**replanning future ticks based on the new memory reality** created by those
offloads. The system remains deterministic, explainable (through timeline and
human-readable reasons), and fully test-driven.

## APX 13.7.0 – Unified Hybrid Autograd (Scaffold)

Version **13.7.0** introduces a **unified autograd scaffold** that operates on
top of the existing hybrid execution and memory abstractions. The goals are
to:

- Track backward computations as abstract nodes with placement decisions.
- Decide where to compute each gradient (CPU vs GPU) based on tensor tiers
  and GPU availability.
- Move gradients between memory tiers when required.
- Enqueue backward work as deterministic `AsyncExecutor` tasks, without
  performing any real kernel execution.

All non-code text remains in English, behavior is fully deterministic, and no
new crates or real hardware dependencies are introduced.

### Core types: gradients and autograd graph (`autograd.rs`)

The autograd module defines gradient identifiers and nodes that reference
existing placement and graph concepts:

```rust
pub type GradId = String;

#[derive(Debug, Clone)]
pub struct TensorGrad {
    pub id: GradId,
    pub tier: MemoryTier,
}

#[derive(Debug, Clone)]
pub struct AutogradNode {
    pub node_id: NodeId,
    pub forward_target: ExecutionTarget,
    pub backward_target: ExecutionTarget,
    pub input_grads: Vec<TensorGrad>,
    pub output_grads: Vec<TensorGrad>,
    pub reason: String,
}

pub struct AutogradGraph {
    nodes: Vec<AutogradNode>,
}

impl AutogradGraph {
    pub fn new() -> Self { ... }
    pub fn add_node(&mut self, node: AutogradNode) { ... }
    pub fn nodes(&self) -> &[AutogradNode] { ... }
}
```

Key points:

- `TensorGrad` mirrors the existing `MemoryTier` abstraction: each gradient has
  an id and a current tier (Ram/Vram/Ssd/Cpu).
- `AutogradNode` references a `NodeId` from `ReconfigurableGraph` and stores:
  - `forward_target` – where the forward op ran (Cpu/Gpu/CpuFallback).
  - `backward_target` – planned target for backward (currently informational;
    the effective target is decided at execution time).
  - `input_grads` / `output_grads` – gradient tensors flowing through the
    node.
  - `reason` – human-readable explanation for placement decisions.
- `AutogradGraph` is a simple ordered collection; backward is executed in
  reverse node order.

### Backward placement rules

Placement for backward nodes is decided by a helper:

```rust
fn decide_backward_target(
    forward_target: ExecutionTarget,
    grad_tiers: &[MemoryTier],
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) -> ExecutionTarget { ... }
```

Rules (v1):

- If **all** of the following hold:
  - `forward_target == ExecutionTarget::Gpu`,
  - every entry in `grad_tiers` is `MemoryTier::Vram`, and
  - `gpu_available == true`,
  - then `backward_target = ExecutionTarget::Gpu`.
- Otherwise:
  - `backward_target = ExecutionTarget::Cpu`.

This ensures that backward only runs on GPU when both the forward path was GPU
and all gradients are already resident in VRAM. Any mixed or non-VRAM
gradients automatically fall back to CPU, avoiding implicit transfers.

### Gradient preparation: tier moves via `HybridMemoryManager`

Before enqueuing a backward compute task, gradients are prepared for the
chosen target using real tier moves handled by `HybridMemoryManager`:

```rust
pub fn prepare_gradients(
    mem: &mut HybridMemoryManager,
    grads: &[TensorGrad],
    target: ExecutionTarget,
    snapshot: &MemorySnapshot,
) -> Result<(), MoveError> { ... }
```

Behavior:

- If `target == ExecutionTarget::Gpu`:
  - All gradients must end up in `MemoryTier::Vram`.
- If `target == ExecutionTarget::Cpu` or `CpuFallback`:
  - All gradients must end up in `MemoryTier::Ram`.

Implementation details:

- For each `TensorGrad`:
  - Queries `mem.get_tier(id)`.
  - If the tensor is already in the desired tier, it is skipped.
  - Otherwise, calls `mem.plan_move(id, desired_tier, snapshot)` followed by
    `mem.apply_move(id, &plan)`.
- Any `MoveError` is propagated to the caller so that higher-level logic can
  decide whether to fall back to CPU.

All moves use the existing RAM/VRAM/SSD logic from 13.2.x, including
compression-aware SSD operations and VRAM adapter fallbacks.

### Backward execution as abstract tasks

Backward execution is orchestrated by:

```rust
pub fn execute_backward(
    exec: &mut AsyncExecutor,
    mem: &mut HybridMemoryManager,
    graph: &AutogradGraph,
    snapshot: &MemorySnapshot,
    gpu_available: bool,
)
```

Per-node behavior:

1. **Reverse traversal**

   - Iterates `graph.nodes()` in reverse order, ensuring that later
     autograd nodes run first, matching typical backward semantics.

2. **Decide backward target**

   - Builds `grad_tiers` by collecting the `tier` of all `input_grads` and
     `output_grads`.
   - Calls `decide_backward_target(node.forward_target, &grad_tiers, snapshot, gpu_available)`.

3. **Prepare gradients & GPU fallback**

   - If the candidate target is `Gpu`:
     - Calls `prepare_gradients` for all gradients, aiming for VRAM.
     - On any `MoveError`:
       - Records a timeline entry:

         ```text
         BACKWARD_FALLBACK node=<id> from=Gpu to=Cpu
         ```

       - Falls back to `ExecutionTarget::Cpu` for this node.
   - If the final target is CPU/CPUFallback:
     - Calls `prepare_gradients` with `ExecutionTarget::Cpu`, moving gradients
       to RAM as needed.
     - Errors here are ignored to keep behavior simple and avoid panics; the
       system remains best-effort but deterministic.

4. **Enqueue abstract compute task**

   - Chooses `StreamKind::Gpu` if the final `ExecutionTarget` is `Gpu`,
     otherwise `StreamKind::Cpu`.
   - Submits a compute task to the `AsyncExecutor`:

   ```rust
   let name = format!("backward:node{}", node.node_id);
   exec.submit(stream, TaskKind::Compute { name: name.clone() }, 1);
   ```

   - Appends a human-readable summary to `exec.timeline`:

   ```text
   BACKWARD node=<id> target=<Cpu|Gpu|CpuFallback> reason=<node.reason>
   ```

`execute_backward` does **not** call `run_to_completion`; callers can choose
when and how to drain the executor queues.

### Tests: `tests/hybrid_autograd_test.rs`

The autograd scaffold is validated by three deterministic tests using a
dedicated cache directory `./.atenia_cache_test_autograd` and a `FakeVramAdapter`.

Covered scenarios:

- **`backward_runs_on_gpu_when_grads_in_vram`**

  - Uses `HybridMemoryManager::new_with_vram` plus `FakeVramAdapter`.
  - Registers gradients `g1`, `g2` in RAM with real data, then moves them to
    VRAM using `plan_move` + `apply_move`.
  - Builds an `AutogradGraph` with a single node:
    - `node_id = 1`.
    - `forward_target = ExecutionTarget::Gpu`.
    - `input_grads = [g1, g2]` with `tier = Vram`.
  - Under low pressure with `gpu_available = true`, calls `execute_backward`.
  - Asserts that `AsyncExecutor.timeline` contains an `ENQUEUE` entry on the
    GPU stream for `backward:node1`.

- **`backward_falls_back_to_cpu_when_grad_on_ram`**

  - Registers `g1` and `g2` in RAM, then moves only `g2` to VRAM
    (`g1` remains in RAM, `g2` is in VRAM).
  - Builds a node with `forward_target = ExecutionTarget::Gpu` and mixed
    gradient tiers (RAM + VRAM).
  - Calls `execute_backward` with `gpu_available = true`.
  - Because not all gradients are in VRAM, `decide_backward_target` selects
    CPU.
  - Asserts that:
    - There is an `ENQUEUE` entry for `backward:node2` on the CPU stream.
    - There is **no** GPU `ENQUEUE` for `backward:node2`.

- **`backward_moves_grads_to_ram_for_cpu`**

  - Registers gradients `g1`, `g2` logically in SSD using
    `HybridMemoryManager::register_tensor` with `MemoryTier::Ssd` (no
    filesystem writes needed for this test).
  - Builds a node with `forward_target = ExecutionTarget::Cpu` and all
    gradients starting in `Ssd`.
  - Calls `execute_backward`.
  - Asserts that:
    - After execution, both `g1` and `g2` have `mem.get_tier(...) == Ram`,
      confirming logical SSD → RAM moves for CPU backward.
    - Timeline contains an `ENQUEUE` on the CPU stream for `backward:node3`.

Together, these tests confirm that the APX 13.7.0 autograd scaffold:

- Makes consistent, deterministic backward placement decisions.
- Uses `HybridMemoryManager` for real tier transitions (RAM/VRAM/SSD) when
  needed.
- Falls back cleanly to CPU when gradients are not fully in VRAM or when
  gradient preparation for GPU fails.
- Enqueues abstract backward work in the existing `AsyncExecutor` with clear
  timeline entries, while remaining entirely kernel-free and
  CI-friendly.

## APX 13.7.1 – Autograd Trace (Deterministic Debugging)

Version **13.7.1** extends the autograd scaffold with a **structured,
deterministic trace** of backward behavior. The goal is to make it easy to
audit, in tests and logs:

- Which backward target (CPU/GPU) was requested and which was actually used.
- Which gradient moves were required for a node.
- Whether those moves were applied, skipped, or failed.
- When and why a GPU plan fell back to CPU.

The existing `execute_backward` API remains unchanged for compatibility; new
types and an additional `execute_backward_with_trace` function are introduced
to expose richer debugging information.

### Trace types: moves and per-node traces

`autograd.rs` defines the following trace structures:

```rust
#[derive(Debug, Clone)
pub enum GradMoveRecord {
    Planned { grad_id: String, from: MemoryTier, to: MemoryTier },
    Applied { grad_id: String, from: MemoryTier, to: MemoryTier },
    Skipped { grad_id: String, reason: String },
    Failed  { grad_id: String, reason: String },
}

#[derive(Debug, Clone)]
pub struct AutogradNodeTrace {
    pub node_id: NodeId,
    pub forward_target: ExecutionTarget,
    pub requested_backward_target: ExecutionTarget,
    pub final_backward_target: ExecutionTarget,
    pub reason: String,
    pub moves: Vec<GradMoveRecord>,
}

#[derive(Debug, Clone)]
pub struct AutogradTrace {
    pub nodes: Vec<AutogradNodeTrace>,
}
```

Semantics:

- `GradMoveRecord` captures the lifecycle of each gradient move:
  - `Planned` – a move from `from` to `to` was deemed necessary.
  - `Applied` – `plan_move` + `apply_move` completed successfully.
  - `Skipped` – no move was attempted (e.g. already in target tier, or tensor
    not registered).
  - `Failed` – either `plan_move` or `apply_move` returned an error; the
    `reason` string contains the debug message.
- `AutogradNodeTrace` aggregates all move records for a single node, along
  with:
  - The original forward target.
  - The **requested** backward target from `decide_backward_target`.
  - The **final** backward target after any fallbacks.
  - A human-readable `reason` (possibly extended with fallback notes).
- `AutogradTrace` is a collection of per-node traces, ordered in the same
  reverse order used by backward execution.

### Executing backward with trace

The new non-breaking API is:

```rust
pub fn execute_backward_with_trace(
    exec: &mut AsyncExecutor,
    mem: &mut HybridMemoryManager,
    graph: &AutogradGraph,
    snapshot: &MemorySnapshot,
    gpu_available: bool,
) -> AutogradTrace
```

This function mirrors the semantics of `execute_backward` but additionally
constructs and returns an `AutogradTrace`, while also emitting compact
`AUTOGRAD` lines into the executor timeline.

Per-node behavior:

1. **Determine requested backward target**

   - Collects `grad_tiers` from all input and output gradients.
   - Calls `decide_backward_target(node.forward_target, &grad_tiers, snapshot, gpu_available)`.
   - Stores the result as `requested_backward_target`.

2. **Plan and apply moves for requested target**

   - Flattens all gradients into a single list for this node.
   - For GPU target:
     - Desired tier is `MemoryTier::Vram`.
   - For CPU/CPUFallback target:
     - Desired tier is `MemoryTier::Ram`.
   - For each gradient:
     - Reads current tier from `HybridMemoryManager`.
     - If the tensor is not registered:
       - Records `Skipped { reason="tensor not registered in memory manager" }`.
     - If current tier already equals the desired tier:
       - Records `Skipped { reason="already in target tier" }`.
     - Otherwise:
       - Records a `Planned` move.
       - Calls `mem.plan_move` and `mem.apply_move`:
         - On success, records `Applied { from, to }`.
         - On error, records `Failed { reason=... }` and marks the node as
           having move failures for this target.

3. **GPU fallback on move failure**

   - If the requested target was `Gpu` and **any** gradient move failed:
     - Sets `final_backward_target = ExecutionTarget::Cpu`.
     - Extends or sets the `reason` string with
       `"GPU fallback due to move failure"`.
     - Attempts to move all gradients to RAM using the same move-recording
       helper, adding any new `Planned/Applied/Skipped/Failed` entries.

4. **Enqueue compute task and log summary**

   - Chooses `StreamKind::Gpu` if `final_backward_target` is `Gpu`, otherwise
     `StreamKind::Cpu`.
   - Submits a compute task `backward:node<id>` to `AsyncExecutor`.
   - Appends a compact summary to `exec.timeline`:

   ```text
   AUTOGRAD node=<id> target=<Cpu|Gpu> moves=<N> fallback=<0|1>
   ```

   where `N` is the total number of `GradMoveRecord`s for this node, and
   `fallback=1` if the node requested GPU but ended up on CPU.

5. **Build node trace**

   - Pushes an `AutogradNodeTrace` into `AutogradTrace.nodes` with:
     - `node_id`, `forward_target` from the autograd node.
     - `requested_backward_target`, `final_backward_target`.
     - Final `reason` string.
     - Collected `moves`.

The resulting `AutogradTrace` provides a complete, structured view of backward
behavior per node, while `AsyncExecutor.timeline` offers a concise, readable
summary.

### Tests: `tests/hybrid_autograd_trace_test.rs`

This file validates that backward tracing is deterministic and captures all
relevant information. It uses a dedicated cache directory
`./.atenia_cache_test_autograd_trace` and two VRAM adapters:

- `FakeVramAdapter` – successful RAM ↔ VRAM transitions.
- `FailingVramAdapter` – simulates upload/download failures to force
  fallbacks.

Covered scenarios:

- **`trace_records_moves_and_target_gpu_success`**

  - Uses `HybridMemoryManager::new_with_vram` and `FakeVramAdapter`.
  - Registers gradients `g1`, `g2` in RAM with real data, then marks their
    logical tiers as `Vram` in `TensorGrad` to request GPU backward.
  - Under low pressure with `gpu_available = true`, calls
    `execute_backward_with_trace`.
  - Asserts that:
    - `requested_backward_target == Gpu` and `final_backward_target == Gpu`.
    - `AutogradNodeTrace.moves` contains `Planned` and `Applied` records for
      both `g1` and `g2` with `from=Ram`, `to=Vram`.
    - Executor timeline contains an `AUTOGRAD node=1` line with
      `target=Gpu` and `fallback=0`.

- **`trace_records_fallback_when_gpu_move_fails`**

  - Uses `HybridMemoryManager::new_with_vram` and `FailingVramAdapter`, which
    always returns `MoveError::BackendUnavailable` on upload/download.
  - Registers `g1`, `g2` in RAM and marks gradients as `tier=Vram` to request
    GPU.
  - Calls `execute_backward_with_trace` under low pressure with
    `gpu_available = true`.
  - Asserts that:
    - `requested_backward_target == Gpu` but `final_backward_target == Cpu`.
    - `reason` string contains the word `fallback`.
    - At least one `GradMoveRecord::Failed` exists with a message referencing
      the simulated failure.
    - Timeline contains `AUTOGRAD node=2 ... target=Cpu ... fallback=1`.

- **`trace_skips_moves_when_already_in_target`**

  - Registers `g1`, `g2` logically in RAM and marks their gradient tiers as
    `Ram` with `forward_target = Cpu`.
  - Calls `execute_backward_with_trace` under low pressure.
  - Asserts that:
    - Both `requested_backward_target` and `final_backward_target` are `Cpu`.
    - `moves` is non-empty and **all** entries are `Skipped` with
      `reason` containing `"already in target tier"`.
    - Timeline contains `AUTOGRAD node=3 ... target=Cpu ... fallback=0`.

Together, these additions turn the APX 13.7 autograd system into an
explainable, debuggable component: every backward decision and gradient move
can be inspected both in structured traces and in compact timeline entries,
while maintaining deterministic, test-friendly behavior.

## APX 13.8.0 – Persistent Hybrid Cache (SSD-backed)

Version **13.8.0** introduces a **persistent, SSD-backed cache** for storing
binary blobs (tensors, gradients, kernel metadata) together with a small
metadata file. The design is:

- Deterministic and vendor-agnostic (operates purely on bytes + metadata).
- Backed by a configurable root directory on the local filesystem.
- Integrity-checked via a simple 32-bit checksum (not cryptographic).
- Safe by default: never overwrites existing entries unless explicitly
  requested.

All non-code text remains in English and no new crates are used.

### Core types and errors (`persistent_cache.rs`)

The cache module defines the following types:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    Tensor,
    Gradient,
    KernelMeta,
}

#[derive(Debug, Clone)]
pub struct CacheEntryMeta {
    pub kind: CacheKind,
    pub len_bytes: usize,
    pub checksum32: u32,
    pub created_unix: u64,
}

#[derive(Debug, Clone)]
pub enum CacheError {
    Io(String),
    NotFound,
    Corrupt(String),
    AlreadyExists,
}

pub struct PersistentHybridCache {
    root: PathBuf,
}
```

Key points:

- `CacheKind` partitions the cache into separate namespaces for tensors,
  gradients, and kernel metadata.
- `CacheEntryMeta` stores minimal information needed for integrity and
  auditing:
  - `kind` – the logical kind.
  - `len_bytes` – expected byte length of the payload.
  - `checksum32` – simple 32-bit checksum over the payload.
  - `created_unix` – logical timestamp provided by the caller (no wall-clock
    dependency).
- `CacheError` distinguishes IO failures, corrupt entries, and attempted
  overwrites.

### API and storage layout

The main API on `PersistentHybridCache` is:

```rust
impl PersistentHybridCache {
    pub fn new(root: impl Into<PathBuf>) -> Self { ... }

    pub fn ensure_root(&self) -> Result<(), CacheError> { ... }

    pub fn put_blob(
        &self,
        kind: CacheKind,
        key: &str,
        bytes: &[u8],
        created_unix: u64,
        overwrite: bool,
    ) -> Result<(), CacheError> { ... }

    pub fn get_blob(&self, kind: CacheKind, key: &str) -> Result<Vec<u8>, CacheError> { ... }

    pub fn exists(&self, kind: CacheKind, key: &str) -> bool { ... }
}
```

Storage layout is stable and human-inspectable:

```text
<root>/<kind>/<key>.bin
<root>/<kind>/<key>.meta
```

where `<kind>` is one of:

- `tensor`
- `gradient`
- `kernel_meta`

The `.bin` file contains the raw payload bytes. The `.meta` file is plain
text with `key=value` lines:

```text
kind=tensor|gradient|kernelmeta|kernel_meta
len=123
checksum32=...
created_unix=...
```

Unknown or malformed keys/values in the meta file cause a `CacheError::Corrupt`
on read, making the format strict and easy to validate.

### Checksum and integrity

To validate integrity, the cache computes a simple deterministic 32-bit
checksum over the payload bytes:

```rust
fn checksum32(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811C9DC5; // FNV offset basis
    let prime: u32 = 0x01000193;    // FNV prime

    for b in data {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(prime);
    }

    hash
}
```

This is an FNV-1a-like checksum implemented without external crates. It is not
cryptographic, but is sufficient to detect common corruption (bit flips,
truncation, accidental modifications).

On `put_blob`, the cache:

- Ensures the root and per-kind subdirectories exist (`ensure_root` +
  `create_dir_all`).
- If `overwrite == false` and the `.bin` file already exists, returns
  `CacheError::AlreadyExists` without modifying any files.
- Writes the `.bin` payload to disk.
- Computes `len_bytes` and `checksum32`, then writes the `.meta` file using
  the stable key/value format.

On `get_blob`, the cache:

- Verifies that both `.bin` and `.meta` exist; otherwise returns
  `CacheError::NotFound`.
- Parses the meta file strictly (rejecting unknown keys or invalid numbers).
- Reads the `.bin` file into memory.
- Compares:
  - `buf.len()` against `meta.len_bytes` and
  - `checksum32(&buf)` against `meta.checksum32`.
- If either check fails, returns `CacheError::Corrupt` with a descriptive
  message.

`exists(kind, key)` simply tests for the presence of both `.bin` and `.meta`.

### Tests: `tests/persistent_hybrid_cache_test.rs`

This test file verifies that the persistent cache obeys the API contract and
detects corruption reliably. Each test uses its own dedicated root directory
under `./.atenia_cache_test_persistent_*` to avoid interference.

Covered scenarios:

- **`put_and_get_roundtrip_tensor`**

  - Uses root `./.atenia_cache_test_persistent_roundtrip`.
  - Calls `put_blob(CacheKind::Tensor, "w1", [1,2,3], created_unix=1234, overwrite=false)`.
  - Asserts that `exists(Tensor, "w1")` is `true`.
  - Calls `get_blob(Tensor, "w1")` and verifies that the returned bytes equal
    the original payload.

- **`put_rejects_overwrite_by_default`**

  - Uses root `./.atenia_cache_test_persistent_overwrite`.
  - Writes a tensor entry `k1` once.
  - Attempts to write `k1` again with `overwrite=false`.
  - Asserts that the second call returns `CacheError::AlreadyExists`.

- **`get_detects_corruption_by_checksum`**

  - Uses root `./.atenia_cache_test_persistent_corrupt`.
  - Writes a tensor entry `c1` with a small payload.
  - Manually opens `<root>/tensor/c1.bin` and appends an extra byte to corrupt
    the file.
  - Calls `get_blob(Tensor, "c1")` and asserts that it returns
    `CacheError::Corrupt` with a message mentioning a mismatch (length or
    checksum).

- **`separate_kinds_use_separate_namespaces`**

  - Uses root `./.atenia_cache_test_persistent_kinds`.
  - Writes two entries with the same key `"k"` but different kinds and
    payloads:
    - `Tensor` with bytes `[1,2,3]`.
    - `Gradient` with bytes `[9,8,7]`.
  - Asserts that `exists(Tensor, "k")` and `exists(Gradient, "k")` are both
    `true`.
  - Calls `get_blob` for each kind and verifies that each returns its own
    payload, confirming that per-kind namespaces are independent.

Together, these pieces form a simple but robust foundation for future
checkpointing and self-training features: a deterministic, SSD-backed cache
with explicit integrity checks and clear, debuggable on-disk layout.

## APX 13.8.1 – HybridMemory ↔ PersistentHybridCache Integration

Version **13.8.1** connects the in-memory hybrid tier model with the
persistent SSD-backed cache introduced in 13.8.0. The goal is to make SSD tier
data **backed by cache keys** when a persistent cache is attached, while
keeping all legacy path-based behavior intact when it is not.

Key properties:

- No public API changes: `HybridMemoryManager` exposes a new optional helper,
  but existing methods remain stable.
- SSD tier continues to use `SsdCache` and `StorageBacking::SsdFile` for
  compatibility.
- When a `PersistentHybridCache` is attached, every RAM/VRAM → SSD move also
  writes to the persistent cache using a deterministic key.
- When SSD → RAM moves occur, data is restored correctly regardless of whether
  the persistent cache is attached.

### Extending HybridMemoryManager with an optional persistent cache

`hybrid_memory.rs` now imports the persistent cache types and adds an optional
field to `HybridMemoryManager`:

```rust
use super::persistent_cache::{CacheKind, PersistentHybridCache};

pub struct HybridMemoryManager {
    tensors: HashMap<String, TensorResidence>,
    cache: SsdCache,
    vram: Box<dyn VramAdapter + Send + Sync>,
    persistent: Option<PersistentHybridCache>,
}
```

The constructors remain unchanged except for initializing the new field:

```rust
pub fn new_with_vram(cache_dir: &str, vram: Box<dyn VramAdapter + Send + Sync>) -> Self {
    HybridMemoryManager {
        tensors: HashMap::new(),
        cache: SsdCache::new(cache_dir),
        vram,
        persistent: None,
    }
}
```

Callers can opt into persistence via:

```rust
impl HybridMemoryManager {
    pub fn attach_persistent_cache(&mut self, cache: PersistentHybridCache) {
        self.persistent = Some(cache);
    }
}
```

This keeps the default behavior (no persistent cache) identical to prior
versions.

### Cache-aware RAM → SSD moves

When applying a move from RAM to SSD, `HybridMemoryManager::apply_move` still
uses `SsdCache` to write an on-disk blob and transitions the backing to
`StorageBacking::SsdFile`. With a persistent cache attached, it now also
persists the same bytes under a deterministic cache key.

Relevant match arm:

```rust
match (residence.tier, &mut residence.backing, plan.to) {
    (MemoryTier::Ram, StorageBacking::Ram(data), MemoryTier::Ssd) => {
        self.cache.ensure_dir()?;
        let path = self.cache.blob_path(id);
        let meta = self
            .cache
            .write_blob(&path, data, CompressionKind::None)?;

        if let Some(pcache) = &self.persistent {
            let logical_len = data.len();
            let key = format!("tensor:{}:len{}", id, logical_len);
            let _ = pcache.put_blob(
                CacheKind::Tensor,
                &key,
                data,
                0,
                true,
            );
        }

        *data = Vec::new();
        residence.backing = StorageBacking::SsdFile {
            path,
            compression: Some(meta),
        };
        residence.tier = MemoryTier::Ssd;
        Ok(())
    }
    // ...
}
```

Notes:

- The key format is simple and deterministic: `"tensor:<id>:len<bytes>"`.
- `put_blob` is called **best-effort**: errors are ignored to preserve the
  core memory move behavior even if the persistent cache root is not writable.
- The SSD backing remains `SsdFile`, so all existing tests and code paths that
  rely on `SsdCache` continue to work.

### Cache-aware VRAM → SSD moves

VRAM → SSD moves are similarly extended to mirror RAM → SSD behavior:

```rust
// VRAM -> SSD: download from VRAM, write to SSD (uncompressed), free VRAM.
(MemoryTier::Vram, StorageBacking::VramHandle { key }, MemoryTier::Ssd) => {
    if !self.vram.is_available() {
        return Err(MoveError::BackendUnavailable(
            "VRAM not available for download".to_string(),
        ));
    }

    let bytes = self.vram.download(key)?;
    residence.footprint.validate_len(bytes.len())?;
    self.cache.ensure_dir()?;
    let path = self.cache.blob_path(id);
    let meta = self
        .cache
        .write_blob(&path, &bytes, CompressionKind::None)?;

    if let Some(pcache) = &self.persistent {
        let key = format!("tensor:{}:len{}", id, bytes.len());
        let _ = pcache.put_blob(
            CacheKind::Tensor,
            &key,
            &bytes,
            0,
            true,
        );
    }

    let _ = self.vram.free(key);
    residence.backing = StorageBacking::SsdFile {
        path,
        compression: Some(meta),
    };
    residence.tier = MemoryTier::Ssd;
    Ok(())
}
```

Again, cache persistence is best-effort and does not alter the visible
backing representation.

### SSD → RAM moves and legacy behavior

SSD → RAM moves continue to rely on `SsdCache` for reading and deleting blob
files:

```rust
(MemoryTier::Ssd, StorageBacking::SsdFile { path, compression }, MemoryTier::Ram) => {
    let bytes = match compression {
        Some(meta) => self.cache.read_blob_with_meta(path, meta)?,
        None => self.cache.read_blob(path)?,
    };
    residence.footprint.validate_len(bytes.len())?;
    self.cache.delete_blob(path)?;
    residence.backing = StorageBacking::Ram(bytes);
    residence.tier = MemoryTier::Ram;
    Ok(())
}
```

This keeps the read path fully backward-compatible. The persistent cache
serves as an additional, durable store that higher-level components (e.g.
checkpointing) can rely on, but `HybridMemoryManager` itself does not change
its public semantics for SSD → RAM.

### Tests: `tests/hybrid_memory_persistent_cache_integration_test.rs`

The integration between `HybridMemoryManager` and `PersistentHybridCache` is
validated by three deterministic tests using
`./.atenia_cache_test_persistent_integration_*` roots.

- **`ram_to_ssd_uses_persistent_cache_when_attached`**

  - Constructs a `HybridMemoryManager` with a `FakeVramAdapter` and attaches a
    `PersistentHybridCache` via `attach_persistent_cache`.
  - Registers tensor `t1` in RAM with bytes `[1,2,3,4]`.
  - Plans and applies a move RAM → SSD.
  - Asserts that:
    - The tier is `MemoryTier::Ssd`.
    - The backing is `StorageBacking::SsdFile`.
    - The deterministic key `tensor:t1:len4` exists in the `Tensor` namespace
      of the persistent cache.
    - `cache.get_blob(CacheKind::Tensor, key)` returns the original bytes.

- **`ssd_to_ram_reads_from_cache_and_restores_bytes`**

  - Uses the same setup with an attached cache.
  - Performs a full roundtrip RAM → SSD → RAM for `t1`.
  - After SSD → RAM, asserts that:
    - The tier is `MemoryTier::Ram`.
    - `backing_for_test("t1")` is `StorageBacking::Ram` containing bytes equal
      to the original payload.

- **`works_without_cache_attached`**

  - Constructs a `HybridMemoryManager` without attaching a persistent cache.
  - Performs RAM → SSD → RAM for `t2`.
  - Asserts that the roundtrip succeeds and the final RAM backing matches the
    original bytes, proving that legacy `SsdCache` behavior is unchanged.

Together, these changes make the SSD tier **cache-aware** without altering its
external representation or breaking existing code. When a
`PersistentHybridCache` is present, every SSD move leaves behind a durable,
keyed blob that later subsystems (checkpointing, self-training) can reuse.

## APX 13.8.2 – Gradient Persistent Cache Integration

Version **13.8.2** extends the persistent cache integration to **gradients**.
Where 13.8.1 focused on tensor data backing the SSD tier, this version allows
autograd to save and restore gradient tensors through `PersistentHybridCache`
using a dedicated `CacheKind::Gradient` namespace.

Key properties:

- Gradients use `CacheKind::Gradient` and do **not** collide with tensor keys.
- Cache keys are deterministic and gradient-specific:
  `"grad:<id>:len<bytes>"`.
- Persistence is opt-in via the same `PersistentHybridCache` attachment used
  for tensors.
- Autograd exposes high-level helpers to persist gradients after backward and
  warm them before the next backward, without changing the existing backward
  execution APIs.

### HybridMemoryManager gradient helpers

`HybridMemoryManager` now tracks a minimal index from gradient ids to their
cache keys:

```rust
pub struct HybridMemoryManager {
    tensors: HashMap<String, TensorResidence>,
    cache: SsdCache,
    vram: Box<dyn VramAdapter + Send + Sync>,
    persistent: Option<PersistentHybridCache>,
    grad_cache: HashMap<String, String>,
}
```

This index is populated and queried through two new non-breaking helpers:

```rust
impl HybridMemoryManager {
    pub fn persist_gradient_to_ssd_cache(
        &mut self,
        grad_id: &str,
        bytes: &[u8],
        created_unix: u64,
        overwrite: bool,
    ) -> Result<(), CacheError> {
        let cache = match &self.persistent {
            Some(c) => c,
            None => {
                return Err(CacheError::Io(
                    "Persistent cache not attached to HybridMemoryManager".to_string(),
                ))
            }
        };

        let key = format!("grad:{}:len{}", grad_id, bytes.len());
        cache.put_blob(CacheKind::Gradient, &key, bytes, created_unix, overwrite)?;
        self.grad_cache.insert(grad_id.to_string(), key);
        Ok(())
    }

    pub fn restore_gradient_from_ssd_cache(
        &mut self,
        grad_id: &str,
    ) -> Result<Vec<u8>, CacheError> {
        let cache = match &self.persistent {
            Some(c) => c,
            None => {
                return Err(CacheError::Io(
                    "Persistent cache not attached to HybridMemoryManager".to_string(),
                ))
            }
        };

        let key = match self.grad_cache.get(grad_id) {
            Some(k) => k.clone(),
            None => {
                return Err(CacheError::NotFound);
            }
        };

        cache.get_blob(CacheKind::Gradient, &key)
    }
}
```

Behavior:

- If no persistent cache is attached, both helpers return a `CacheError::Io`
  explaining that the cache is not available.
- `persist_gradient_to_ssd_cache` derives a deterministic key based on the
  gradient id and byte length, writes the blob using `CacheKind::Gradient`,
  and records the mapping in `grad_cache`.
- `restore_gradient_from_ssd_cache` looks up the key in `grad_cache` and
  loads the bytes from the gradient namespace.

These helpers intentionally do not modify tensor tiers or backings; they only
manage the persistent representation of gradients.

### Autograd integration helpers

`autograd.rs` introduces a small report type summarizing persistence results:

```rust
#[derive(Debug, Clone)]
pub struct GradPersistReport {
    pub saved: usize,
    pub skipped: usize,
}
```

and two helper functions to persist and warm gradients around backward:

```rust
pub fn persist_grads_after_backward(
    mem: &mut HybridMemoryManager,
    grad_ids: &[String],
    created_unix: u64,
) -> Result<GradPersistReport, CacheError> { /* ... */ }

pub fn warm_grads_before_backward(
    mem: &mut HybridMemoryManager,
    grad_ids: &[String],
) -> usize { /* ... */ }
```

#### `persist_grads_after_backward`

This helper walks a list of gradient ids and attempts to persist each one
through the `HybridMemoryManager` cache helpers. Its behavior is:

- For each `grad_id`:
  - If the gradient does not exist in memory: increment `skipped`.
  - If the gradient resides in **RAM**:
    - Inspect the backing via `backing_for_test`.
    - If the backing is `StorageBacking::Ram`, clone the bytes and call
      `persist_gradient_to_ssd_cache(grad_id, bytes, created_unix, true)`.
    - Increment `saved` on success.
  - If the gradient resides in **VRAM**:
    - Use `plan_move(grad_id, MemoryTier::Ram, empty_snapshot)` and
      `apply_move` to move the gradient into RAM using the configured
      `VramAdapter`.
    - Once the move succeeds, read the RAM backing and persist it via
      `persist_gradient_to_ssd_cache` as above.
    - Increment `saved` on success; otherwise increment `skipped`.
  - If the gradient resides on **SSD** or **CPU**, increment `skipped` and
    leave it unchanged (no duplicate persistence policy is defined).

It returns a `GradPersistReport` summarizing how many gradients were actually
saved versus skipped due to missing entries, unsupported backing, or move
failures. Any `CacheError` from the underlying helpers is propagated to the
caller.

#### `warm_grads_before_backward`

This helper is a best-effort "warm-up" step that reconstructs missing
gradients from the persistent cache before running backward:

- For each `grad_id`:
  - If `HybridMemoryManager` already has a tier entry for this id, it is left
    untouched.
  - Otherwise, `restore_gradient_from_ssd_cache` is called.
  - On success, the returned bytes are registered as a RAM tensor via
    `register_tensor_with_data(grad_id, bytes, MemoryTier::Ram)`.
  - The internal counter `restored` is incremented when registration
    succeeds.
- Any individual cache errors (missing entry, I/O issues) are ignored, so the
  function never panics and never fails the overall backward.

The function returns the number of gradients successfully restored into RAM.

### Tests: `tests/gradient_persistent_cache_test.rs`

The gradient cache integration is validated with three deterministic tests,
using separate roots under `./.atenia_cache_test_gradients*` and a local
`FakeVramAdapter` implementation.

- **`persist_and_restore_gradient_roundtrip`**

  - Creates a `HybridMemoryManager` with VRAM support and attaches a
    `PersistentHybridCache` rooted at `./.atenia_cache_test_gradients_roundtrip__grad_cache`.
  - Registers gradient `g1` in RAM with bytes `[9,8,7]`.
  - Calls `persist_grads_after_backward(&["g1"], created_unix=1)` and asserts
    `saved == 1`.
  - Verifies that the gradient key `grad:g1:len3` exists in the
    `CacheKind::Gradient` namespace.
  - Removes `g1` from memory using a test-only helper and then calls
    `warm_grads_before_backward(&["g1"])`.
  - Asserts that one gradient is restored, the tier is `MemoryTier::Ram`, and
    the restored RAM backing matches the original bytes.

- **`gradients_use_separate_namespace_from_tensors`**

  - Creates a standalone `PersistentHybridCache` under
    `./.atenia_cache_test_gradients_namespace__grad_cache`, cleaning up the
    directory before each run.
  - Writes two entries with the **same key string** `"k"` but different
    kinds and payloads:
    - `CacheKind::Tensor` with bytes `[1,2,3]`.
    - `CacheKind::Gradient` with bytes `[4,5,6]`.
  - Asserts that both `exists(Tensor, "k")` and `exists(Gradient, "k")` are
    `true`.
  - Calls `get_blob` for each kind and verifies that each returns its own
    payload, proving that tensor and gradient namespaces are independent.

- **`persist_moves_vram_grad_to_ram_before_saving`**

  - Uses a `HybridMemoryManager` with `FakeVramAdapter` and attached
    `PersistentHybridCache` rooted at
    `./.atenia_cache_test_gradients_vram__grad_cache`.
  - Registers gradient `g2` in RAM with bytes `[1,1,1,1]`.
  - Builds an empty `MemorySnapshot` and calls `plan_move` + `apply_move` to
    move `g2` from RAM to VRAM, using the fake adapter for upload.
  - Calls `persist_grads_after_backward(&["g2"], created_unix=2)`.
  - Asserts that `report.saved == 1` and that the gradient key
    `grad:g2:len4` exists in the `Gradient` namespace of the persistent
    cache.

These tests verify that gradients are persisted using a dedicated
`CacheKind::Gradient` namespace, that the helper functions can warm gradients
back into RAM, and that VRAM-resident gradients are moved to RAM before being
saved, all without changing the existing backward execution APIs.

## APX 13.9.0 – Hybrid Checkpointing V1 (Hardware-agnostic)

Version **13.9.0** introduces a first, hardware-agnostic checkpointing
mechanism for the hybrid memory system. It allows saving and restoring:

- The logical tensor/gradient registry.
- The physical tier placement (RAM, SSD, VRAM as logical tiers).
- Cache references for SSD-backed data.

The design is explicitly **portable**:

- The manifest is a plain-text file with `key=value` lines, easy to inspect
  and diff.
- No vendor-specific identifiers or binary metadata are stored in the
  checkpoint; only logical ids, tiers, and cache keys.
- Persistent blobs remain in `PersistentHybridCache`; the manifest never
  duplicates large byte arrays.

### Checkpoint data model

The new `checkpoint` module defines two main structs and an error type:

```rust
pub struct CheckpointEntry {
    pub id: String,
    pub is_grad: bool,
    pub tier: MemoryTier,
    pub cache_kind: Option<CacheKind>,
    pub cache_key: Option<String>,
    pub len_bytes: usize,
}

pub struct HybridCheckpoint {
    pub version: u32,
    pub created_unix: u64,
    pub entries: Vec<CheckpointEntry>,
}

pub enum CheckpointError {
    Io(String),
    InvalidFormat(String),
    MissingBlob(String),
}
```

Semantics:

- `id` is the logical identifier used by `HybridMemoryManager` for tensors and
  gradients.
- `is_grad` distinguishes gradients from regular tensors. It is derived via
  `HybridMemoryManager::is_grad_id`.
- `tier` is the logical `MemoryTier` at the time of checkpoint (Ram, Ssd,
  Vram, or Cpu).
- `cache_kind` and `cache_key` describe the `PersistentHybridCache` entry used
  for backing (Tensor vs Gradient). When no cache reference exists,
  `cache_kind`/`cache_key` are `None`.
- `len_bytes` records the footprint length in bytes, derived from
  `HybridMemoryManager::tensor_len_bytes`.

### Manifest format (`checkpoint.meta`)

Each checkpoint is described by a manifest file located at:

```text
<root>/checkpoint.meta
```

The format is a sequence of `key=value` lines:

```text
version=1
created_unix=...
entry_count=N

id=...
is_grad=0|1
tier=ram|ssd|vram|cpu
len=...
cache_kind=tensor|gradient|kernel_meta|none
cache_key=...|none

id=...
...
```

Rules:

- The header contains `version`, `created_unix`, and `entry_count`, followed
  by a blank line.
- Each entry is a block of key/value lines separated by a blank line.
- `is_grad` is encoded as `0` or `1`.
- `tier` is a lowercase string mapping directly to `MemoryTier`.
- `cache_kind` is `tensor`, `gradient`, `kernel_meta`, or `none`.
- `cache_key` is the exact key string used in `PersistentHybridCache`, or
  `none` when there is no cache reference.
- Ordering is deterministic: entries are sorted by `(is_grad, id)`.

The parser is strict: missing keys, unknown tiers or cache kinds, inconsistent
`cache_kind`/`cache_key` pairs, and mismatched `entry_count` all produce
`CheckpointError::InvalidFormat`.

### Saving a checkpoint

The main entry point for saving is:

```rust
pub fn save_checkpoint(
    root: impl Into<PathBuf>,
    created_unix: u64,
    mem: &HybridMemoryManager,
) -> Result<HybridCheckpoint, CheckpointError>
```

Behavior:

- Ensures the checkpoint root directory exists.
- Builds a list of `CheckpointEntry` values by querying the
  `HybridMemoryManager`:
  - Uses `ids_for_checkpoint()` to enumerate all registered ids.
  - For each id, uses `get_tier` and `tensor_len_bytes` to capture the current
    logical tier and footprint.
  - Derives `is_grad` via `is_grad_id`.
  - If a `PersistentHybridCache` is attached, constructs a deterministic key
    based on id and length:
    - Tensors: `"tensor:<id>:len<bytes>"`.
    - Gradients: `"grad:<id>:len<bytes>"`.
    - If `cache.exists(kind, key)` is true, `cache_kind`/`cache_key` are
      populated; otherwise they are `None`.
- Sorts entries by `(is_grad, id)` and writes `checkpoint.meta` using
  `key=value` lines.
- Returns the in-memory `HybridCheckpoint` for introspection in tests or
  higher-level orchestration.

Important constraints:

- `save_checkpoint` **never** writes blobs itself. It assumes that data is
  already present in `PersistentHybridCache` from prior moves or gradient
  persistence steps.
- If an SSD entry has no cache reference, it is still included with
  `cache_kind=None` and `cache_key=None`, preserving legacy behavior.

### Restoring a checkpoint (hardware-agnostic)

Restoring is handled by:

```rust
pub fn restore_checkpoint(
    root: impl Into<PathBuf>,
    mem: &mut HybridMemoryManager,
) -> Result<HybridCheckpoint, CheckpointError>
```

Behavior:

- Reads and parses `checkpoint.meta` using the strict parser.
- Requires that `HybridMemoryManager` has an attached `PersistentHybridCache`;
  otherwise returns `CheckpointError::Io`.
- For each `CheckpointEntry`:
  - **RAM entries**:
    - Require a valid `cache_kind` and `cache_key`.
    - Load bytes via `PersistentHybridCache::get_blob` and register the tensor
      in RAM using `register_tensor_with_data(id, bytes, MemoryTier::Ram)`.
  - **SSD entries**:
    - Require a valid `cache_kind` and `cache_key`.
    - Load bytes from the persistent cache and register the tensor into SSD
      using `register_tensor_with_data(id, bytes, MemoryTier::Ssd)`. This
      ensures that subsequent SSD → RAM moves can validate against the same
      data, while the manifest itself remains vendor-agnostic.
  - **VRAM entries**:
    - Treat VRAM as a **logical** tier only. On restore, always load the
      bytes from the cache and register them in RAM, even if VRAM used to be
      available.
    - This guarantees safe, portable restores on hosts without GPU support.
  - **CPU entries**:
    - Are currently ignored; no restore policy is defined in this version.

Cache-related failures (missing blobs, I/O errors, corrupt metadata) are
mapped into `CheckpointError::MissingBlob` or `CheckpointError::InvalidFormat`
via a small adapter, making it clear whether the manifest or the underlying
storage is at fault.

### Tests: `tests/hybrid_checkpoint_v1_test.rs`

The initial checkpoint implementation is validated by three deterministic
tests that use dedicated cache and checkpoint roots under
`./.atenia_cache_test_checkpoint_cache_*` and
`./.atenia_checkpoint_test_v1_*`.

- **`save_and_restore_checkpoint_ram_entries`**

  - Creates a `HybridMemoryManager` with VRAM support and a
    `PersistentHybridCache` rooted at
    `./.atenia_cache_test_checkpoint_cache_ram`.
  - Registers tensor `t1` in RAM with bytes `[1,2,3]`.
  - Writes the corresponding tensor blob directly into the cache using the
    deterministic key `tensor:t1:len3`.
  - Calls `save_checkpoint`, verifying that a single entry for `t1` is
    recorded.
  - Constructs a fresh `HybridMemoryManager` using the same cache root and
    reattaches the existing cache.
  - Calls `restore_checkpoint` and asserts that:
    - The restored checkpoint has one entry.
    - `t1` exists in RAM.
    - The RAM backing bytes match the original payload.

- **`restore_ssd_entry_as_reference_without_loading`**

  - Uses a `HybridMemoryManager` with an attached persistent cache rooted at
    `./.atenia_cache_test_checkpoint_cache_ssd`.
  - Registers tensor `t2` in RAM with bytes `[5,6,7,8]`.
  - Builds an empty `MemorySnapshot` and uses `plan_move` + `apply_move` to
    move `t2` from RAM to SSD, causing the bytes to be written to the SSD
    cache and recorded in the persistent cache.
  - Calls `save_checkpoint` and verifies that there is a single SSD entry.
  - Restores into a new `HybridMemoryManager` that uses the same cache root
    and a fresh `PersistentHybridCache`.
  - After `restore_checkpoint`, asserts that:
    - The restored checkpoint has one entry.
    - `t2` is logically on SSD.
  - Then moves `t2` from SSD back to RAM via `plan_move` + `apply_move` and
    asserts that the RAM backing bytes match the original payload, proving
    that the restored SSD reference was valid and consistent.

- **`vram_entries_restore_safely_to_ram`**

  - Constructs a `HybridMemoryManager` with `FakeVramAdapter` and an attached
    `PersistentHybridCache` rooted at
    `./.atenia_cache_test_checkpoint_cache_vram`.
  - Registers tensor `t3` in RAM with bytes `[9,9]` and moves it to VRAM via
    `plan_move` + `apply_move`, which uploads the data into the fake VRAM
    storage.
  - Persists the bytes into the cache under `tensor:t3:len2` using
    `put_blob`.
  - Calls `save_checkpoint` and verifies that there is exactly one VRAM entry.
  - Restores into a new `HybridMemoryManager` sharing the same cache root and
    cache instance.
  - After `restore_checkpoint`, asserts that:
    - The restored checkpoint has one entry.
    - `t3` is present in RAM (not VRAM), honoring the hardware-agnostic
      restore policy.
    - The RAM backing bytes match the original payload.

Together, these components provide a first, portable snapshot mechanism for
the hybrid memory system, suitable for experimentation with offline training
loops and cross-device transfers without depending on specific GPU hardware.

## APX 13.9.1 – Checkpoint Hints (Desired Tier + Plan Summary)

Version **13.9.1** extends Hybrid Checkpointing V1 with **placement hints**
that survive across save/restore cycles:

- A per-id *desired tier* hint (`desired_tier`), indicating where the planner
  would like a tensor or gradient to live (Ram, Ssd, Vram, Cpu).
- A compact, human-readable *plan summary* string (`last_plan_summary`) that
  explains why a particular placement was chosen.

These hints are stored both in-memory (inside `HybridMemoryManager`) and in the
checkpoint manifest, so that the next planning tick can reuse prior decisions
even after a full process restart.

### Placement hints in `HybridMemoryManager`

The hybrid memory manager now maintains a small, internal registry of
placement hints keyed by logical id:

- `desired_tier: Option<MemoryTier>`
- `last_plan_summary: Option<String>`

New, non-breaking methods expose this registry:

```rust
pub fn set_desired_tier_hint(&mut self, id: &str, tier: Option<MemoryTier>);
pub fn get_desired_tier_hint(&self, id: &str) -> Option<MemoryTier>;

pub fn set_last_plan_summary(&mut self, id: &str, summary: Option<String>);
pub fn get_last_plan_summary(&self, id: &str) -> Option<String>;
```

These APIs allow higher-level planners to record their placement choices and
the reasoning behind them without changing the physical tier immediately.

### Extended checkpoint entry model

`CheckpointEntry` is extended with two optional fields:

```rust
pub struct CheckpointEntry {
    pub id: String,
    pub is_grad: bool,
    pub tier: MemoryTier,
    pub cache_kind: Option<CacheKind>,
    pub cache_key: Option<String>,
    pub len_bytes: usize,
    pub desired_tier: Option<MemoryTier>,
    pub last_plan_summary: Option<String>,
}
```

- `desired_tier` mirrors the in-memory desired tier hint.
- `last_plan_summary` carries a short English description of the last
  placement decision for debugging and offline analysis.

### Manifest format: `desired_tier` and `plan_summary`

The manifest (`<root>/checkpoint.meta`) is extended in a
backward-compatible way. Each entry block now includes two additional
`key=value` lines:

```text
id=...
is_grad=0|1
tier=ram|ssd|vram|cpu
len=...
cache_kind=tensor|gradient|kernel_meta|none
cache_key=...|none
desired_tier=ram|ssd|vram|cpu|none
plan_summary=<one line>|none
```

Rules and guarantees:

- If a field is missing (old manifests), it is treated as `none`.
- `desired_tier=none` means "no explicit preference".
- `plan_summary=none` means "no recorded explanation".
- `plan_summary` is always a **single line**: any newlines or carriage
  returns are replaced by spaces when writing the manifest.
- Entry ordering and all previously documented strictness rules remain
  unchanged.

### Save/restore semantics for hints

On **save** (`save_checkpoint`):

- The checkpoint builder queries `HybridMemoryManager` for each id:
  - `desired_tier = mem.get_desired_tier_hint(id)`.
  - `last_plan_summary = mem.get_last_plan_summary(id)`.
- These values are written into the corresponding `CheckpointEntry` and
  emitted to `checkpoint.meta` as `desired_tier` and `plan_summary`.

On **restore** (`restore_checkpoint`):

- The manifest parser reads `desired_tier` and `plan_summary` per entry.
- After materializing the tensor (RAM/SSD/VRAM→RAM/CPU), the function applies
  the hints back into the manager:

  ```rust
  mem.set_desired_tier_hint(&entry.id, desired_tier);
  mem.set_last_plan_summary(&entry.id, last_plan_summary.clone());
  ```

- This ensures that the next planning tick sees the same desired tier and
  explanation that existed when the checkpoint was created.

### VRAM hints under hardware-agnostic restore

Hybrid Checkpointing V1 remains hardware-agnostic: VRAM entries are **always
restored as RAM** to guarantee safe behavior on hosts without GPUs.

APX 13.9.1 refines this by preserving the **original desired tier hint** even
when the physical restore target is downgraded:

- A tensor saved with `tier=vram` and `desired_tier=vram` will be restored as
  `tier=ram`, but its `desired_tier` hint stays as `Vram`.
- This allows future planners to attempt re-promotion to VRAM when suitable
  hardware becomes available, without encoding any vendor-specific details in
  the checkpoint.

### Tests: `tests/hybrid_checkpoint_hints_test.rs`

The hint extension is validated by three focused tests:

- **`save_and_restore_preserves_hints`**

  - Registers a RAM tensor, writes its blob into the persistent cache, and
    sets both `desired_tier=Vram` and a non-empty `last_plan_summary`.
  - Calls `save_checkpoint` and then `restore_checkpoint` into a fresh
    `HybridMemoryManager` using the same cache.
  - Asserts that the restored checkpoint entry carries a `plan_summary` string
    containing the original explanation.

- **`restore_old_manifest_without_hints_is_ok`**

  - Crafts a minimal legacy manifest without `desired_tier` or
    `plan_summary` lines.
  - Restores it into `HybridMemoryManager` with an attached persistent cache.
  - Verifies that `get_desired_tier_hint(id)` and `get_last_plan_summary(id)`
    both return `None`, confirming backward compatibility.

- **`vram_hint_survives_safe_restore_to_ram`**

  - Builds a manifest entry that marks a tensor as `tier=vram` with
    `desired_tier=vram` and a valid cache-backed blob.
  - Restores it into a manager without GPU hardware.
  - Confirms that the tensor is materialized in RAM while the
    `desired_tier` hint remains `Vram`, preserving the planner’s intent
    across hardware-agnostic restores.

Together, APX 13.9.0 and 13.9.1 provide a portable checkpoint format that not
only captures the **state** of the hybrid memory system (tiers, cache
references) but also its **intent** (desired tier and placement rationale),
without tying the manifest to any specific vendor or device model.

## APX 13.9.2 – Checkpoint Drift Detection (Observability Only)

Version **13.9.2** adds a lightweight **drift detection model** on top of
Hybrid Checkpointing V1 and the hint mechanism introduced in 13.9.1. This
subversion is strictly observational:

- It does **not** change the behavior or return types of `restore_checkpoint`.
- It does **not** modify physical tier placement or hints.
- It only records and exposes structured drift reports for diagnostics and
  tests.

### Drift model and reports

The checkpoint module defines an internal drift model in
`src/v13/checkpoint/drift.rs` and exposes it as a nested module under
`v13::checkpoint`:

```rust
pub enum CheckpointDrift {
    MissingBackend {
        desired: MemoryTier,
    },
    TierDowngrade {
        desired: MemoryTier,
        restored: MemoryTier,
    },
    PlanMismatch {
        summary: String,
    },
}

pub struct DriftReport {
    pub entry_id: TensorId,
    pub drifts: Vec<CheckpointDrift>,
}
```

Semantics:

- **`MissingBackend`** – the desired tier requires a backend (e.g. VRAM/GPU)
  that is not available on the restore host.
- **`TierDowngrade`** – the tensor or gradient was restored on a lower tier
  than its `desired_tier` (for example, desired `Vram` vs restored `Ram`).
- **`PlanMismatch`** – the last plan summary mentions GPU/CPU/SSD keywords
  that no longer match the hardware conditions at restore time.

Reports are stored in a small global collector:

```rust
pub struct DriftReport {
    pub entry_id: TensorId,
    pub drifts: Vec<CheckpointDrift>,
}

pub(crate) fn push_report(report: DriftReport) { ... }
pub(crate) fn clear_reports() { ... }
pub fn take_all_for_test() -> Vec<DriftReport> { ... }
```

- The collector is backed by `OnceLock<Mutex<Vec<DriftReport>>>`.
- `push_report` and `clear_reports` are internal helpers used by
  `restore_checkpoint`.
- `take_all_for_test` is a small utility used only by tests to inspect and
  reset the collected drifts.

### Drift analysis during restore

Drift detection is hooked into `restore_checkpoint` **after** each entry has
been materialized, and **before** hints are re-applied to the
`HybridMemoryManager`:

```rust
drift::clear_reports();

// Hybrid checkpointing v1 is hardware-agnostic and safe on CPU-only hosts.
// Drift detection assumes `gpu_available = false` and only observes.
let gpu_available = false;

for entry in &checkpoint.entries {
    let desired_tier = entry.desired_tier;
    let last_plan_summary = entry.last_plan_summary.clone();
    let mut actual_tier = entry.tier;

    // ... existing restore logic sets `actual_tier` to Ram/Ssd/Ram-for-Vram ...

    let mut drifts: Vec<drift::CheckpointDrift> = Vec::new();

    if let Some(desired) = desired_tier {
        if !gpu_available && matches!(desired, MemoryTier::Vram) {
            drifts.push(drift::CheckpointDrift::MissingBackend { desired });
        }

        if matches!(desired, MemoryTier::Vram) && matches!(actual_tier, MemoryTier::Ram) {
            drifts.push(drift::CheckpointDrift::TierDowngrade {
                desired,
                restored: actual_tier,
            });
        }
    }

    if let Some(ref summary) = last_plan_summary {
        let summary_upper = summary.to_uppercase();
        if !gpu_available && summary_upper.contains("GPU") {
            drifts.push(drift::CheckpointDrift::PlanMismatch {
                summary: summary.clone(),
            });
        }
    }

    if !drifts.is_empty() {
        let report = drift::DriftReport {
            entry_id: TensorId(entry.id.clone()),
            drifts,
        };

        drift::push_report(report);
    }

    mem.set_desired_tier_hint(&entry.id, desired_tier);
    mem.set_last_plan_summary(&entry.id, last_plan_summary.clone());
}
```

Key points:

- `restore_checkpoint` still returns only `Result<HybridCheckpoint, CheckpointError>`.
- No early returns or different error cases are introduced by drift detection.
- The physical tier chosen for restore (Ram/Ssd/Ram-for-Vram/Cpu) is exactly
  the same as in 13.9.0/13.9.1.
- Drift detection uses **simple keyword matching** on `last_plan_summary`
  ("GPU"), as requested, and does not try to infer complex semantics.

### Tests: `tests/checkpoint_drift_detection_test.rs`

Three focused tests validate that drift is detected and reported without
changing restore behavior:

- **`checkpoint_warns_on_tier_downgrade`**

  - Crafts a manifest entry with `tier=vram`, `desired_tier=vram` and a valid
    tensor blob.
  - Restores on a CPU-only `HybridMemoryManager`.
  - Verifies that `restore_checkpoint` succeeds and that
    `take_all_for_test()` returns exactly one `DriftReport` containing a
    `TierDowngrade { desired: Vram, restored: Ram }` entry.

- **`checkpoint_detects_missing_backend`**

  - Similar to the previous case but focused on backend availability:
    `desired_tier=vram` on a host where `gpu_available` is treated as false.
  - Asserts that the collected drift report contains a
    `MissingBackend { desired: Vram }` entry.

- **`checkpoint_plan_summary_mismatch`**

  - Constructs a manifest with `tier=ram`, `desired_tier=ram`, and a
    `plan_summary` string mentioning `GPU`.
  - Restores on a CPU-only manager and confirms that the resulting
    `DriftReport` contains a `PlanMismatch { summary }` where the summary
    still includes the `GPU` keyword.

Collectively, APX 13.9.2 turns Hybrid Checkpointing V1 + hints into an
observable system: it can report when the **intent** of a placement plan (VRAM
preference, GPU-centric summaries) diverges from the **actual** restore
conditions, without altering execution or introducing new failure modes.

## APX 13.9.3 – Lazy Restore (Opt-in)

Version **13.9.3** introduces an **opt-in lazy restore path** for
checkpoints. Instead of eagerly loading bytes for every entry during
`restore_checkpoint`, this subversion adds a separate, internal API that:

- Restores checkpoint entries **logically** (id, tier, length, hints, drift)
  without loading their data.
- Defers materialization of backing bytes to the first real access.
- Preserves all semantics of the eager restore path once materialization has
  occurred.

The existing `save_checkpoint` and `restore_checkpoint` APIs remain
unchanged. Lazy restore is wired as an additional module and is only used when
explicitly called.

### Lazy entry model

The lazy module (`src/v13/checkpoint/lazy.rs`) defines a small internal model
to track whether an entry has been materialized and where its data comes from:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyState {
    Unmaterialized,
    Materialized,
}

#[derive(Debug, Clone)]
pub enum LazySource {
    PersistentCache { kind: CacheKind, key: String },
    SsdPath { path: String },
    RamSnapshot { bytes: Vec<u8> },
}

#[derive(Debug, Clone)]
pub struct LazyBacking {
    pub state: LazyState,
    pub tier: MemoryTier,
    pub length: usize,
    pub source: LazySource,
}
```

This version uses `PersistentCache`-backed entries; `SsdPath` and
`RamSnapshot` are placeholders for future extensions.

Lazy backings are stored in a process-global registry keyed by `TensorId`:

```rust
static LAZY_REGISTRY: OnceLock<Mutex<HashMap<TensorId, LazyBacking>>> = OnceLock::new();
```

Internal helpers insert and query this registry, and tests use
`state_for_test(id)` / `clear_for_test()` to assert behavior.

### Lazy restore entry point

Lazy restore is implemented as a separate function under
`v13::checkpoint::lazy`:

```rust
pub fn restore_checkpoint_lazy(
    root: impl Into<PathBuf>,
    mem: &mut HybridMemoryManager,
) -> Result<HybridCheckpoint, CheckpointError>
```

Behavior per entry:

- Reads the manifest using the same `read_manifest` helper as the eager path.
- Computes an **effective tier**:
  - `Vram` entries are mapped to `Ram` for safety, as in 13.9.0.
- Registers a logical tensor in the manager:

  ```rust
  mem.register_tensor(&entry.id, entry.len_bytes as u64, effective_tier);
  ```

- If a persistent cache reference exists (`cache_kind` + `cache_key`), inserts
  a `LazyBacking` with `state = Unmaterialized` and a
  `LazySource::PersistentCache { kind, key }` record into `LAZY_REGISTRY`.
- Runs the same **drift detection** logic as the eager restore (13.9.2), but
  without loading bytes.
- Applies `desired_tier` and `last_plan_summary` hints back into the
  `HybridMemoryManager` just like `restore_checkpoint`.

The existing `restore_checkpoint` function continues to perform eager
materialization and is not modified by this subversion.

### On-demand materialization

On-demand loading is driven by a single helper:

```rust
pub fn ensure_materialized(
    mem: &mut HybridMemoryManager,
    id: &str,
) -> Result<(), CheckpointError>
```

Semantics:

- If `id` is **not** present in `LAZY_REGISTRY`, the function returns
  `Ok(())` immediately, assuming the tensor is already managed via the eager
  path.
- If the entry exists and `state == Materialized`, it is a no-op.
- For `LazySource::PersistentCache { kind, key }`:
  - Fetches the persistent cache from the manager.
  - Calls `get_blob(kind, &key)` to load the bytes.
  - Invokes `register_tensor_with_data(id, bytes, backing.tier)` to attach a
    real backing in RAM/SSD according to the stored tier.
  - Updates the registry to `state = Materialized`.
- Other sources (`SsdPath`, `RamSnapshot`) are defined but unused in this
  subversion.

Any I/O or cache-related failure is mapped into `CheckpointError` in the same
style as the eager path, with no panics or unwraps.

### Transparent semantics

From the perspective of the rest of the engine, a tensor restored via:

- `restore_checkpoint` (eager), or
- `restore_checkpoint_lazy` followed by `ensure_materialized`,

ends up with:

- The same effective tier (VRAM entries restored safely as RAM, SSD/RAM
  preserved),
- The same hint state (`desired_tier`, `last_plan_summary`),
- The same byte contents in RAM/SSD backing.

Lazy restore is thus **opt-in and observational**: it changes when bytes are
loaded, not how they are interpreted or used once materialized.

### Tests: `tests/lazy_restore_test.rs`

The lazy restore behavior is validated by three tests:

- **`lazy_restore_does_not_load_unused_entries`**

  - Creates a manifest + cache-backed tensor `t_lazy`.
  - Calls `restore_checkpoint_lazy` without ever touching the data.
  - Asserts that:
    - `mem.get_tier("t_lazy") == Some(MemoryTier::Ram)`.
    - `lazy::state_for_test("t_lazy") == Some(LazyState::Unmaterialized)`.

- **`on_demand_materialization_is_correct`**

  - Similar setup for `t_lazy_mat`.
  - After lazy restore, calls `lazy::ensure_materialized(&mut mem, id)`.
  - Verifies that:
    - The lazy state transitions to `Materialized`.
    - `backing_for_test(id)` shows a `StorageBacking::Ram` with the original
      bytes.

- **`lazy_restore_preserves_hints_and_drift`**

  - Uses a manifest with `tier=vram`, `desired_tier=vram`, and a plan summary
    mentioning `GPU`.
  - Calls `restore_checkpoint_lazy` and checks that:
    - `get_desired_tier_hint(id) == Some(Vram)`.
    - `get_last_plan_summary(id)` contains the `GPU` keyword.
    - `drift::take_all_for_test()` returns a single `DriftReport` containing
      `MissingBackend`, `TierDowngrade`, and `PlanMismatch` entries.
  - Then calls `ensure_materialized` and confirms that hints and plan summary
    remain intact after materialization.

Together with 13.9.1 and 13.9.2, this subversion completes a small stack of
checkpointing features that:

- Persist and restore placement hints and plan summaries.
- Detect and report drift between intent and actual restore conditions.
- Allow deferring I/O and backing materialization until the data is actually
  used, while keeping behavior fully transparent to the rest of the engine.

## APX 13.9.4 – Warm-Start Placement Planner

Version **13.9.4** adds a **warm-start placement planner** that runs after
checkpoint restore. It consumes:

- The restored `HybridCheckpoint` entries.
- Placement hints (`desired_tier`, `last_plan_summary`) restored into
  `HybridMemoryManager`.
- Drift reports produced by 13.9.2.

The goal is to build a **deterministic, vendor-agnostic plan** describing how
the system *would like* to adjust placements on the next planning tick, without
moving any data or materializing lazy entries.

### Warm-start plan structures

The warm-start planner reuses three public structs defined in
`src/v13/checkpoint.rs`:

```rust
#[derive(Debug, Clone)]
pub enum WarmStartAction {
    Keep,
    HintPromote { to: MemoryTier },
    DegradeSafe { to: MemoryTier },
}

#[derive(Debug, Clone)]
pub struct WarmStartDecision {
    pub id: String,
    pub is_grad: bool,
    pub current: MemoryTier,
    pub desired: Option<MemoryTier>,
    pub action: WarmStartAction,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct WarmStartPlan {
    pub decisions: Vec<WarmStartDecision>,
    pub summary: String,
}
```

Semantics:

- **`Keep`** – no change is planned; current tier is acceptable.
- **`HintPromote`** – the planner suggests a future promotion to a higher
  tier (e.g. Ssd→Ram, Ram→Vram) when safe.
- **`DegradeSafe`** – the planner recommends a safe downgrade due to drift or
  missing backends (e.g. Vram→Ram when no GPU is available).

### Planner API

The main planner entry point is implemented in `src/v13/warm_start.rs`:

```rust
pub fn build_warm_start_plan(
    mem: &HybridMemoryManager,
    checkpoint: &HybridCheckpoint,
    drift: &[DriftReport],
    gpu_available: bool,
) -> WarmStartPlan
```

Key properties:

- **Deterministic ordering** – decisions are sorted by `(is_grad, id)`.
- **No side effects** – no data movement is performed; no lazy entries are
  materialized.
- **Vendor-agnostic** – relies only on `MemoryTier`, hints, and drift; never
  inspects vendor-specific identifiers.

Decision rules:

- If a `DriftReport` for the id contains a
  `CheckpointDrift::TierDowngrade { desired, restored }`, the planner chooses:

  ```rust
  action = DegradeSafe { to: restored };
  reason = format!(
      "Tier downgraded from {} to {} during restore drift",
      tier_to_str(desired),
      tier_to_str(restored),
  );
  ```

- Else, if `desired_tier == Some(Vram)`:
  - When `gpu_available == true`:

    ```rust
    action = HintPromote { to: Vram };
    reason = "Desired VRAM and GPU available";
    ```

  - When `gpu_available == false`:

    ```rust
    action = DegradeSafe { to: Ram };
    reason = "Desired VRAM but GPU unavailable";
    ```

- Else, if `desired_tier == Some(Ram)` and the current tier is `Ssd`:

  ```rust
  action = HintPromote { to: Ram };
  reason = "Prefer RAM for faster access when safe";
  ```

- Else, if `desired_tier` is `Some(Cpu)` or `Some(Ssd)`:

  ```rust
  action = Keep;
  reason = "Desired tier is <X> and current placement is <Y>, no change planned";
  ```

- Else, when there is **no desired tier hint**:

  ```rust
  action = Keep;
  reason = "No desired tier hint";
  ```

The planner also produces a compact, single-line summary:

```text
"warm_start: keep=X promote=Y degrade=Z"
```

where `X`, `Y`, `Z` count how many decisions fall into each action kind.

### Applying warm-start reasons to memory

An optional helper updates the in-memory hints and plan summaries without
modifying physical tiers:

```rust
pub fn apply_warm_start_plan_summaries(
    mem: &mut HybridMemoryManager,
    plan: &WarmStartPlan,
)
```

For each `WarmStartDecision` it:

- Sets `last_plan_summary` to the decision’s `reason`:

  ```rust
  mem.set_last_plan_summary(&d.id, Some(d.reason.clone()));
  ```

- Refreshes the desired tier hint, if any:

  ```rust
  mem.set_desired_tier_hint(&d.id, d.desired);
  ```

This does **not** change current tiers, does **not** materialize lazy
entries, and only updates placement metadata for future planning.

### Tests: `tests/warm_start_planner_test.rs`

Four tests validate the planner:

- **`warm_start_prefers_vram_when_gpu_available`**

  - Checkpoint with `t1` having `tier=Ram`, `desired_tier=Vram`.
  - `gpu_available = true`.
  - Verifies that the decision is `HintPromote { to: Vram }` with a reason
    mentioning both "Desired VRAM" and "GPU available", and that the
    summary is `warm_start: keep=0 promote=1 degrade=0`.

- **`warm_start_degrades_when_gpu_missing`**

  - Same setup but `gpu_available = false`.
  - Ensures the decision is `DegradeSafe { to: Ram }` and the reason contains
    "GPU unavailable".

- **`warm_start_respects_drift_downgrade_reason`**

  - Injects a `DriftReport` for `t3` with
    `CheckpointDrift::TierDowngrade { desired=Vram, restored=Ram }`.
  - Confirms that the planner chooses `DegradeSafe { to: Ram }` with a reason
    mentioning a downgrade or drift.

- **`warm_start_does_not_materialize_lazy_entries`**

  - Uses a lazily restored SSD entry `t_lazy` (via
    `restore_checkpoint_lazy`).
  - Builds a warm-start plan using the logical checkpoint and drift, without
    calling any materialization helpers.
  - Asserts that `lazy::state_for_test("t_lazy") == Some(LazyState::Unmaterialized)`,
    proving that planning does not trigger I/O or backing allocation.

Together, APX 13.9.4 turns the restored checkpoint + hints + drift into a
compact, deterministic **warm-start plan** that can guide future placement
decisions without changing current state or forcing materialization of lazy
entries.

## APX 13.10.0 – Hybrid Self-Training Loop V1 (Scaffold)

Version **13.10.0** introduces a minimal, deterministic **self-training
loop** focused on learning backend placement policies (CPU vs GPU) from
execution episodes. This phase does **not** update model weights; it only
learns from contextual information, warm-start decisions, drift, and simple
scalar outcomes.

All code lives under `src/v13/` and uses only in-memory data structures (no
I/O, no new crates).

### Data model for episodes

The self-trainer works with a small set of value types defined in
`src/v13/self_trainer.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendChoice {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutionContext {
    pub gpu_available: bool,
    pub vram_pressure: f32,
    pub ram_pressure: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct DecisionSummary {
    pub backend: BackendChoice,
    pub promote_count: usize,
    pub degrade_count: usize,
    pub keep_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EpisodeOutcome {
    pub success: bool,
    pub score: i32,
    pub had_drift: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingEpisode {
    pub ctx: ExecutionContext,
    pub decision: DecisionSummary,
    pub outcome: EpisodeOutcome,
}
```

This allows the trainer to record structured episodes consisting of:

- A coarse execution context (GPU availability and RAM/VRAM pressures).
- A summary of the warm-start plan (backend choice and counts of
  keep/promote/degrade decisions).
- A simple outcome: success flag, scalar score, and whether drift occurred.

### Binned context and stats table

To avoid storing high-dimensional continuous context, the trainer bins
pressures into a small **ContextBucket**:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextBucket {
    pub gpu_available: bool,
    pub vram_band: u8,
    pub ram_band: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct ChoiceStats {
    pub count: u32,
    pub success_count: u32,
    pub score_sum: i64,
    pub drift_count: u32,
}
```

Banding rule:

- `band = 0` if `pressure < 0.50`
- `band = 1` if `pressure < 0.75`
- `band = 2` if `pressure < 0.90`
- `band = 3` otherwise

The main learning state is a hash table:

```rust
pub struct SelfTrainer {
    table: HashMap<(ContextBucket, BackendChoice), ChoiceStats>,
}
```

For each recorded episode, the trainer updates:

- `count += 1`
- `success_count += 1` when `outcome.success == true`
- `score_sum += outcome.score`
- `drift_count += 1` when `outcome.had_drift == true`

### Trainer API

The public API on `SelfTrainer` is:

```rust
impl SelfTrainer {
    pub fn new() -> Self;

    pub fn record_episode(&mut self, ep: TrainingEpisode);

    pub fn recommend_backend(&self, ctx: ExecutionContext) -> BackendChoice;

    pub fn stats_for(
        &self,
        ctx: ExecutionContext,
        backend: BackendChoice,
    ) -> Option<ChoiceStats>;
}
```

Recommendation policy (deterministic):

- If `gpu_available == false` → always return `BackendChoice::Cpu`.
- Otherwise, compute a **value** per backend for the corresponding bucket:

  ```rust
  value = avg_score - (drift_rate * drift_penalty)
  ```

  where:

  - `avg_score = score_sum / count` (integer division).
  - `drift_rate = drift_count / count`.
  - `drift_penalty` is a fixed constant (5).

- If `gpu_value > cpu_value` → choose `BackendChoice::Gpu`.
- If tie or CPU value is higher → choose `BackendChoice::Cpu` (stability
  preference).

When there is no data for a `(bucket, backend)` combination, the trainer uses
`value = 0`, making the default behavior lean towards CPU when there is no
clear signal.

### Summarizing warm-start plans

To connect the warm-start planner (13.9.4) with the self-trainer, this version
adds a helper that converts a `WarmStartPlan` into a `DecisionSummary`:

```rust
pub fn summarize_warm_start_plan(plan: &WarmStartPlan) -> DecisionSummary
```

Behavior:

- Counts `keep`, `promote`, and `degrade` actions across all decisions in the
  plan, based on `WarmStartAction`.
- Infers a coarse `BackendChoice` using a simple heuristic:
  - If any decision reason (case-insensitive) contains the substring
    "GPU" → `BackendChoice::Gpu`.
  - Otherwise → `BackendChoice::Cpu`.

This keeps the self-trainer logic generic and loosely coupled to the details
of the warm-start planner.

### Tests: `tests/hybrid_self_trainer_v1_test.rs`

Four tests validate the basic learning behavior:

- **`cpu_when_gpu_unavailable`**

  - Creates a context with `gpu_available = false`.
  - Calls `recommend_backend` on a fresh `SelfTrainer`.
  - Asserts that the result is `BackendChoice::Cpu`, independent of any
    hypothetical GPU history.

- **`learns_gpu_when_successful_and_low_drift`**

  - Uses a context with `gpu_available = true` and low pressures.
  - Records multiple GPU episodes with high scores and no drift using
    `record_episode` and a `DecisionSummary` derived from a warm-start plan
    that prefers GPU.
  - Records a smaller number of CPU episodes with lower scores.
  - Asserts that `recommend_backend` returns `BackendChoice::Gpu`.

- **`penalizes_gpu_when_drift_frequent`**

  - Uses the same kind of context as the previous test.
  - Records GPU episodes with moderate scores but `had_drift = true`.
  - Records CPU episodes with slightly lower scores but `had_drift = false`.
  - Confirms that drift penalty makes CPU preferable:
    `recommend_backend(ctx) == BackendChoice::Cpu`.

- **`stats_update_is_correct`**

  - Records a single GPU episode (`success = true`, `score = 7`,
    `had_drift = true`).
  - Calls `stats_for(ctx, BackendChoice::Gpu)` and checks that:
    - `count == 1`
    - `success_count == 1`
    - `score_sum == 7`
    - `drift_count == 1`

Together, APX 13.10.0 provides a minimal but complete scaffold for learning
backend placement preferences from execution episodes, integrating smoothly
with the warm-start planner while remaining deterministic, vendor-agnostic,
and free of side effects on lazy materialization or model weights.

## APX 13.10.1 – Self-Trainer Integration (WarmStart + Drift + Context)

Version **13.10.1** wires the self-training scaffold from 13.10.0 into the
rest of the v13 stack. It introduces a small integration adapter that:

- Builds `ExecutionContext` values from external pressure signals.
- Summarizes `WarmStartPlan` into `DecisionSummary` (reusing 13.9.4/13.10.0
  types).
- Derives a drift flag from real `DriftReport` values.
- Records `TrainingEpisode` instances into `SelfTrainer`.
- Exposes a simple, deterministic entrypoint for "record + recommend" loops.

All integration code lives in `src/v13/self_trainer_integration.rs` and does
not require real GPU execution.

### Context construction

The adapter provides a safe builder for `ExecutionContext`:

```rust
pub fn context_from_pressures(
    gpu_available: bool,
    vram_pressure: f32,
    ram_pressure: f32,
) -> ExecutionContext
```

This helper clamps the pressure values into the `[0.0, 1.0]` range:

- Values below `0.0` are treated as `0.0`.
- Values above `1.0` are treated as `1.0`.
- `NaN` is treated as `0.0`.

This ensures the downstream banding logic used by `SelfTrainer` remains
well-defined and guards against invalid or out-of-range telemetry.

### Drift flag helper

Using the real `DriftReport` type from 13.9.2, the adapter defines a small
utility:

```rust
pub fn had_drift(drifts: &[DriftReport]) -> bool {
    !drifts.is_empty()
}
```

This is used to propagate a single `had_drift` flag into
`EpisodeOutcome::had_drift` when recording episodes.

### Execution result mapping

For integration, a lightweight execution result enum is introduced:

```rust
#[derive(Debug, Clone, Copy)]
pub enum ExecResult {
    Ok { score: i32 },
    Err { score: i32 },
}

pub fn outcome_from_exec_result(res: ExecResult, drift: bool) -> EpisodeOutcome
```

Mapping rules:

- `ExecResult::Ok { score }` → `EpisodeOutcome { success: true, score, had_drift: drift }`.
- `ExecResult::Err { score }` → `EpisodeOutcome { success: false, score, had_drift: drift }`.

This keeps the integration layer simple and testable while allowing callers to
plug in synthetic or real execution scores.

### Main integration entrypoints

Two helpers provide the main integration API between warm-start planning,
drift detection, and self-training:

```rust
pub fn record_from_warm_start(
    trainer: &mut SelfTrainer,
    ctx: ExecutionContext,
    plan: &WarmStartPlan,
    drifts: &[DriftReport],
    res: ExecResult,
)
```

Behavior:

- Uses `summarize_warm_start_plan(plan)` (from 13.10.0) to obtain a
  `DecisionSummary` that encodes backend choice and keep/promote/degrade
  counts.
- Computes `had_drift(drifts)` from the list of real `DriftReport` entries.
- Builds an `EpisodeOutcome` via `outcome_from_exec_result(res, had_drift)`.
- Records a full `TrainingEpisode { ctx, decision, outcome }` into the
  provided `SelfTrainer` using `record_episode`.

For recommending the next backend choice, the adapter simply delegates to the
trainer:

```rust
pub fn recommend_for_next_tick(
    trainer: &SelfTrainer,
    ctx: ExecutionContext,
) -> BackendChoice {
    trainer.recommend_backend(ctx)
}
```

This preserves the deterministic policy selection logic defined in APX
13.10.0.

### Tests: `tests/hybrid_self_trainer_integration_test.rs`

Integration is validated by four tests that use real `WarmStartPlan` and
`DriftReport` types (no mocks for these structures):

- **`records_episode_from_warm_start_no_drift`**

  - Builds a GPU-leaning `WarmStartPlan` with a single
    `HintPromote { to: Vram }` decision and a reason mentioning GPU.
  - Uses `context_from_pressures(true, 0.1, 0.2)` and an empty drift list.
  - Calls `record_from_warm_start` with `ExecResult::Ok { score: 10 }`.
  - Asserts that `stats_for(ctx, BackendChoice::Gpu)` exists with
    `count == 1`, `success_count == 1`, and `drift_count == 0`.

- **`records_episode_with_drift_flag`**

  - Same setup as above but supplies a non-empty `Vec<DriftReport>` containing
    a `CheckpointDrift::TierDowngrade` entry.
  - After recording, verifies that `drift_count == 1` while `success_count`
    remains `1`.

- **`recommend_changes_after_recording`**

  - Records several GPU episodes using the GPU-promote plan with
    `Ok { score: 10 }`.
  - Records CPU episodes using a CPU-leaning plan with lower scores
    (`Ok { score: 2 }`).
  - Ensures that `recommend_for_next_tick(&trainer, ctx)` returns
    `BackendChoice::Gpu`, demonstrating that the integration path correctly
    populates the self-trainer’s statistics.

- **`clamp_pressures_is_safe`**

  - Calls `context_from_pressures(true, 2.0, -1.0)`.
  - Asserts that `vram_pressure` and `ram_pressure` are within `[0.0, 1.0]`,
    confirming that out-of-range inputs are safely clamped.

Collectively, APX 13.10.1 turns the self-training scaffold from 13.10.0 into a
usable component wired to real warm-start plans and drift reports, while
remaining deterministic, GPU-agnostic, and free of side effects beyond
updating in-memory training statistics.

## APX 13.10.2 – Automatic Trainer Loop (Tick/Batched Recording)

Version **13.10.2** introduces an **automatic trainer loop** abstraction that
can be called from a future batch/tick loop. It combines:

- The self-trainer from 13.10.0.
- The integration adapter from 13.10.1.

into a single stateful component that:

- Records one training episode per tick.
- Recommends the backend (CPU/GPU) for the next tick.
- Applies a simple, deterministic cooldown mechanism to avoid rapid
  oscillations between backends.

All code lives in `src/v13/auto_trainer_loop.rs` and is independent from any
concrete engine loop implementation.

### Loop configuration and state

The loop is configured and tracked via:

```rust
#[derive(Debug, Clone, Copy)]
pub struct AutoTrainerConfig {
    pub cooldown_ticks: u32,
    pub drift_penalty: i32,
}

pub struct AutoTrainerLoop {
    trainer: SelfTrainer,
    cfg: AutoTrainerConfig,
    tick: u64,
    last_choice: BackendChoice,
    last_switch_tick: u64,
    last_reason: String,
}
```

Semantics:

- `cooldown_ticks` – minimum number of ticks between backend switches.
- `drift_penalty` – reserved for future extensions; currently stored but not
  applied.
- `trainer` – internal `SelfTrainer` instance that accumulates episodes.
- `tick` – monotonically increasing tick counter.
- `last_choice` – last effective backend; defaults to `BackendChoice::Cpu`.
- `last_switch_tick` – tick index when the backend last changed; defaults to
  `0`.
- `last_reason` – last human-readable reason for the choice (single line,
  English), defaults to `"init"`.

### Public API

The loop exposes a small, focused API:

```rust
impl AutoTrainerLoop {
    pub fn new(cfg: AutoTrainerConfig) -> Self;

    pub fn on_tick(
        &mut self,
        ctx: ExecutionContext,
        plan: &WarmStartPlan,
        drifts: &[DriftReport],
        res: ExecResult,
    ) -> BackendChoice;

    pub fn last_debug_reason(&self) -> &str;

    pub fn inner_trainer(&self) -> &SelfTrainer;
}
```

### Tick handling and stabilization

On each call to `on_tick`:

1. **Record episode**

   ```rust
   record_from_warm_start(&mut self.trainer, ctx, plan, drifts, res);
   ```

2. **Compute raw recommendation**

   ```rust
   let raw = recommend_for_next_tick(&self.trainer, ctx);
   ```

3. **Apply stabilization rules**

- If `ctx.gpu_available == false`:
  - Force `BackendChoice::Cpu`.
  - `last_reason = "hold: gpu unavailable"`.
  - If the previous choice was GPU, `last_switch_tick` is updated to the
    current tick.

- If `ctx.gpu_available == true`:
  - When `raw != last_choice`:
    - If a cooldown is configured and the time since the last switch is less
      than `cooldown_ticks`, keep `last_choice` and set
      `last_reason = "hold: cooldown active"`.
    - Otherwise, accept the switch, update `last_switch_tick`, and set a
      reason of the form:

      - `"switch: cpu->gpu due to learned score"`,
      - `"switch: gpu->cpu due to learned score"`, or
      - `"switch: backend changed due to learned score"`.

  - When `raw == last_choice`:
    - Keep `last_choice` and set
      `last_reason = "hold: same backend preferred"`.

4. **Advance tick**

- `tick` is incremented after all decisions are made using a saturating add.

This logic is fully deterministic and does not depend on any random sources or
real GPU execution.

### Tests: `tests/auto_trainer_loop_test.rs`

The auto trainer loop is verified by four tests:

- **`cooldown_prevents_thrashing`**

  - Uses `cooldown_ticks = 3` with `gpu_available = true`.
  - Tick 0: GPU-leaning plan with high score → choice becomes `Gpu`.
  - Ticks 1–2: CPU-leaning plan, but cooldown prevents switching → choice
    remains `Gpu`, and `last_debug_reason()` contains `"cooldown"`.
  - Tick 3: cooldown expires, allowing a switch to `Cpu` when the trainer
    prefers it.

- **`always_cpu_when_gpu_unavailable`**

  - Uses `gpu_available = false`.
  - Provides a GPU-leaning plan for multiple ticks.
  - Asserts that `on_tick` always returns `BackendChoice::Cpu` and that
    `last_debug_reason()` contains `"gpu unavailable"`.

- **`records_episodes_every_tick`**

  - Runs five ticks with `gpu_available = true` and a fixed plan.
  - Queries the inner `SelfTrainer` via `inner_trainer()`.
  - Confirms that the sum of episode counts across CPU and GPU backends for
    the given context is at least five, showing that each tick results in a
    recorded episode.

- **`deterministic_given_same_inputs`**

  - Runs two independent `AutoTrainerLoop` instances with identical
    configuration, context, plans, drifts, and execution results.
  - Collects the sequence of backend choices for each run and asserts that
    both sequences are identical.

Together with 13.10.0 and 13.10.1, APX 13.10.2 completes a small, fully
deterministic self-training stack that can observe warm-start plans and drift,
learn from outcomes, and produce stable backend recommendations on a
tick-by-tick basis without integrating into the actual batch loop yet.

## APX 13.10.3 – Persistent Self-Training Table (SSD-backed)

Version **13.10.3** adds a simple, versioned **persistence layer** for the
`SelfTrainer` learning table used by APX 13.10.0–13.10.2. The goal is to make
placement-learning state survive process restarts while remaining:

- Deterministic (stable output ordering).
- Tolerant to corruption or partial writes.
- Backward/forward compatible via a text-based, versioned format.

All persistence code lives in `src/v13/self_trainer_persistence.rs` and
operates purely on the `SelfTrainer` internal table.

### File format

The on-disk representation is a UTF-8 text file with a versioned header and
one record per line:

```text
ATENIA_SELFTRAINER_V1
gpu_avail=<0|1>;vram_band=<0..3>;ram_band=<0..3>;backend=<cpu|gpu>;
    count=<u32>;success=<u32>;score_sum=<i64>;drift=<u32>
```

Characteristics:

- **Header line** – must be exactly `ATENIA_SELFTRAINER_V1`. Unknown headers
  cause the loader to return an empty `SelfTrainer` instead of failing.
- **Key/value pairs** – required keys are `gpu_avail`, `vram_band`,
  `ram_band`, `backend`, `count`, `success`, `score_sum`, `drift`.
- **Unknown keys** – ignored on load to allow forward-compatible extensions.
- **Missing/invalid fields** – any parse error or missing required field
  results in skipping that line (best-effort load).

### Persistence API

The persistence module exposes a small error type and two helpers:

```rust
#[derive(Debug, Clone)]
pub enum PersistError {
    Io(String),
    Format(String),
}

pub fn save_trainer_to_path(
    trainer: &SelfTrainer,
    path: &std::path::Path,
) -> Result<(), PersistError>

pub fn load_trainer_from_path(
    path: &std::path::Path,
) -> Result<SelfTrainer, PersistError>
```

Saving behavior:

- Ensures the parent directory of `path` exists via `create_dir_all`.
- Writes to a temporary file `path.with_extension("tmp")` first:
  - Removes any existing temp file for that path (best-effort).
  - Writes the header and all entries using a buffered writer.
  - Flushes the buffer, then atomically renames the temp file to the final
    destination.
- Enumerates entries via `SelfTrainer::all_stats()` and sorts them
  deterministically by `(gpu_available, vram_band, ram_band, backend)`,
  ensuring stable output ordering across runs.

Loading behavior:

- Attempts to open `path`:
  - If the file does not exist or cannot be opened, returns an empty
    `SelfTrainer::new()`.
- Reads the first line as the header:
  - Any I/O error or empty file → returns `SelfTrainer::new()`.
  - Header mismatch → returns `SelfTrainer::new()`.
- For each subsequent non-empty line:
  - Splits on `;` into `key=value` segments.
  - Parses only known keys; unknown keys are ignored.
  - If any required key is missing or fails to parse, the line is skipped.
  - For well-formed lines, constructs a `ContextBucket` and `ChoiceStats` and
    calls `SelfTrainer::set_stats_entry(bucket, backend, stats)`.

As a result, partially corrupted files simply lose the broken records but do
not cause panics or fatal load failures.

### Auto-trainer loop integration

To make persistence easy to use from higher-level components, APX 13.10.3
extends `AutoTrainerLoop` with two optional helpers:

```rust
impl AutoTrainerLoop {
    pub fn save_learning(&self, path: &std::path::Path) -> Result<(), PersistError> {
        save_trainer_to_path(&self.trainer, path)
    }

    pub fn load_learning(&mut self, path: &std::path::Path) -> Result<(), PersistError> {
        match load_trainer_from_path(path) {
            Ok(trainer) => {
                self.trainer = trainer;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}
```

These functions are **opt-in** and do not change the default behavior of the
auto trainer loop; they simply provide a convenient way to persist and reload
the learning state when a caller decides to use them.

### Tests: `tests/self_trainer_persistence_test.rs`

Four tests validate the persistence behavior:

- **`persistence_roundtrip_preserves_recommendations`**

  - Trains a `SelfTrainer` using real warm-start plans and integration helpers
    (`summarize_warm_start_plan`, `outcome_from_exec_result`).
  - Saves the trainer to disk and reloads it via `load_trainer_from_path`.
  - Asserts that `recommend_backend(ctx)` returns the same backend before and
    after the roundtrip.

- **`persistence_roundtrip_preserves_stats`**

  - Records a single GPU episode with known stats (`count`, `success_count`,
    `score_sum`, `drift_count`).
  - Saves and reloads the trainer.
  - Compares stats before and after using `stats_for(ctx, BackendChoice::Gpu)`
    when I/O succeeds; gracefully skips strict comparison if the environment
    prevents file I/O.

- **`corrupted_lines_are_ignored`**

  - Writes a file with the correct header, one invalid line, and one valid
    line.
  - Loads the trainer and inspects stats for the matching context.
  - Confirms that only the valid line contributes to the stats when the file
    can be read.

- **`missing_file_returns_empty_trainer`**

  - Calls `load_trainer_from_path` on a non-existent path.
  - Verifies that `SelfTrainer::new()` is returned and that
    `recommend_backend` behaves deterministically (e.g. `gpu_available = false`
    → `BackendChoice::Cpu`).

Collectively, APX 13.10.3 turns the in-memory self-training table into a
disk-backed, versioned store that can survive process restarts and tolerate
partial or corrupted data without compromising determinism or safety.

## APX 13.10.3.1 – AutoSave Policy (Tick/Time + Dirty Flag)

Version **13.10.3.1** introduces an optional, deterministic **autosave
policy** for the self-training stack. It allows `AutoTrainerLoop` to
periodically persist its learning table without changing any public APIs or
requiring callers to opt in by default.

### Extended configuration

`AutoTrainerConfig` gains three optional fields:

```rust
#[derive(Debug, Clone)]
pub struct AutoTrainerConfig {
    pub cooldown_ticks: u32,
    pub drift_penalty: i32,

    // Optional autosave configuration.
    pub autosave_every_ticks: Option<u32>,
    pub autosave_every_seconds: Option<u64>,
    pub autosave_path: Option<std::path::PathBuf>,
}
```

Semantics:

- When all autosave fields are `None`, autosave is **disabled**.
- `autosave_every_ticks` – save at most once every N ticks when `dirty`.
- `autosave_every_seconds` – save when at least T seconds have elapsed since
  the last successful save.
- `autosave_path` – target file for persistence; autosave is off if this is
  `None`.

### Loop state and dirty tracking

`AutoTrainerLoop` now tracks autosave state:

```rust
pub struct AutoTrainerLoop {
    trainer: SelfTrainer,
    cfg: AutoTrainerConfig,
    tick: u64,
    last_choice: BackendChoice,
    last_switch_tick: u64,
    last_reason: String,
    dirty: bool,
    last_save_tick: u64,
    last_save_instant: std::time::Instant,
}
```

- `dirty` – set to `true` after each recorded episode (`on_tick`), cleared on a
  successful autosave or explicit `load_learning`.
- `last_save_tick` – tick index at which the last successful autosave occurred.
- `last_save_instant` – wall-clock time of the last successful autosave.

### Autosave decision logic

A private helper encapsulates the trigger conditions:

```rust
fn should_autosave(&self) -> bool {
    if !self.dirty {
        return false;
    }
    if self.cfg.autosave_path.is_none() {
        return false;
    }

    if let Some(n) = self.cfg.autosave_every_ticks {
        if n > 0 {
            let since = self.tick.saturating_sub(self.last_save_tick);
            if since >= n as u64 {
                return true;
            }
        }
    }

    if let Some(secs) = self.cfg.autosave_every_seconds {
        if self.last_save_instant.elapsed().as_secs() >= secs {
            return true;
        }
    }

    false
}
```

This ensures autosave is only considered when there are pending changes, a
target path is configured, and either a tick interval or time interval has
elapsed.

### Autosave API

Two small methods implement the autosave policy on top of the persistence
helpers from 13.10.3:

```rust
impl AutoTrainerLoop {
    pub fn try_autosave(&mut self) -> Result<bool, PersistError> {
        if !self.should_autosave() {
            return Ok(false);
        }

        let path = match &self.cfg.autosave_path {
            Some(p) => p,
            None => return Ok(false),
        };

        match self.save_learning(path) {
            Ok(()) => {
                self.dirty = false;
                self.last_save_tick = self.tick;
                self.last_save_instant = std::time::Instant::now();
                self.last_reason = "autosave: wrote learning table".to_string();
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }
}
```

Key properties:

- `Ok(true)` – an autosave was performed.
- `Ok(false)` – conditions were not met; nothing was written.
- `Err(PersistError)` – autosave attempted but failed due to I/O or format
  issues.

The existing `save_learning` / `load_learning` helpers remain unchanged except
that `load_learning` now clears the `dirty` flag when a trainer is
successfully loaded.

### Integration into `on_tick`

`on_tick` remains the single entrypoint for per-tick processing and backend
recommendation. Autosave is invoked at the end of the method:

```rust
// After recording and stabilization logic.
self.last_choice = final_choice;
self.last_reason = reason;
self.tick = current_tick.saturating_add(1);

// Autosave is best-effort and must not affect the backend choice.
if let Err(_e) = self.try_autosave() {
    self.last_reason = "autosave: failed (io)".to_string();
}

final_choice
```

This guarantees that:

- Backend decisions are computed independently of autosave.
- Autosave failures are surfaced only via `last_reason` and returned errors
  from `try_autosave`, never as panics.

### Tests: `tests/self_trainer_autosave_test.rs`

Four tests exercise the autosave behavior:

- **`autosave_not_triggered_when_disabled`**

  - Uses a config where all autosave fields are `None`.
  - Runs several ticks and asserts that no file is created at the test path.

- **`autosave_triggers_by_tick_interval`**

  - Sets `autosave_every_ticks = Some(2)` and a valid `autosave_path`.
  - Verifies that after the second tick, the autosave file exists, while after
    the first tick it does not.

- **`autosave_respects_dirty_flag`**

  - Configures `autosave_every_ticks = Some(1)`.
  - Confirms that the first tick triggers an autosave and that a subsequent
    explicit `try_autosave()` call returns `Ok(false)` when no new episodes
    have been recorded (dirty flag cleared).

- **`autosave_failure_does_not_break_tick`**

  - Points `autosave_path` to an invalid target (e.g. a directory treated as a
    file).
  - Ensures that `on_tick` still returns a valid `BackendChoice` and that
    `last_debug_reason()` contains `"autosave: failed"`, proving that autosave
    errors do not affect core decision logic or cause panics.

Collectively, APX 13.10.3.1 adds a rate-limited, dirty-aware autosave policy on
top of the self-training persistence layer, keeping behavior deterministic,
opt-in, and resilient to I/O failures.

## APX 13.11.0 – Learning Introspection (Read-Only Snapshot)

Version **13.11.0** introduces a **read-only introspection API** for the
self-training subsystem. The goal is to make the learning state of
`SelfTrainer` observable in a deterministic and stable way without exposing
mutable internals or tying the engine to any particular UI, log format, or
external representation.

This snapshot API is designed to be the foundation for:

- Debugging and behavioral tests.
- Future CLI tools (e.g. `atenia stats`).
- UI or JSON export layers built on top of the core engine.

### Snapshot data model

The snapshot lives in `src/v13/learning_snapshot.rs` and consists of a small
set of immutable structs:

```rust
#[derive(Debug, Clone)]
pub struct LearningSnapshot {
    pub entries: Vec<LearningEntrySnapshot>,
    pub summary: LearningSummary,
}

#[derive(Debug, Clone, Copy)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
pub struct LearningEntrySnapshot {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub stats: LearningStatsSnapshot,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningContextSnapshot {
    pub gpu_available: bool,
    pub vram_band: u8,
    pub ram_band: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningStatsSnapshot {
    pub episodes: u32,
    pub successes: u32,
    pub drift_events: u32,
    pub average_score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningSummary {
    pub total_entries: usize,
    pub total_episodes: u64,
    pub gpu_preference_ratio: f32,
}
```

Design constraints:

- All fields are `Copy`/`Clone`-friendly value types.
- No references or pointers into internal structures.
- `BackendKind` abstracts over `BackendChoice` so the snapshot does not leak
  engine-specific details beyond "CPU" vs "GPU".

The module is exported from `v13::mod` as `v13::learning_snapshot`.

### `SelfTrainer::snapshot()`

The core of the API is a new method on `SelfTrainer`:

```rust
impl SelfTrainer {
    pub fn snapshot(&self) -> LearningSnapshot { /* ... */ }
}
```

Implementation details:

- Iterates over the internal learning table via `all_stats()`; for each
  `(ContextBucket, BackendChoice, ChoiceStats)` triplet it builds:

  - A `LearningContextSnapshot` from `gpu_available`, `vram_band`,
    `ram_band`.
  - A `BackendKind` reflecting the currently preferred backend for that
    context.
  - A `LearningStatsSnapshot` where:
    - `episodes` = `count`.
    - `successes` = `success_count`.
    - `drift_events` = `drift_count`.
    - `average_score` = `score_sum / count` when `count > 0`, otherwise `0.0`.

- Pushes each `LearningEntrySnapshot` into a local `Vec`.

#### Deterministic ordering

Before building the summary, entries are sorted deterministically by
context-only fields:

```rust
entries.sort_by(|a, b| {
    let ag = if a.context.gpu_available { 1u8 } else { 0u8 };
    let bg = if b.context.gpu_available { 1u8 } else { 0u8 };
    ag.cmp(&bg)
        .then_with(|| a.context.vram_band.cmp(&b.context.vram_band))
        .then_with(|| a.context.ram_band.cmp(&b.context.ram_band))
});
```

This guarantees a stable ordering across calls and runs for the same learning
state, which is critical for deterministic tests and reproducible output.

#### Summary computation

After sorting, `SelfTrainer::snapshot()` computes a `LearningSummary`:

- `total_entries` – `entries.len()`.
- `total_episodes` – sum of `episodes` across all entries (using saturating
  addition to avoid overflow).
- `gpu_preference_ratio` – number of entries with `BackendKind::Gpu` divided by
  `total_entries` (or `0.0` when there are no entries).

The method then returns a fully materialized `LearningSnapshot` containing both
the list of entries and the aggregate summary.

### Read-only, non-invasive semantics

`snapshot()` is strictly read-only:

- It does not mutate any internal state of `SelfTrainer`.
- It does not mark the trainer as dirty, touch autosave state, or change any
  counters.
- It does not perform I/O or logging; callers are free to serialize or print
  the snapshot in any desired format.

This makes it safe to call from tests, debuggers, or future CLI/UI code at any
point without side effects.

### Tests: `tests/learning_snapshot_test.rs`

Four tests validate the behavior of the introspection API:

- **`snapshot_empty_trainer`**

  - Calls `SelfTrainer::new()` followed by `snapshot()`.
  - Asserts that `entries.len() == 0` and that the summary fields
    (`total_entries`, `total_episodes`, `gpu_preference_ratio`) all reflect the
    empty state (`0` or `0.0`).

- **`snapshot_contains_learned_entry`**

  - Records a single GPU episode with success using real `TrainingEpisode`
    data.
  - Ensures the snapshot contains exactly one entry whose context fields match
    the episode, whose `recommended_backend` is `BackendKind::Gpu`, and whose
    stats counters and `average_score` are consistent with the episode.

- **`snapshot_is_deterministic_ordered`**

  - Inserts multiple contexts in non-sorted order.
  - Takes two snapshots in a row and verifies that both have the same entry
    count and that the sequence of contexts is identical in both snapshots,
    demonstrating deterministic ordering.

- **`summary_fields_are_correct`**

  - Creates two entries, one preferring GPU and one preferring CPU.
  - Verifies that `summary.total_entries` matches `entries.len()`,
    `summary.total_episodes` matches the number of recorded episodes, and
    `gpu_preference_ratio` is `0.5` (one GPU-preferring entry out of two).

Together, these tests ensure that the learning snapshot API is immutable,
deterministic, and accurate, turning the previously opaque self-training table
into a well-structured, inspectable view that higher-level tooling can build
upon.

## APX 13.11.1 – Human-Readable Decision Explanation (Single Context)

Version **13.11.1.0** adds a **human-readable explanation layer** on top of the
learning snapshot. It answers the question:

> "Why did Atenia choose CPU or GPU for this context?"

without changing any underlying decisions, learning behavior, or planning
logic. The explanation API is deterministic, read-only, and free of I/O or
logging side effects.

### Explanation data structure

The explanation model lives in `src/v13/learning_explanation.rs`:

```rust
#[derive(Debug, Clone)]
pub struct DecisionExplanation {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub confidence: f32,
    pub explanation: String,
}
```

Constraints:

- `confidence` is always clamped to `[0.0, 1.0]`.
- `explanation` is stable, human-readable English with no timestamps or
  environment-dependent details.
- The struct is fully owned (no references) and safe to serialize or log by
  higher layers.

### `SelfTrainer::explain_decision`

On top of the snapshot API, `SelfTrainer` exposes:

```rust
impl SelfTrainer {
    pub fn explain_decision(
        &self,
        context: LearningContextSnapshot,
    ) -> Option<DecisionExplanation> { /* ... */ }
}
```

Behavior:

- Returns `None` when no learned data exists for the given context
  (`LearningContextSnapshot`).
- Otherwise, it:
  - Uses `snapshot()` to obtain a read-only view of all entries.
  - Filters entries whose `context` matches the provided snapshot context.
  - Picks the entry with the highest `episodes` count as the representative
    sample for stats.

From the chosen entry it derives:

- `episodes` = number of observations.
- `successes` = successful executions.
- `drift_events` = how often drift was observed.

#### Confidence computation

Confidence reflects empirical reliability:

```rust
let raw_confidence = if episodes == 0 {
    0.0
} else {
    successes as f32 / episodes as f32
};

let confidence = DecisionExplanation::clamp_confidence(raw_confidence);
```

This yields a value in `[0.0, 1.0]`, where `1.0` means all observed episodes
for this context were successful.

#### Backend selection for explanation

Explanations must not invent their own policy; they describe the **actual
decision** the trainer would make. To do this, `explain_decision` derives an
approximate `ExecutionContext` from the snapshot bands and calls
`recommend_backend`:

```rust
fn band_to_pressure(band: u8) -> f32 {
    match band {
        0 => 0.2,
        1 => 0.5,
        2 => 0.8,
        _ => 0.5,
    }
}

let exec_ctx = ExecutionContext {
    gpu_available: context.gpu_available,
    vram_pressure: band_to_pressure(context.vram_band),
    ram_pressure: band_to_pressure(context.ram_band),
};

let backend_choice = self.recommend_backend(exec_ctx);
let recommended_backend = match backend_choice {
    BackendChoice::Cpu => BackendKind::Cpu,
    BackendChoice::Gpu => BackendKind::Gpu,
};
```

This guarantees that the explanation matches the real engine behavior for the
same context bucket.

#### Explanation text

The function then builds a concise, 2–3 sentence explanation string based on
the recommended backend and the observed stats.

For **GPU** decisions:

```text
"GPU was selected because this context has been observed N times, with S
successful executions and low drift/some drift observed. Historical
performance favors GPU execution under similar memory conditions."
```

For **CPU** decisions:

```text
"CPU was selected because GPU execution in this context showed instability or
drift, or historical data does not strongly favor GPU. Historical data
indicates CPU is more reliable under the current memory pressure, based on N
observations."
```

These templates mention:

- Number of observations (episodes).
- Success behavior and drift/stability when applicable.
- A human-understandable reason for preferring CPU or GPU under current
  memory pressure.

### Read-only and deterministic

`explain_decision` is strictly observational:

- Uses `snapshot()` only; it does not mutate the trainer, mark it as dirty, or
  trigger autosave.
- Performs no I/O, logging, or printing.
- Given the same trainer state and context, it always returns the same
  `DecisionExplanation` (including identical `explanation` text).

### Tests: `tests/learning_explanation_test.rs`

Four tests validate the explanation layer:

- **`explanation_none_when_context_unknown`**

  - New trainer with no recorded episodes.
  - Calls `explain_decision` on an arbitrary context and asserts that the
    result is `None`.

- **`explanation_for_gpu_context`**

  - Records a single successful GPU episode.
  - Uses `snapshot()` to obtain the actual learned context and passes it to
    `explain_decision`.
  - Asserts that `recommended_backend` is `BackendKind::Gpu`, `confidence > 0.5`
    and that the explanation string mentions "GPU".

- **`explanation_for_cpu_context_due_to_drift`**

  - Records a drift-heavy GPU episode with low success and a subsequent
    successful CPU episode for the same context.
  - Obtains the learned context via `snapshot()` and explains it.
  - Verifies that `recommended_backend` is `BackendKind::Cpu`, the explanation
    mentions "CPU", and the text references "drift" or "instability".

- **`explanation_is_deterministic`**

  - Records a stable CPU-only episode.
  - Calls `explain_decision` twice for the same context and checks that the
    backend, confidence, and explanation string are identical across calls.

Together, APX 13.11.1.0 turns the previously opaque backend recommendation into
an explainable, testable, and human-readable decision, without changing any
underlying learning or planning behavior.

## APX 13.11.2 – Structured Decision Explanation (Factors + Weights)

Version **13.11.2.0** complements the human-readable explanation layer from
13.11.1 by adding a **structured, factor-based explanation**. This view is
optimized for tooling, UI, and quantitative analysis rather than direct human
consumption.

Where 13.11.1 focuses on narrative text, 13.11.2 exposes the same reasoning in
terms of explicit decision factors and normalized weights suitable for JSON
export, dashboards, or papers.

### Factor model

The structured explanation types live in `src/v13/learning_factors.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFactorKind {
    HistoricalSuccessRate,
    DriftPenalty,
    ObservationCount,
    MemoryStability,
}

#[derive(Debug, Clone)]
pub struct DecisionFactor {
    pub kind: DecisionFactorKind,
    pub weight: f32,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct StructuredDecisionExplanation {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub confidence: f32,
    pub factors: Vec<DecisionFactor>,
}
```

Rules:

- `weight` is always clamped to the `[0.0, 1.0]` interval.
- `description` is deterministic, human-readable English.
- `factors` are always returned in a fixed, documented order.

Internally a small `clamp01` helper ensures that all weights and confidences
honor the `[0.0, 1.0]` invariant.

### `SelfTrainer::explain_decision_structured`

`SelfTrainer` exposes a structured explanation entry point:

```rust
impl SelfTrainer {
    pub fn explain_decision_structured(
        &self,
        context: LearningContextSnapshot,
    ) -> Option<StructuredDecisionExplanation> { /* ... */ }
}
```

Behavior:

- Returns `None` when no learned stats exist for the given context.
- Uses `snapshot()` to obtain a read-only view of learned entries.
- Filters entries whose `context` matches the provided snapshot context.
- Chooses the entry with the largest `episodes` count as the representative
  sample for factor computation.

From the chosen entry it reads:

- `episodes` – total observations for the context.
- `successes` – how many episodes succeeded.
- `drift_events` – how many episodes experienced drift.

### Factor computation

Given `(episodes, successes, drift_events)`, the method computes four factors
with the following weights and descriptions:

1. **HistoricalSuccessRate**

   - Weight: `successes / episodes` (or `0.0` when `episodes == 0`).
   - Description: reports the empirical success ratio, e.g.
     `"Historical success rate for this context is 0.80 (8 successes over 10 episodes)."`.

2. **DriftPenalty**

   - Raw weight: `drift_events / episodes` (or `0.0` when `episodes == 0`).
   - Clamped into `[0.0, 1.0]`.
   - Description:
     - If `drift_events > 0`: mentions how many times drift was observed and
       that this indicates potential instability.
     - Otherwise: notes that no drift has been observed.

3. **ObservationCount**

   - Weight: `episodes / 50.0`, clamped into `[0.0, 1.0]`.
   - Interpretation: how strong the sample size is (approaches 1.0 around
     50+ episodes).
   - Description: states the total number of observations and that more data
     increases confidence.

4. **MemoryStability**

   - Weight: `1.0 - DriftPenalty.weight` (implicitly clamped via
     `DriftPenalty`).
   - Description:
     - If `drift_events == 0`: indicates that executions have been stable under
       current memory pressure.
     - Otherwise: notes that some instability was observed due to drift.

The factors are always returned in the **fixed order** above, making the model
easy to consume from tooling without additional sorting or introspection.

### Confidence and backend consistency

`StructuredDecisionExplanation` also exposes a `confidence` field. It is
computed using the same logic as the textual explanation:

```rust
let raw_confidence = if episodes == 0 {
    0.0
} else {
    successes as f32 / episodes as f32
};

let confidence = StructuredDecisionExplanation::clamp_confidence(raw_confidence);
```

To keep both explanation layers consistent with actual engine behavior,
`explain_decision_structured` derives an approximate `ExecutionContext` from
the snapshot bands and calls `recommend_backend`, exactly as in 13.11.1:

```rust
fn band_to_pressure(band: u8) -> f32 { /* 0.2 / 0.5 / 0.8 / fallback */ }

let exec_ctx = ExecutionContext {
    gpu_available: context.gpu_available,
    vram_pressure: band_to_pressure(context.vram_band),
    ram_pressure: band_to_pressure(context.ram_band),
};

let backend_choice = self.recommend_backend(exec_ctx);
let recommended_backend = match backend_choice {
    BackendChoice::Cpu => BackendKind::Cpu,
    BackendChoice::Gpu => BackendKind::Gpu,
};
```

The `recommended_backend` field in the structured explanation therefore matches
both the human-readable explanation and the actual decision logic.

### Read-only semantics

`explain_decision_structured` is a pure observer:

- It only reads from the snapshot; it does not modify the trainer state or mark
  it as dirty.
- It performs no I/O (no file access, logging, or printing).
- Given the same trainer state and context, it returns the same factor list and
  weights.

### Tests: `tests/learning_factors_test.rs`

Five tests validate the structured explanation behavior:

- **`structured_none_when_context_unknown`**

  - New trainer with no episodes.
  - Calls `explain_decision_structured` on an arbitrary context and asserts
    that it returns `None`.

- **`structured_contains_all_factors`**

  - Records a simple GPU-favoring episode.
  - Uses `snapshot()` to obtain a learned context and explains it.
  - Asserts that exactly four factors are returned in the expected order:
    `HistoricalSuccessRate`, `DriftPenalty`, `ObservationCount`,
    `MemoryStability`.

- **`weights_are_clamped_and_safe`**

  - Records many unstable episodes to drive counts high.
  - Verifies that every factor weight is within `[0.0, 1.0]` regardless of
    episode count or drift frequency.

- **`gpu_structured_explanation_has_high_success_weight`**

  - Records many successful GPU episodes for a context.
  - Confirms that the `HistoricalSuccessRate` factor has a weight greater than
    `0.6`, reflecting strong evidence in favor of GPU.

- **`drift_penalty_present_for_unstable_context`**

  - Records multiple episodes with drift for a context.
  - Checks that the `DriftPenalty` factor weight is strictly greater than `0.0`
    and that its description mentions instability or drift.

Collectively, APX 13.11.2.0 provides a machine-friendly, auditable decomposition
of backend decisions into explicit factors and weights, ready to power UIs,
dashboards, and JSON exports without altering the core decision engine.

## APX 13.11.3 – Narrative Explanation Builder (CLI-ready Text)

Version **13.11.3.0** builds the final layer in the explanation stack: a
deterministic **narrative builder** that combines the human-readable base
explanation (13.11.1) with the structured factors (13.11.2) into a single
string suitable for presentation in a CLI or UI.

This layer does not perform any I/O or decide *where* to show the explanation;
it only constructs the text.

### `NarrativeExplanation` and `build_narrative`

The narrative model lives in `src/v13/learning_narrative.rs`:

```rust
#[derive(Debug, Clone)]
pub struct NarrativeExplanation {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub confidence: f32,
    pub narrative: String,
}

pub fn build_narrative(
    text: &DecisionExplanation,
    structured: &StructuredDecisionExplanation,
) -> NarrativeExplanation
```

Rules enforced by the implementation:

- **Read-only** – uses only the provided `DecisionExplanation` and
  `StructuredDecisionExplanation`; does not query or mutate `SelfTrainer`.
- **Deterministic** – given the same inputs, always produces the same
  `NarrativeExplanation` (including the text).
- **No I/O** – no logging, printing, or file access.

The `confidence` field is taken from the base `DecisionExplanation` and
clamped defensively into `[0.0, 1.0]`. A percentage (rounded integer) is used
in the narrative text.

### Narrative structure

`build_narrative` constructs three logical paragraphs:

1. **Decision summary**

   - States the backend and confidence:

     ```text
     "Atenia selected GPU execution with a confidence of 82%. ..."
     ```

   - Appends the base textual explanation (`DecisionExplanation::explanation`)
     to provide a short, human-oriented reason.

2. **Evidence overview**

   - Derives a qualitative description of the historical success rate from the
     `HistoricalSuccessRate` factor weight:
     - `>= 0.75` → "strong historical success rate".
     - `>= 0.40` → "moderate historical success rate".
     - `< 0.40` → "limited historical success rate".
   - Derives a drift/instability sentence from the `DriftPenalty` factor:
     - `weight > 0.0` → drift or instability has been observed.
     - `weight == 0.0` → no significant drift/instability observed.

   The paragraph summarizes these elements without exposing raw numeric
   details:

   ```text
   "The decision is based on a strong historical success rate under similar
   memory conditions. No significant drift or instability has been observed in
   this context."
   ```

3. **Factor breakdown**

   - Iterates over the four factor kinds in a **fixed order**:
     1. `HistoricalSuccessRate` → "historical success".
     2. `DriftPenalty` → "drift impact".
     3. `ObservationCount` → "observation count".
     4. `MemoryStability` → "memory stability".
   - For each factor, looks up its weight and maps it to a qualitative label:

     ```text
     weight >= 0.75 → "high"
     weight >= 0.40 → "medium"
     else           → "low"
     ```

   - Assembles a sentence such as:

     ```text
     "Key contributing factors include historical success factor has high
     influence. drift impact factor has low influence. observation count factor
     has medium influence. memory stability factor has high influence."
     ```

The final narrative is built as:

```text
<paragraph 1>\n\n<paragraph 2>\n\n<paragraph 3>
```

The implementation never prints raw floating-point values for factor weights,
only qualitative labels.

### Tests: `tests/learning_narrative_test.rs`

Four tests validate the narrative builder:

- **`narrative_contains_backend_and_confidence`**

  - Builds a GPU explanation with a given confidence.
  - Asserts that the narrative contains the backend name ("CPU" or "GPU") and
    a percentage sign (`"%"`).

- **`narrative_mentions_drift_when_present`**

  - Uses a structured explanation where `DriftPenalty.weight > 0.0`.
  - Checks that the narrative mentions "drift" or "instability".

- **`narrative_mentions_all_factors`**

  - Builds a narrative from explanations with all four factor kinds present.
  - Verifies that the narrative text includes the four factor labels:
    "historical success", "drift impact", "observation count",
    "memory stability".

- **`narrative_is_deterministic`**

  - Calls `build_narrative` twice with the same `DecisionExplanation` and
    `StructuredDecisionExplanation`.
  - Asserts that `recommended_backend`, `confidence`, and `narrative` are
    identical across calls.

Together, APX 13.11.3.0 completes the explanation stack by providing a
deterministic, human-ready narrative builder that remains fully read-only and
side-effect free, ready to be consumed by future CLI and UI layers.

## APX 13.11.4 – Minimal CLI Explanation Tool (`atenia explain`)

Version **13.11.4.0** adds a minimal CLI entry point that exposes the
explanation pipeline built in 13.11.0–13.11.3. The goal is to provide a
human-readable explanation for a single context via a simple command:

```text
atenia explain --gpu-available=<true|false> --vram-band=<0..3> --ram-band=<0..3>
```

This tool does **not** learn, does not mutate any internal state, and does not
perform any hardware probing. It simply parses the context, queries the
explanation stack, and prints the resulting narrative.

### Binary: `src/bin/atenia.rs`

The CLI is implemented as a separate binary crate entry point:

```rust
fn main() {
    // parse args
    // build LearningContextSnapshot
    // instantiate SelfTrainer
    // run explanation pipeline
    // print narrative or fallback message
}
```

Key properties:

- Lives under `src/bin/` to keep core logic in `src/v13/` untouched.
- Uses no external crates for argument parsing.
- Is deterministic for the same inputs.

### Argument parsing

The CLI expects the subcommand `explain` followed by three required flags:

- `--gpu-available=<true|false>`
- `--vram-band=<0..3>`
- `--ram-band=<0..3>`

Behavior:

- On missing or invalid arguments, prints:

  ```text
  Usage: atenia explain --gpu-available=<true|false> --vram-band=<0..3> --ram-band=<0..3>
  ```

  and exits with status code `1`.

- On valid arguments, constructs a `LearningContextSnapshot` from the parsed
  values.

### Trainer setup and behavior

The CLI currently instantiates a fresh `SelfTrainer` in memory. In this
subversion there is no persistence or warm-start; this is intentional:

- If the trainer has **no learned data** for the given context (which is the
  default behavior for a new trainer), the CLI prints:

  ```text
  No learned data available for the given context.
  ```

- If learned data exists (in future scenarios where the trainer is preloaded),
  the CLI runs the full explanation pipeline.

### Explanation pipeline integration

Given a `LearningContextSnapshot` (`ctx`) and a `SelfTrainer` (`trainer`), the
CLI executes:

```rust
if let Some(text) = trainer.explain_decision(ctx) {
    if let Some(structured) = trainer.explain_decision_structured(ctx) {
        let narrative = build_narrative(&text, &structured);
        println!("{}", narrative.narrative);
    } else {
        println!("No learned data available for the given context.");
    }
} else {
    println!("No learned data available for the given context.");
}
```

This wiring ensures the CLI:

- Uses the same `SelfTrainer` logic as the rest of the engine.
- Combines the textual and structured explanations via `build_narrative`.
- Falls back gracefully when no data is available.

### Tests: `tests/cli_explain_smoke_test.rs`

The CLI is validated by a smoke test that executes the compiled binary via the
`CARGO_BIN_EXE_atenia` environment-provided path:

- Runs:

  ```text
  atenia explain --gpu-available=true --vram-band=0 --ram-band=0
  ```

- Asserts that the process exits successfully.
- Asserts that `stdout` contains either:
  - `"Atenia selected"` (when a narrative is produced), or
  - `"No learned data available for the given context."`.

This keeps the test decoupled from the exact narrative wording while still
ensuring that the CLI is wired correctly and produces human-readable output.

Collectively, APX 13.11.4.0 exposes the explanation stack to users and tools
via a minimal, deterministic CLI without introducing new dependencies or
polluting the core logic.
