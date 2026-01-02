# APX 14.0 — Execution Timeline Engine

This version introduces a **100% observational** layer to record a deterministic and reproducible execution timeline.

It does not change engine behavior or planning: it only observes and serializes what happened, in which order, on which device, and with which relative timestamp.

---

## Goal

Provide an `ExecutionTimeline` that allows:

- Recording execution events in order.
- Assigning **monotonic**, relative timestamps (no wall-clock).
- Exporting the timeline to JSON in a **stable and deterministic** way.

Without:

- Metrics.
- Statistics.
- New heuristics.
- Real hardware dependencies.
- Direct GPU/CPU integration (that will come in APX 14.1+).

---

## Module layout

Base path: `src/v14/`

- `mod.rs`
  - Exposes the `timeline` module.
- `timeline/`
  - `mod.rs`
    - Re-exports the `timeline_event` and `execution_timeline` submodules.
  - `timeline_event.rs`
    - Defines the **pure data** enum `TimelineEvent`.
  - `execution_timeline.rs`
    - Implements `ExecutionTimeline` and `RecordedEvent`.

Associated tests:

- `tests/apx14_execution_timeline.rs`
  - Integration tests that exercise `ExecutionTimeline` and `TimelineEvent` via `#[path = "../src/v14/mod.rs"]`.

---

## TimelineEvent

File: `timeline/timeline_event.rs`

Enum with no logic, only minimal data to describe execution:

- `KernelStart { kernel_id: String, device: String }`
- `KernelEnd { kernel_id: String, device: String }`
- `DeviceSelected { device: String }`
- `MemoryTransfer { src_device: String, dst_device: String, bytes: u64 }`

**Important:**

- It does not contain a `timestamp`.
- The timeline assigns the timestamp when the event is recorded.

---

## ExecutionTimeline

File: `timeline/execution_timeline.rs`

### Structures

- `ExecutionTimeline`
  - `events: Vec<RecordedEvent>`
  - `next_timestamp: u64`
- `RecordedEvent`
  - `timestamp: u64`
  - `event: TimelineEvent`

### Behavior

- `new()`
  - Creates an empty timeline with `next_timestamp = 0`.

- `record(event: TimelineEvent)`
  - Assigns `timestamp = next_timestamp`.
  - Increments `next_timestamp` (monotonic, relative).
  - Pushes the event at the end of the list, preserving order.

- `reset()`
  - Clears the event list completely.
  - Resets `next_timestamp` to 0.

- `events(&self) -> &[RecordedEvent]`
  - Read-only access to recorded events.

- `export_json(&self) -> String`
  - Manual JSON serialization, without `SystemTime` or random fields.
  - Fixed field order and names per variant:
    - `timestamp` (u64)
    - `kind` (string: `"KernelStart"`, `"KernelEnd"`, `"DeviceSelected"`, `"MemoryTransfer"`).
    - Variant-specific fields (e.g. `kernel_id`, `device`, `src_device`, `dst_device`, `bytes`).

---

## Basic usage (hooks)

Intended usage example (not yet integrated with the real runtime):

```rust
use v14::timeline::execution_timeline::ExecutionTimeline;
use v14::timeline::timeline_event::TimelineEvent;

let mut timeline = ExecutionTimeline::new();

timeline.record(TimelineEvent::DeviceSelected {
    device: "cpu0".to_string(),
});

timeline.record(TimelineEvent::KernelStart {
    kernel_id: "k1".to_string(),
    device: "cpu0".to_string(),
});

let json = timeline.export_json();
```

The timeline:

- Does not alter execution decisions.
- Does not introduce waits or real-time reads.
- Can be called from tests or upper layers as a purely observational trace.

---

## Tests

File: `tests/apx14_execution_timeline.rs`

They verify that:

- **Insertion and timestamps**
  - Events are recorded in the expected order.
  - `timestamp` values are strictly increasing.

- **Stable JSON**
  - `export_json()` returns the same string if state does not change.
  - The JSON structure and field order are deterministic.

- **reset()**
  - Completely clears internal state.
  - After `reset()`, new events start again at `timestamp = 0`.

Running only APX 14.0 tests:

```bash
cargo test --test apx14_execution_timeline
```

---

## Design principles

- Passive layer: only observes and records.
- No IO or logs inside the timeline.
- No real hardware integration.
- No metrics or derived statistics.
- Stable JSON as a basis for deep debugging and later analysis (APX 14.1+).

---

# APX 14.1 — Memory Pressure & Fragmentation Analyzer

APX 14.1 adds a passive memory pressure and fragmentation analyzer, also fully isolated from the real runtime.

It observes logical memory states in three conceptual layers (VRAM, RAM, SSD) and classifies them, without making decisions or changing engine behavior.

---

## Goal

Answer, in a structured and deterministic way:

- How much memory is in use?
- How close are we to the limit?
- Is pressure going up, stable, or going down?
- Is there observable fragmentation (under a simple model)?
- Are we in a risk zone (pre-OOM) under a fixed threshold?

Always in a **passive** way: no offloading, no fallbacks, no policy changes.

---

## Module layout

Within `src/v14/`:

- `mod.rs`
  - Also exposes the `memory` module.
- `memory/`
  - `mod.rs`
    - Re-exports `memory_layer`, `pressure_snapshot`, `fragmentation`, `pressure_analyzer`.
  - `memory_layer.rs`
    - Defines the `MemoryLayer` enum.
  - `pressure_snapshot.rs`
    - Defines `PressureSnapshot` and `MemoryRiskLevel`.
  - `fragmentation.rs`
    - Implements the simple fragmentation model.
  - `pressure_analyzer.rs`
    - Implements `MemoryPressureAnalyzer`, `PressureTrend` and `AnalyzerResult`.

Associated tests:

- `tests/apx14_memory_pressure_test.rs`
  - Integration tests that exercise the analyzer via `#[path = "../src/v14/mod.rs"]`.

---

## MemoryLayer

File: `memory/memory_layer.rs`

Enum for logical memory layers (not real hardware):

- `MemoryLayer::VRAM`
- `MemoryLayer::RAM`
- `MemoryLayer::SSD`

Used to tag snapshots and separate observation by "type" of memory.

---

## PressureSnapshot & risk

File: `memory/pressure_snapshot.rs`

### MemoryRiskLevel

Simple enum to classify risk based on `pressure_ratio`:

- `Safe`
- `Warning`
- `Critical`
- `PreOOM`

Fixed classification rule (`classify_risk`):

- `pressure_ratio >= 0.98` → `PreOOM`
- `0.90 <= ratio < 0.98` → `Critical`
- `0.75 <= ratio < 0.90` → `Warning`
- `< 0.75` → `Safe`

This guarantees that `PreOOM` triggers **before** `used == capacity`.

### PressureSnapshot

Immutable snapshot of a logical memory state:

- `layer: MemoryLayer`
- `used_bytes: u64`
- `capacity_bytes: u64`
- `pressure_ratio: f64` (computed as `used / capacity` or `0.0` if `capacity == 0`)
- `fragmentation_ratio: f64` (computed externally, e.g. using `compute_fragmentation_ratio`)
- `risk_level: MemoryRiskLevel` (derived from `pressure_ratio`)
- `timestamp: u64` (monotonic, relative)

The `PressureSnapshot::new(...)` constructor is pure and does not touch real memory or real time.

---

## Fragmentation model

File: `memory/fragmentation.rs`

Simple and deterministic model:

```text
fragmentation = 1.0 - (largest_block / total_free)
```

Where:

- `total_free_bytes`: total logical free memory.
- `largest_block_bytes`: largest logical contiguous free block.

Special cases:

- If `total_free_bytes == 0` → fragmentation `0.0`.
- The result is clamped to `[0.0, 1.0]`.

It does not simulate real alloc/free; it only operates on numbers provided by the caller.

---

## MemoryPressureAnalyzer

File: `memory/pressure_analyzer.rs`

### Types

- `PressureTrend`
  - `Up`
  - `Stable`
  - `Down`

- `AnalyzerResult`
  - `latest: Option<PressureSnapshot>`
  - `trend: Option<PressureTrend>`

- `MemoryPressureAnalyzer`
  - `history: Vec<PressureSnapshot>`
  - `max_history: usize`

### Behavior

- `new(max_history: usize)`
  - Initializes the analyzer with a maximum history size (at least 1).

- `record(snapshot: PressureSnapshot)`
  - Appends a new snapshot to the end of the history.
  - If `max_history` is exceeded, discards the oldest snapshots (FIFO).

- `reset()`
  - Completely clears the history.

- `history(&self) -> &[PressureSnapshot]`
  - Passive, read-only access to the history.

- `analyze() -> AnalyzerResult`
  - Returns the latest snapshot (`latest`) if present.
  - Computes the trend (`trend`) from `pressure_ratio` values.

### Trend computation

`compute_trend(history: &[PressureSnapshot])`:

- If fewer than 2 snapshots → `None`.
- Iterates over `prev, next` windows and counts:
  - `up++` if `next.pressure_ratio > prev.pressure_ratio`.
  - `down++` if `next.pressure_ratio < prev.pressure_ratio`.
- Decision rules:
  - `up == 0 && down == 0` → `Stable`.
  - `up > down` → `Up`.
  - `down > up` → `Down`.
  - Tie → `Stable`.

Everything is purely deterministic and side-effect free.

---

## APX 14.1 tests

File: `tests/apx14_memory_pressure_test.rs`

Covered cases:

- **Pressure ratio + risk calculation**
  - Different `used/capacity` combinations to verify `Safe`, `Warning`, `Critical`, `PreOOM`.

- **Trend detection**
  - Stable sequence → `Stable`.
  - Increasing sequence → `Up`.
  - Decreasing sequence → `Down`.
  - Also validates history length and timestamp of the latest snapshot.

- **Reproducible fragmentation**
  - Test cases where `fragmentation = 1 - largest/total` yields known values (0.0, 0.5, 0.9, etc.).
  - Well-defined `total_free = 0` case (0.0).
  - Explicit use of `MemoryLayer::SSD` to document that the layer exists and is usable.

Running only APX 14.1 tests:

```bash
cargo test --test apx14_memory_pressure_test
```

---

## Design principles (APX 14.1)

- Passive analysis layer: only observes and classifies.
- No access to real memory or vendor APIs.
- No offloading, no fallback, no policy changes.
- No logging or IO.
- Fully deterministic behavior (same input → same output).

---

# APX 14.2 — Decision Reasoning Engine

APX 14.2 introduces a decision reasoning engine that **does not decide**; it only explains why an already-taken decision is reasonable in its context.

It builds on APX 14.0 (logical time) and APX 14.1 (memory state) by passing data explicitly, without auto-wiring into the runtime.

---

## Goal

Answer, clearly and structurally:

- What decision was made?
- In what context?
- Which factors influenced it?
- Which alternatives were considered or avoided?
- Why was this decision reasonable (via structured codes)?

Always **without** changing the original decision or triggering new actions.

---

## Module layout

Within `src/v14/`:

- `mod.rs`
  - Also exposes the `reasoning` module.
- `reasoning/`
  - `mod.rs`
    - Re-exports `decision_event`, `reasoning_factors`, `decision_record`, `decision_reasoner`.
  - `decision_event.rs`
    - Defines `DecisionEventKind` and `DecisionEvent`.
  - `reasoning_factors.rs`
    - Defines `ReasoningFactors` using `PressureSnapshot` and `MemoryRiskLevel` from APX 14.1.
  - `decision_record.rs`
    - Defines `DecisionRecord` as a complete explainable unit.
  - `decision_reasoner.rs`
    - Implements `DecisionReasoner` and deterministic JSON export.

Associated tests:

- `tests/apx14_decision_reasoning_test.rs`
  - Integration tests that call the reasoner manually (no real runtime) via `#[path = "../src/v14/mod.rs"]`.

---

## DecisionEvent

File: `reasoning/decision_event.rs`

### DecisionEventKind

Enum for the **type of decision**:

- `DeviceSelection`
- `TensorMovement`
- `KernelPlacement`
- `FallbackAvoided`

It is extensible but has no internal logic.

### DecisionEvent

Structure that captures what happened:

- `id: String` — logical identifier of the decision.
- `kind: DecisionEventKind` — decision type.
- `object_id: String` — affected object (e.g. tensor, kernel, graph).
- `timestamp: u64` — logical time, assigned by the reasoner.

---

## ReasoningFactors

File: `reasoning/reasoning_factors.rs`

Groups explanatory factors that already exist or are computed in other layers:

- `memory_snapshot: Option<PressureSnapshot>`
- `memory_risk: Option<MemoryRiskLevel>`
- `fragmentation_ratio: Option<f64>`
- `device_available: Option<bool>`
- `recent_decisions_count: Option<u64>`

Constructor:

```rust
pub fn new(
    memory_snapshot: Option<PressureSnapshot>,
    memory_risk: Option<MemoryRiskLevel>,
    fragmentation_ratio: Option<f64>,
    device_available: Option<bool>,
    recent_decisions_count: Option<u64>,
) -> Self
```

This module **does not compute** these factors: it only receives and packages them for explanation.

---

## DecisionRecord

File: `reasoning/decision_record.rs`

Complete reasoning unit for a decision:

- `event: DecisionEvent` — what happened.
- `factors: ReasoningFactors` — explanatory context.
- `avoided_alternative: Option<String>` — main avoided alternative (e.g. `"CPUFallback"`).
- `justification_code: u32` — structured justification code (no free text).
- `timestamp: u64` — logical time (same as the `event` timestamp).

Constructor:

```rust
DecisionRecord::new(event, factors, avoided_alternative, justification_code, timestamp)
```

This allows building explainable, structured narratives without losing information.

---

## DecisionReasoner

File: `reasoning/decision_reasoner.rs`

### Structure

- `DecisionReasoner`
  - `records: Vec<DecisionRecord>`
  - `next_timestamp: u64`

### Behavior

- `new()`
  - Empty reasoner, `next_timestamp = 0`.

- `record_decision(...)`
  - Receives **an already-taken decision** and its factors.
  - Builds a `DecisionEvent` and a `DecisionRecord` with a monotonic logical `timestamp`.
  - Appends the record to `records`, preserving temporal order.
  - Does not decide, correct, or re-plan.

- `records(&self) -> &[DecisionRecord]`
  - Read-only access to the history.

- `reset()`
  - Clears the history and resets `next_timestamp` to 0.

- `export_json(&self) -> String`
  - Manually serializes the list of `DecisionRecord` to JSON.
  - Fixed fields and order:
    - `timestamp`
    - `event` (nested)
    - `factors` (nested)
    - `avoided_alternative`
    - `justification_code`
  - Uses fixed strings (`"DeviceSelection"`, `"Critical"`, `"VRAM"`, etc.).
  - No random fields, no reordering → stable JSON.

Serialization of `factors` and `memory_snapshot` uses APX 14.1 types (`PressureSnapshot`, `MemoryRiskLevel`, `MemoryLayer`).

---

## APX 14.2 tests

File: `tests/apx14_decision_reasoning_test.rs`

Covered cases:

- **Decision + factors + stable JSON**
  - Creates a `DecisionReasoner`.
  - Builds a VRAM `PressureSnapshot` and a rich `ReasoningFactors` (risk `Critical`, fragmentation, availability, history).
  - Records a `DeviceSelection` decision with an avoided alternative and `justification_code = 42`.
  - Verifies:
    - History contains 1 record.
    - IDs, `object_id`, and timestamps are correct.
    - `export_json()` is stable (same output on repeated calls).
    - JSON contains the expected key fields.

- **Temporal order and reset**
  - Records several simple decisions with empty factors.
  - Checks that timestamps grow monotonically.
  - After `reset()`, the next record starts again at `timestamp = 0`.

Running only APX 14.2 tests:

```bash
cargo test --test apx14_decision_reasoning_test
```

---

## Design principles (APX 14.2)

- The reasoner **does not make decisions**: it only explains external decisions.
- Consumes data from APX 14.0 and 14.1 without auto-wiring.
- No hidden heuristics or "intelligent" logic.
- Structures and names designed for the paper and explainable debugging.
- Stable JSON output, suitable for offline analysis and tooling.

---

# APX 14.3 — Failure & Recovery Trace

APX 14.3 introduces a structured tracer for failures, failure risks, and recoveries that have already happened (or are simulated).

It does not perform recovery or trigger fallbacks: it only records **what** failed, **what** was avoided, **which** recovery was taken externally, and **what** the final outcome was.

---

## Goal

Provide structured, deterministic evidence to:

- Detect and describe failure risks (e.g. pre-OOM).
- Record concrete failures.
- Record recovery actions chosen externally.
- Classify the outcome (recovered, degraded, failed, avoided).

All without touching the real runtime or triggering new decisions.

---

## Module layout

Within `src/v14/`:

- `mod.rs`
  - Also exposes the `failure` module.
- `failure/`
  - `mod.rs`
    - Re-exports `failure_kind`, `recovery_action`, `failure_event`, `recovery_record`, `failure_trace`.
  - `failure_kind.rs`
    - Defines the `FailureKind` enum.
  - `recovery_action.rs`
    - Defines the `RecoveryAction` enum.
  - `failure_event.rs`
    - Defines `FailureEvent` and `FailureSeverity`.
  - `recovery_record.rs`
    - Defines `RecoveryRecord` and `RecoveryResult`.
  - `failure_trace.rs`
    - Implements `FailureTrace` and deterministic JSON export.

Associated tests:

- `tests/apx14_failure_trace_test.rs`
  - Integration tests that inject failures and recoveries manually via `#[path = "../src/v14/mod.rs"]`.

---

## FailureKind

File: `failure/failure_kind.rs`

Enum to categorize failures and pre-failures:

- `OutOfMemoryRisk`
- `OutOfMemory`
- `KernelLaunchFailure`
- `DeviceUnavailable`
- `TransferFailure`
- `Unknown`

Purely descriptive; no logic.

---

## RecoveryAction

File: `failure/recovery_action.rs`

Enum to describe **recovery actions already taken**:

- `Retry`
- `FallbackToCPU`
- `MoveTensorToRAM`
- `MoveTensorToSSD`
- `ReduceBatch`
- `SkipKernel`
- `Abort`
- `None`

These actions are not executed here; they are only documented.

---

## FailureEvent

File: `failure/failure_event.rs`

### FailureSeverity

Simple severity enum:

- `Info`
- `Warning`
- `Critical`

### FailureEvent

Represents a failure or pre-failure event:

- `kind: FailureKind`
- `timestamp: u64` — logical, relative.
- `message: String` — short, stable message.
- `device: Option<String>` — involved device (if applicable).
- `tensor_id: Option<String>` — affected tensor (if applicable).
- `kernel_id: Option<String>` — affected kernel (if applicable).
- `severity: FailureSeverity`

Constructor:

```rust
FailureEvent::new(kind, timestamp, message, device, tensor_id, kernel_id, severity)
```

---

## RecoveryRecord

File: `failure/recovery_record.rs`

### RecoveryResult

Enum to classify recovery outcome:

- `Recovered`
- `Degraded`
- `Failed`
- `Avoided`

### RecoveryRecord

Combines into a single record:

- `failure_event: FailureEvent` — what failed or was about to fail.
- `action_taken: RecoveryAction` — how the system responded (externally).
- `action_reason: String` — short, stable reason.
- `result: RecoveryResult` — recovery outcome.
- `timestamp: u64` — logical time (same as the `failure_event` timestamp).

Constructor:

```rust
RecoveryRecord::new(failure_event, action_taken, action_reason, result, timestamp)
```

---

## FailureTrace

File: `failure/failure_trace.rs`

### Structure

- `FailureTrace`
  - `records: Vec<RecoveryRecord>`
  - `next_timestamp: u64`

### Behavior

- `new()`
  - Initializes an empty trace with `next_timestamp = 0`.

- `record_failure_with_recovery(...)`
  - Receives full information about a failure/pre-failure and its recovery already executed externally.
  - Assigns a monotonic logical `timestamp`.
  - Builds the `FailureEvent` and `RecoveryRecord`.
  - Appends it at the end of `records` (preserving temporal order).

- `records(&self) -> &[RecoveryRecord]`
  - Passive exposure of the trace.

- `reset()`
  - Clears the trace and resets `next_timestamp` to 0.

- `export_json(&self) -> String`
  - Manually serializes the list of `RecoveryRecord`.
  - Fixed fields and order:
    - `timestamp`
    - `failure_event` (with `kind`, `timestamp`, `message`, `device`, `tensor_id`, `kernel_id`, `severity`)
    - `action_taken`
    - `action_reason`
    - `result`
  - Fixed strings for enums (e.g. `"OutOfMemoryRisk"`, `"MoveTensorToRAM"`, `"Avoided"`).
  - No randomness or reordering → stable JSON.

---

## APX 14.3 tests

File: `tests/apx14_failure_trace_test.rs`

Covered cases:

- **Pre-failure + descriptive recovery**
  - Records an `OutOfMemoryRisk` on `gpu0` with `Warning` severity.
  - Action: `MoveTensorToRAM` with `RecoveryResult::Avoided`.
  - Verifies `FailureEvent`, `RecoveryRecord`, and timestamps.
  - Checks that `export_json()` is stable and contains the expected keys.

- **Temporal order and reset**
  - Records several failures (`TransferFailure` → `Recovered`, `DeviceUnavailable` → `Degraded`).
  - Verifies increasing timestamps.
  - After `reset()`, a new failure (`Unknown` → `Failed`) again has `timestamp = 0`.

Running only APX 14.3 tests:

```bash
cargo test --test apx14_failure_trace_test
```

---

## Design principles (APX 14.3)

- The module **does not recover** anything: it only describes how recovery happened.
- No automatic integration with the real runtime.
- No heuristics, no new decisions.
- Structured traces, suitable for debugging, auditing, and the paper.
- Stable, reproducible JSON (same input → same output).
