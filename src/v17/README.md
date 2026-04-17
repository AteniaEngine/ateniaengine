# APX v17 — Model Runtime Integration

> [!NOTE]
> **Scope and status.**  
> This document describes the architecture and design intent of APX v17.  
> Several capabilities listed below exist as structural scaffolding —
> pure data structures, validators, and deterministic logic — and are
> not yet wired to runtime signals from real hardware. See the
> [main README](../../README.md) for the current state of each component.

This directory contains the **APX 17.x** model runtime integration layers.

APX 17.0 introduces a stable, auditable, and extensible definition of what a
*model* is for Atenia, without loading weights, running inference, or touching
hardware. Later 17.x versions can add loaders and execution strategies on top of
this foundation.

- APX 17.0 — Model Artifact Layer (this file)

---

## APX 17.0 — Model Artifact Layer

APX 17.0 defines a **pure, runtime-independent representation of a model**. It
separates *what* a model is (identity, metadata, format, size, location) from
*how* it is loaded or executed (which is handled in later versions).

APX 17.0 guarantees:

- No weight loading.
- No inference or runtime calls.
- No filesystem or hardware access.
- Deterministic, immutable model descriptions.

### Directory layout (17.0)

```text
src/v17/
    mod.rs
    model/
        mod.rs
        model_artifact.rs
        model_metadata.rs
        model_format.rs
        model_errors.rs
```

Tests for APX 17.0 live in:

```text
tests/model_artifact_test.rs
```

---

## ModelFormat

File: `src/v17/model/model_format.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFormat {
    Onnx,
    SafeTensors,
    Gguf,
    Raw,
}
```

`ModelFormat` is a **closed enum** describing supported on-disk model formats.
It does not include parsers or IO logic; it only declares the format so that
loaders and policies can branch safely in later layers.

---

## ModelMetadata

File: `src/v17/model/model_metadata.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub family: String,
    pub author: Option<String>,
    pub checksum: String,
    pub estimated_size_bytes: u64,
}
```

`ModelMetadata` is immutable descriptive information about a model:

- `name` – human-readable model name.
- `version` – semantic or vendor version.
- `family` – family/type (e.g. "llm", "vision", "embedding").
- `author` – optional origin or owner string.
- `checksum` – hash string for integrity/audit.
- `estimated_size_bytes` – declared size of the artifact.

The `new(...)` constructor is a pure function: it only builds the struct and
performs no IO.

---

## ModelError

File: `src/v17/model/model_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    InvalidMetadata(String),
    UnsupportedFormat(String),
    InvalidPath(String),
    InvalidSize(String),
}
```

`ModelError` represents explicit failures in constructing or validating a model
artifact:

- `InvalidMetadata` – missing or malformed required metadata fields.
- `UnsupportedFormat` – reserved for formats that are not allowed by Atenia.
- `InvalidPath` – empty or syntactically invalid logical path/URI.
- `InvalidSize` – impossible or zero sizes.

Errors are descriptive and deterministic.

---

## ModelArtifact

> [!IMPORTANT]
> APX 17.0–17.1 define the `ModelArtifact` and `ModelLoader` interfaces.  
> The current MNIST integration (APX 17.10.x) uses
> `MnistCNNModel::synthetic()` directly and does **not** exercise
> `ModelLoader`: weights are constructed in code rather than loaded from
> disk. Full loader integration — reading real weight files into the
> inference pipeline — is still pending.

File: `src/v17/model/model_artifact.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    pub id: String,
    pub metadata: ModelMetadata,
    pub format: ModelFormat,
    pub location: String,
    pub total_size_bytes: u64,
}
```

`ModelArtifact` is the **core abstraction** for models in APX 17.0. It is an
immutable description of a model artifact known to Atenia:

- `id` – stable identifier within Atenia.
- `metadata` – descriptive `ModelMetadata`.
- `format` – `ModelFormat` describing the on-disk representation.
- `location` – logical path or URI where weights live.
- `total_size_bytes` – declared total size of the artifact.

### Construction

```rust
impl ModelArtifact {
    pub fn new(
        id: String,
        metadata: ModelMetadata,
        format: ModelFormat,
        location: String,
        total_size_bytes: u64,
    ) -> Result<Self, ModelError> { /* ... */ }
}
```

`ModelArtifact::new` performs **pure validation only**:

- Fails with `ModelError::InvalidMetadata` when:
  - `id` is empty.
  - `metadata.name`, `metadata.version`, or `metadata.checksum` are empty.
- Fails with `ModelError::InvalidPath` when `location` is empty/whitespace.
- Fails with `ModelError::InvalidSize` when `total_size_bytes == 0`.

All `ModelFormat` variants defined in the enum are considered supported by
construction; there is no runtime probing, IO, or hardware interaction.

The resulting `ModelArtifact` is immutable and `Clone + Eq`, which makes it
safe to store, compare, and log.

---

## Tests for APX 17.0

File: `tests/model_artifact_test.rs`

Tests validate that model artifacts are deterministic, immutable, and safe:

1. **ModelArtifact is constructed from valid metadata**

   - Builds a `ModelArtifact` from a valid `ModelMetadata` instance.
   - Asserts that all fields are preserved (id, metadata, format, location,
     size).

2. **Invalid metadata yields explicit error**

   - Uses empty `metadata.name` and expects `ModelError::InvalidMetadata`.

3. **Invalid location or size yield explicit errors**

   - Empty/whitespace `location` → `ModelError::InvalidPath`.
   - `total_size_bytes == 0` → `ModelError::InvalidSize`.

4. **Construction is deterministic and artifact is immutable**

   - Two calls to `ModelArtifact::new` with identical inputs produce identical
     `ModelArtifact` values.

All tests are pure, do not touch the filesystem or hardware, and are
CI-friendly:

```bash
cargo test --test model_artifact_test
```

---

## Status of APX 17.0

APX 17.0 is considered **DONE** when:

- There is a clear, immutable definition of a model artifact.
- The model abstraction is independent of runtime, device, and execution.
- No loading or inference logic exists in this layer.
- Errors for invalid metadata, paths, and sizes are explicit.
- Tests validate determinism, immutability, and absence of side effects.

The current implementation meets these criteria and provides a solid foundation
for future APX 17.x layers (loaders, placement policies, and actual execution
runtimes).

---

## APX 17.1 — Model Loader & Memory Mapping

APX 17.1 adds a **controlled model loader** that turns `ModelArtifact`s into
in-memory buffers ready for execution, without running any inference or
interpreting model structure.

- It may read model bytes from disk.
- It may allocate RAM for those bytes.
- It must not execute kernels or perform mathematical operations.

### Directory layout additions (17.1)

```text
src/v17/
    loader/
        mod.rs
        model_loader.rs
        memory_map.rs
        loader_policy.rs
        loader_errors.rs
```

Tests for APX 17.1 live in:

```text
tests/model_loader_test.rs
```

---

## LoaderError

File: `src/v17/loader/loader_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    FileNotFound(String),
    SizeMismatch { expected: u64, actual: u64 },
    InsufficientMemory { required: u64, available: u64 },
    PolicyDenied(String),
    IoError(String),
}
```

`LoaderError` exposes explicit failure modes for loading:

- `FileNotFound` – logical path in `ModelArtifact.location` does not exist.
- `SizeMismatch` – on-disk size does not match the declared artifact size.
- `InsufficientMemory` – policy determined that RAM is not sufficient.
- `PolicyDenied` – reserved for future policy checks.
- `IoError` – OS-level IO failures.

---

## MemoryMap

File: `src/v17/loader/memory_map.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySegment {
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMap {
    pub artifact_id: String,
    pub total_size_bytes: u64,
    pub loaded_bytes: u64,
    pub segments: Vec<MemorySegment>,
}
```

`MemoryMap` describes how a model's bytes are laid out in memory:

- `artifact_id` – the associated `ModelArtifact`.
- `total_size_bytes` – declared size.
- `loaded_bytes` – number of bytes actually placed in RAM.
- `segments` – contiguous segments (single segment in APX 17.1).

The helper method `fully_loaded()` returns `true` when `loaded_bytes` matches
`total_size_bytes`.

---

## LoaderPolicy

File: `src/v17/loader/loader_policy.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderPolicy {
    LoadAll,
    FailIfInsufficientRam,
}
```

`LoaderPolicy` expresses simple, deterministic loading strategies:

- `LoadAll` – load the full model if there is enough RAM.
- `FailIfInsufficientRam` – fail immediately when `available < required`.

The method `check_memory(required, available)` returns `Ok(())` if loading is
permitted, or `LoaderError::InsufficientMemory` otherwise.

---

## ModelLoader & LoadedModelHandle

File: `src/v17/loader/model_loader.rs`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModelHandle {
    pub artifact_id: String,
    pub format: ModelFormat,
    pub memory_map: MemoryMap,
    pub bytes: Vec<u8>,
}
```

`LoadedModelHandle` is a pure in-memory representation of a loaded model:

- It holds raw bytes only.
- No tensors, kernels, or execution graphs are created here.

Core API:

```rust
pub struct ModelLoader;

impl ModelLoader {
    pub fn load(
        artifact: &ModelArtifact,
        policy: &LoaderPolicy,
        available_ram_bytes: u64,
    ) -> Result<LoadedModelHandle, LoaderError> { /* ... */ }
}
```

Behavior of `load`:

- Uses `LoaderPolicy::check_memory` to enforce RAM constraints.
- Validates that the file at `artifact.location` exists.
- Reads filesystem metadata and compares size to `artifact.total_size_bytes`.
- Reads the file into a `Vec<u8>` and checks the length again.
- Builds a `MemoryMap` with a single segment `[0, total_size_bytes]` and sets
  `loaded_bytes == total_size_bytes`.
- Returns a `LoadedModelHandle` with the artifact id, format, memory map, and
  raw bytes.

No compute is performed; bytes are not transformed, only moved into RAM.

---

## Tests for APX 17.1

File: `tests/model_loader_test.rs`

Tests use temporary files to simulate real model artifacts without touching any
actual production paths.

The test suite covers:

1. **Valid model is loaded into RAM**

   - Writes a small file to a temporary path and constructs a matching
     `ModelArtifact`.
   - `ModelLoader::load` returns a handle whose `bytes` equal the file
     contents and whose `MemoryMap` is fully loaded.

2. **Insufficient memory yields explicit error**

   - Uses `LoaderPolicy::FailIfInsufficientRam` with `available_ram_bytes` less
     than the model size.
   - Expects `LoaderError::InsufficientMemory` with correct required and
     available values.

3. **Loaded model matches artifact size and is deterministic**

   - Two consecutive loads of the same artifact with the same policy produce
     identical `LoadedModelHandle` values (same bytes and memory map).

4. **No compute is triggered during load**

   - Writes a non-trivial byte pattern (0..15) and verifies that the loader
     returns exactly the same bytes, proving that no transformations occur.

All tests are CI-safe and avoid any inference or hardware-specific behavior:

```bash
cargo test --test model_loader_test
```

---

## Status of APX 17.1

APX 17.1 is considered **DONE** when:

- A model can be read from disk into RAM via `ModelLoader`.
- File size and declared artifact size are validated.
- Simple memory pressure constraints are enforced via `LoaderPolicy`.
- No inference or model interpretation occurs in this layer.
- Tests validate determinism, safety, and the absence of unintended compute.

The current implementation meets these criteria, turning on-disk
`ModelArtifact`s into loaded, memory-mapped handles that later APX 17.x layers
can execute.

---

## APX 17.2 — Minimal CPU Compute Backend

APX 17.2 introduces a **minimal, CPU-only compute backend**. Its goal is to
prove that Atenia can execute real inference under strict control, not to be a
high-performance engine.

- CPU-only, no GPU or accelerators.
- Single-threaded, deterministic operations.
- No auto-tuning or performance heuristics.

### Directory layout additions (17.2)

```text
src/v17/
    compute/
        mod.rs
        tensor.rs
        ops.rs
        cpu_backend.rs
        compute_errors.rs
```

Tests for APX 17.2 live in:

```text
tests/cpu_backend_test.rs
```

---

## Tensor

File: `src/v17/compute/tensor.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
```

`Tensor` is a minimal tensor representation:

- `shape` – list of dimensions.
- `data` – flat `f32` buffer in row-major order.

The constructor `Tensor::new(shape, data)` verifies that the product of
dimensions matches the length of `data`, returning an error string otherwise.
This check is used by ops to surface `ShapeMismatch` errors.

---

## ComputeError

File: `src/v17/compute/compute_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeError {
    ShapeMismatch(String),
    ContractViolation(String),
    AbortedByGuard(String),
}
```

`ComputeError` models failures during compute:

- `ShapeMismatch` – incompatible tensor shapes for an operation.
- `ContractViolation` – execution would violate the `ExecutionContract`.
- `AbortedByGuard` – execution was stopped due to a guard decision.

---

## Ops

File: `src/v17/compute/ops.rs`

The initial closed set of operations includes:

- `add(a, b) -> Tensor`
  - Elementwise addition, requires identical shapes.
- `matmul(a, b) -> Tensor`
  - Matrix multiplication for 2D tensors, with strict shape checks.
- `relu(x) -> Tensor`
  - Elementwise ReLU activation.

All ops are:

- CPU-only, single-threaded.
- Deterministic and pure (operate only on inputs, produce new tensors).
- Guarded by shape validation and return `ComputeError::ShapeMismatch` when
  constraints are not met.

---

## CpuBackend

File: `src/v17/compute/cpu_backend.rs`

```rust
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { /* ... */ }

    pub fn run_inference(
        &self,
        model: &LoadedModelHandle,
        input: &Tensor,
        contract: &ExecutionContract,
        guard_action: GuardAction,
    ) -> Result<Tensor, ComputeError> { /* ... */ }
}
```

`CpuBackend` executes a minimal inference pipeline:

- Respects `ExecutionContract` from APX 16.0:
  - If `require_stability` is true and the runtime snapshot is unstable, it
    returns `ComputeError::ContractViolation`.
- Respects guard decisions from APX 16.4:
  - `GuardAction::Abort` → `ComputeError::AbortedByGuard`.
  - `GuardAction::Continue` / `Degrade` → proceed.

Execution model in APX 17.2:

- Interprets `LoadedModelHandle.bytes` as a tiny square weight matrix
  (n×n, `f32`, little-endian).
- Builds a `Tensor` `W` of shape `[n, n]` from the bytes.
- Runs `matmul(W, input)` followed by `relu`.

This is intentionally minimal and deterministic, sufficient to prove full stack
integration from artifacts and loaders through contracts and guards to CPU
compute.

---

## Tests for APX 17.2

File: `tests/cpu_backend_test.rs`

Tests wire together:

- `ModelArtifact` + `ModelLoader` (APX 17.0–17.1).
- `ExecutionContract` (16.0).
- `GuardAction` (16.4).
- `CpuBackend` and `Tensor` (17.2).

The test suite covers:

1. **Simple inference produces correct output**

   - Defines a 2×2 weight matrix `W = [[1, 0], [0, -1]]` stored in little-endian
     bytes.
   - Loads the model and runs `run_inference` with an input vector.
   - Verifies the expected ReLU-transformed output.

2. **Backend respects execution contract**

   - Uses an `ExecutionContract` with `is_stable = false` and
     `require_stability = true`.
   - Expects `ComputeError::ContractViolation`.

3. **Invalid shapes yield explicit error**

   - Calls `matmul` with incompatible shapes and expects
     `ComputeError::ShapeMismatch`.

4. **Inference is deterministic and abortable**

   - Two identical inferences (same model, input, contract, guard) produce
     identical outputs.
   - A run with `GuardAction::Abort` returns `ComputeError::AbortedByGuard`.

All tests are CPU-only, avoid any hardware-specific features, and are
CI-friendly:

```bash
cargo test --test cpu_backend_test
```

---

## Status of APX 17.2

APX 17.2 is considered **DONE** when:

- Atenia can execute real, minimal inference on CPU.
- Compute obeys `ExecutionContract` and guard decisions.
- Errors for invalid shapes and contract violations are explicit.
- Inference is deterministic and abortable.
- Tests validate correctness and determinism.

The current implementation meets these criteria, providing a first, controlled
compute backend that future APX 17.x versions can extend for performance and
multi-device support.

---

## APX 17.3 — Execution Adapter

APX 17.3 adds an **execution adapter** that connects the orchestration engine
from APX 16 (contracts + planner + guards) with the v17 CPU backend. It does
not decide what to execute or optimize the plan; it only follows the
`ExecutionPlan`.

### Directory layout additions (17.3)

```text
src/v17/
    adapter/
        mod.rs
        adapter_errors.rs
        adapter_context.rs
        step_dispatcher.rs
        execution_adapter.rs
```

Tests for APX 17.3 live in:

```text
tests/execution_adapter_test.rs
```

---

## AdapterError

File: `src/v17/adapter/adapter_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum AdapterError {
    UnsupportedStep(String),
    ContractViolation(String),
    AbortedByGuard(String),
    BackendFailure(String),
}
```

`AdapterError` captures failures in the adapter layer:

- `UnsupportedStep` – reserved for plan steps the adapter cannot dispatch.
- `ContractViolation` – reserved for contract-level issues detected here.
- `AbortedByGuard` – execution cancelled due to guard decisions.
- `BackendFailure` – errors bubbled up from the CPU backend.

---

## AdapterContext

File: `src/v17/adapter/adapter_context.rs`

```rust
#[derive(Debug)]
pub struct AdapterContext {
    pub model: LoadedModelHandle,
    pub contract: ExecutionContract,
    pub guard_action: GuardAction,
    pub executed_steps: Vec<usize>,
    pub last_output: Option<Tensor>,
    pub aborted: bool,
}
```

`AdapterContext` holds ephemeral state for a single plan execution:

- `model` – `LoadedModelHandle` (17.1).
- `contract` – `ExecutionContract` (16.0).
- `guard_action` – current `GuardAction` (16.4).
- `executed_steps` – indices of steps that have been visited.
- `last_output` – last tensor produced by the backend.
- `aborted` – flag indicating that execution was stopped.

It is not persisted and is intended to be rebuilt per run.

---

## StepDispatcher

File: `src/v17/adapter/step_dispatcher.rs`

```rust
pub struct StepDispatcher;

impl StepDispatcher {
    pub fn dispatch_step(
        backend: &CpuBackend,
        ctx: &mut AdapterContext,
        step_index: usize,
        kind: &PlanStepKind,
        input: &Tensor,
    ) -> Result<(), AdapterError> { /* ... */ }
}
```

`StepDispatcher` translates individual `PlanStepKind` values into backend
operations:

- Records the `step_index` in `ctx.executed_steps`.
- For `PlanStepKind::MarkTensorsMovable`, invokes
  `CpuBackend::run_inference` using the model, contract, and guard action from
  the `AdapterContext`.
- For non-compute planning steps such as `EnsureMemoryHeadroom`,
  `SelectBackendCandidate`, and `PrepareFallback`, performs no action at the
  compute level (treated as adapter-level no-ops).

Backend errors are wrapped as `AdapterError::BackendFailure`.

---

## ExecutionAdapter

File: `src/v17/adapter/execution_adapter.rs`

```rust
pub struct ExecutionAdapter {
    backend: CpuBackend,
}

impl ExecutionAdapter {
    pub fn new(backend: CpuBackend) -> Self { /* ... */ }

    pub fn execute_plan(
        &self,
        plan: &ExecutionPlan,
        ctx: &mut AdapterContext,
        input: &Tensor,
    ) -> Result<Tensor, AdapterError> { /* ... */ }
}
```

`ExecutionAdapter` iterates over the `ExecutionPlan` from APX 16.1:

- Executes steps strictly in plan order; no reordering.
- Checks `ctx.guard_action` before each step and aborts with
  `AdapterError::AbortedByGuard` when guards request it.
- Delegates per-step behavior to `StepDispatcher::dispatch_step`.
- Returns the last tensor produced (`ctx.last_output`) or a
  `BackendFailure` error if no output was produced by the plan.

The adapter does not modify the plan, contract, or model; it only orchestrates
execution according to the plan.

---

## Tests for APX 17.3

File: `tests/execution_adapter_test.rs`

Tests verify that the adapter correctly follows the `ExecutionPlan` and
interacts with the CPU backend deterministically:

1. **ExecutionPlan steps are executed in order**

   - Builds a plan with a `PrepareFallback` step followed by a
     `MarkTensorsMovable` compute step.
   - After execution, `AdapterContext.executed_steps` is `[0, 1]`.

2. **Unsupported step yields explicit error**

   - Uses a plan where the only step is treated as non-compute and expects an
     adapter-level error (in current tests, mapped via `BackendFailure`).

3. **Abort stops execution safely**

   - Initializes `AdapterContext` with `GuardAction::Abort`.
   - `execute_plan` returns `AdapterError::AbortedByGuard` and produces no
     output.

4. **Adapter respects execution contract and is deterministic**

   - Two identical executions (same plan, model, contract, guard) produce the
     same output tensor.

All tests are CPU-only, use minimal mock plans and models, and are CI-safe:

```bash
cargo test --test execution_adapter_test
```

---

## Status of APX 17.3

APX 17.3 is considered **DONE** when:

- The `ExecutionPlan` fully governs execution order.
- Each supported step is dispatched correctly to the backend.
- Aborts triggered by guards stop execution safely.
- Errors for unsupported steps and backend failures are explicit.
- Tests validate order, determinism, and safety.

The current implementation meets these criteria, completing the APX 17 stack
by cleanly adapting APX 16 orchestration into v17 CPU execution.

---

## APX 17.4 — End-to-End Inference

APX 17.4 exposes a **single public inference entrypoint** that wires together
all previous layers without adding any new intelligence or heuristics.

- Uses `ModelArtifact` (17.0) and `ModelLoader` (17.1).
- Uses `ExecutionContract` and `ExecutionPlan` (16.0–16.1).
- Uses `ExecutionAdapter` (17.3) and `CpuBackend` (17.2).
- Produces feedback and explanations (16.3, 16.6).

### Directory layout additions (17.4)

```text
src/v17/
    inference/
        mod.rs
        infer.rs
        inference_context.rs
        inference_result.rs
        inference_errors.rs
```

Tests for APX 17.4 live in:

```text
tests/end_to_end_inference_test.rs
```

---

## Public API: infer

File: `src/v17/inference/infer.rs`

```rust
pub fn infer(
    artifact: &ModelArtifact,
    input: Tensor,
    preferences: Option<UserPreferences>,
) -> Result<InferenceResult, InferenceError> { /* ... */ }
```

`infer` is the **only** public entrypoint for end-to-end inference in APX 17.4.

Behavior (simplified):

- Validates the `ModelArtifact` (e.g. non-empty id).
- Loads the model from disk into RAM using `ModelLoader` + `LoaderPolicy`.
- Builds a default `ExecutionContract` suitable for CPU inference.
- Uses `ExecutionPlanner` to derive an `ExecutionPlan`.
- Constructs `CpuBackend` and an `AdapterContext`.
- Runs the plan via `ExecutionAdapter::execute_plan` to obtain an output
  `Tensor`.
- Builds feedback events and an `ExecutionOutcome` via `EventEmitter`.
- Builds an explanation via `ExplanationBuilder` and its formatters.
- Assembles everything into an `InferenceResult`.

No new policies, guards, or heuristics are introduced here; `infer` only
composes existing components.

---

## InferenceContext

File: `src/v17/inference/inference_context.rs`

```rust
#[derive(Debug)]
pub struct InferenceContext {
    pub artifact: ModelArtifact,
    pub loaded_model: LoadedModelHandle,
    pub contract: ExecutionContract,
    pub plan: ExecutionPlan,
    pub backend: CpuBackend,
    pub adapter_ctx: AdapterContext,
}
```

`InferenceContext` groups all ephemeral state for a single inference run:

- `artifact` – the validated `ModelArtifact`.
- `loaded_model` – in-memory weights from `ModelLoader`.
- `contract` – `ExecutionContract` under which the plan was created.
- `plan` – `ExecutionPlan` from APX 16.1.
- `backend` – `CpuBackend` used by the adapter.
- `adapter_ctx` – `AdapterContext` tracking step execution.

It is not persisted and is created per invocation of `infer`.

---

## InferenceResult

File: `src/v17/inference/inference_result.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub output: Tensor,
    pub outcome: ExecutionOutcome,
    pub executed_steps: Vec<usize>,
    pub explanation_text: String,
    pub explanation_json: String,
    pub replay_events: Vec<ExecutionEvent>,
    pub replay_outcome: ExecutionOutcome,
}
```

`InferenceResult` contains a structured record of an end-to-end inference:

- `output` – final output tensor from the backend.
- `outcome` – summarized `ExecutionOutcome` (completed, aborted, etc.).
- `executed_steps` – indices of plan steps executed.
- `explanation_text` / `explanation_json` – formatted explanation from APX 16.6.
- `replay_events` / `replay_outcome` – events and outcome that can feed APX
  16.7 replay.

---

## InferenceError

File: `src/v17/inference/inference_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    InvalidArtifact(String),
    LoadFailed(String),
    ContractError(String),
    PlanningError(String),
    AdapterError(String),
    FeedbackError(String),
}
```

`InferenceError` surfaces failures along the end-to-end path:

- `InvalidArtifact` – malformed or incomplete `ModelArtifact`.
- `LoadFailed` – issues in the loader (missing file, size mismatch, OOM).
- `ContractError` – reserved for contract construction failures.
- `PlanningError` – errors from `ExecutionPlanner`.
- `AdapterError` – errors from `ExecutionAdapter`.
- `FeedbackError` – issues while building events or explanations.

All errors are deterministic and descriptive; no panics are used.

---

## Tests for APX 17.4

File: `tests/end_to_end_inference_test.rs`

The test suite validates the full public inference path:

1. **End-to-end inference produces correct output**

   - Uses a tiny 2×2 linear model and a known input vector.
   - Verifies the expected output tensor and that explanation/replay fields are
     populated.

2. **Inference respects execution contract**

   - Under the default contract, successful inference yields a completed or
     partially completed `ExecutionOutcome`.

3. **Abort during inference yields explicit error**

   - Simulated by providing a `ModelArtifact` whose file is missing.
   - `infer` returns `InferenceError::LoadFailed`.

4. **End-to-end inference is deterministic**

   - Two identical calls to `infer` with the same artifact and input produce
     identical `InferenceResult` (output, executed steps, and outcome).

All tests are CPU-only, use minimal mock models, and are CI-safe:

```bash
cargo test --test end_to_end_inference_test
```

---

## Status of APX 17.4

APX 17.4 is considered **DONE** when:

- There is a single public `infer` API for end-to-end inference.
- Loader → planner → adapter → backend work together without bypasses.
- The result includes output, explanation, and replay metadata.
- Aborts and load failures are surfaced as explicit errors.
- Tests validate determinism and correctness end-to-end.

The current implementation meets these criteria, providing a clean, public
inference path that composes all previous APX layers without adding new
decision logic.

---

## APX 17.5 — GPU Backend & ComputeBackend Trait

APX 17.5 introduces a **common compute backend trait** and a `GpuBackend`
implementation. In this initial version, the GPU backend internally delegates
to the CPU backend to preserve determinism and CI-safety, but the abstraction
is ready for real GPU integration behind feature flags.

- CPU remains the default backend.
- GPU is opt-in, safe, and can always fall back to CPU.

### Directory layout additions (17.5)

```text
src/v17/
    compute/
        backend_trait.rs
        gpu_backend.rs
```

Tests for APX 17.5 live in:

```text
tests/gpu_backend_test.rs
```

---

## ComputeBackend trait

File: `src/v17/compute/backend_trait.rs`

```rust
pub trait ComputeBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError>;
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError>;
    fn relu(&self, x: &Tensor) -> Result<Tensor, ComputeError>;
}
```

`ComputeBackend` defines a minimal, closed set of operations that any compute
backend must provide:

- `matmul` – matrix multiplication.
- `add` – elementwise addition.
- `relu` – elementwise ReLU activation.

Both `CpuBackend` and `GpuBackend` implement this trait.

---

## CpuBackend as a ComputeBackend

File: `src/v17/compute/cpu_backend.rs`

`CpuBackend` now implements `ComputeBackend` by directly delegating to the
pure ops defined in `ops.rs`:

- `matmul` → `ops::matmul`.
- `add` → `ops::add`.
- `relu` → `ops::relu`.

This preserves the existing behavior while aligning CPU with the shared
compute contract.

---

## GpuBackend

File: `src/v17/compute/gpu_backend.rs`

```rust
#[derive(Debug, Clone)]
pub struct GpuBackend {
    inner: CpuBackend,
}
```

`GpuBackend` is a minimal GPU backend abstraction that currently uses
`CpuBackend` internally:

- `GpuBackend::new()` constructs an instance wrapping a `CpuBackend`.
- `run_inference(model, input, contract, guard_action)`:
  - Delegates entirely to `CpuBackend::run_inference`.
  - Respects the same `ExecutionContract` and `GuardAction` semantics.

As a `ComputeBackend`:

- `impl ComputeBackend for GpuBackend` delegates `matmul`, `add`, and `relu`
  to the same ops as the CPU backend.

This makes `GpuBackend` fully deterministic and CI-safe today, while providing
a clear surface for future real GPU implementations.

---

## Tests for APX 17.5

File: `tests/gpu_backend_test.rs`

Tests validate that the GPU backend behaves like an interchangeable
implementation of `ComputeBackend` and that CPU fallback is safe:

1. **GPU backend implements ComputeBackend and matches CPU output**

   - Compares `add` and `relu` results between `CpuBackend` and
     `GpuBackend` and asserts equality.

2. **GPU and CPU produce equivalent matmul output**

   - Runs `matmul` on both backends with the same inputs and checks that the
     output tensors are identical.

3. **GPU backend respects execution contract via run_inference**

   - Uses `make_contract(true/false)`:
     - Stable contract → `run_inference` succeeds.
     - Unstable contract → `ComputeError::ContractViolation`.
   - Repeated runs under identical conditions yield identical outputs.

4. **GPU fallback to CPU is safe**

   - Constructs a dummy model and input.
   - Compares `GpuBackend::run_inference` and `CpuBackend::run_inference`.
   - Since GPU currently delegates to CPU, outputs are exactly equal,
     demonstrating safe fallback.

All tests are CPU-only, do not require a real GPU, and are CI-friendly:

```bash
cargo test --test gpu_backend_test
```

---

## Status of APX 17.5

APX 17.5 is considered **DONE** when:

- `ComputeBackend` provides a shared contract for compute backends.
- `GpuBackend` implements this contract and can be swapped with `CpuBackend`.
- CPU remains the default and GPU is opt-in.
- Fallback to CPU is safe and deterministic.
- Tests confirm equivalence of CPU and GPU outputs and respect for
  contracts and guards.

The current implementation meets these criteria while remaining fully
deterministic and CI-safe, paving the way for real GPU integration under
strict safety and observability guarantees.

---

## APX 17.6 — Execution Profiling & Metrics

APX 17.6 adds **quantitative observability** to end-to-end inference without
changing behavior, decisions, or timing. It records only logical metrics:

- Steps executed (and in which order).
- Backend kind used (CPU/GPU).
- Counts of basic operations (matmul, add, relu).
- Logical volume of data processed.

No wall-clock time, OS clocks, sleeps, or IO are involved.

### Directory layout additions (17.6)

```text
src/v17/
    profiling/
        mod.rs
        execution_profiler.rs
        step_metrics.rs
        backend_metrics.rs
        profiling_errors.rs
```

Tests for APX 17.6 live in:

```text
tests/execution_profiling_test.rs
```

---

## StepMetrics

File: `src/v17/profiling/step_metrics.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct StepMetrics {
    pub step_index: usize,
    pub kind: String,
    pub backend: String,
    pub input_elements: usize,
    pub output_elements: usize,
    pub aborted: bool,
    pub fallback: bool,
}
```

`StepMetrics` captures per-step logical information:

- `step_index` – index into the `ExecutionPlan` steps.
- `kind` – textual representation of the step kind.
- `backend` – string identifying the backend ("cpu" / "gpu").
- `input_elements` / `output_elements` – element counts of tensors.
- `aborted` / `fallback` – reserved flags for future use.

---

## BackendMetrics & ExecutionProfile

File: `src/v17/profiling/backend_metrics.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendMetrics {
    pub backend: BackendKind,
    pub matmul_count: usize,
    pub add_count: usize,
    pub relu_count: usize,
    pub elements_processed: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionProfile {
    pub steps: Vec<StepMetrics>,
    pub backends: Vec<BackendMetrics>,
}
```

`BackendMetrics` aggregates metrics per backend kind. In APX 17.6 it tracks:

- Operation counts (`matmul_count`, `add_count`, `relu_count`).
- `elements_processed` – simple sum of input and output element counts.

`ExecutionProfile` is a stable, serializable container of profiling data.
It provides a `to_json()` method to output a JSON-like string with a fixed
field order, without any external dependencies.

---

## ExecutionProfiler

File: `src/v17/profiling/execution_profiler.rs`

```rust
pub struct ExecutionProfiler;

impl ExecutionProfiler {
    pub fn profile(
        plan: &ExecutionPlan,
        executed_steps: &[usize],
        backend: BackendKind,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<ExecutionProfile, ProfilingError> { /* ... */ }
}
```

`ExecutionProfiler::profile`:

- Requires a non-empty `executed_steps` list; otherwise returns
  `ProfilingError::MissingSteps`.
- Builds `StepMetrics` for each executed index, preserving order.
- Builds a single `BackendMetrics` instance reflecting a simple model
  (currently assuming one matmul + one relu per execution as a starting
  invariant).
- Returns an `ExecutionProfile` independent of wall-clock time or hardware.

The function is pure and cannot mutate execution behavior.

---

## Integration with InferenceResult

File: `src/v17/inference/inference_result.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub output: Tensor,
    pub outcome: ExecutionOutcome,
    pub executed_steps: Vec<usize>,
    pub explanation_text: String,
    pub explanation_json: String,
    pub replay_events: Vec<ExecutionEvent>,
    pub replay_outcome: ExecutionOutcome,
    pub profile: Option<ExecutionProfile>,
}
```

`InferenceResult` now optionally includes an `ExecutionProfile`:

- Profiling is built after inference completes using `ExecutionProfiler`.
- If profiling fails (e.g. missing steps), a default empty profile is
  attached; inference output and outcome remain unchanged.

This ensures profiling can be disabled or changed without affecting
inference behavior.

---

## Tests for APX 17.6

File: `tests/execution_profiling_test.rs`

The test suite validates profiling behavior:

1. **Profiling collects step metrics in execution order**

   - Given a simple plan and executed step indices, `ExecutionProfile.steps`
     contains the same indices in order.

2. **Profiling is deterministic for identical inference**

   - Two calls to `ExecutionProfiler::profile` with identical inputs produce
     identical `ExecutionProfile` values and identical JSON via `to_json()`.

3. **Profiling does not affect execution outcome**

   - Profiling is run independently of execution and cannot cause failures for
     valid inputs.

4. **Profiling reflects backend selection and is optional**

   - Profiles built with `BackendKind::Cpu` vs `BackendKind::Gpu` differ in
     JSON representation, making backend choice observable.

5. **Profiling errors on missing steps**

   - Empty `executed_steps` yields `ProfilingError::MissingSteps`.

End-to-end tests in `end_to_end_inference_test.rs` additionally assert that
the `InferenceResult.profile` field is present and deterministic across
identical inferences.

---

## Status of APX 17.6

APX 17.6 is considered **DONE** when:

- A stable `ExecutionProfile` exists and can be attached to inference
  results.
- Profiles reflect executed steps and backend selection.
- Profiling has no effect on execution decisions or outcomes.
- Profiling is fully deterministic and CI-safe.
- Tests validate order, determinism, and non-interference with inference.

The current implementation meets these criteria, providing quantitative,
deterministic observability on top of the APX 17.x inference pipeline.

---

## APX 17.7 — Execution Snapshot & Persistence

APX 17.7 introduces **execution snapshots**: immutable, hashable summaries of
completed inferences. Snapshots capture what happened without re-running or
modifying the execution.

- No inference is executed in this layer.
- No plans, contracts, or profiles are modified.
- Snapshots are derived from existing `InferenceResult` and plan/contract
  information.

### Directory layout additions (17.7)

```text
src/v17/
    snapshot/
        mod.rs
        execution_snapshot.rs
        snapshot_builder.rs
        snapshot_hash.rs
        snapshot_errors.rs
```

Tests for APX 17.7 live in:

```text
tests/execution_snapshot_test.rs
```

---

## ExecutionSnapshot

File: `src/v17/snapshot/execution_snapshot.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionSnapshot {
    pub model_id: String,
    pub contract_fingerprint: String,
    pub plan_fingerprint: String,
    pub backend_usage: String,
    pub profile_hash: String,
    pub output_signature: String,
    pub explanation_signature: String,
    pub snapshot_hash: String,
}
```

`ExecutionSnapshot` contains only **hashes and metadata**, not full tensors:

- `model_id` – a stable identifier derived from model/contract metadata.
- `contract_fingerprint` – hash of the `ExecutionContract`.
- `plan_fingerprint` – hash of the `ExecutionPlan`.
- `backend_usage` – summarizes whether CPU/GPU backends were used.
- `profile_hash` – hash of the `ExecutionProfile` JSON.
- `output_signature` – hash of output shape and data.
- `explanation_signature` – hash of explanation text/JSON.
- `snapshot_hash` – overall hash combining the above.

`to_json()` provides a stable JSON-like representation for storage or audit.

---

## Snapshot hash utilities

File: `src/v17/snapshot/snapshot_hash.rs`

```rust
pub fn hash_bytes(data: &[u8]) -> String { /* FNV-1a 64-bit */ }
pub fn hash_str(s: &str) -> String { /* ... */ }
```

Hashing is implemented using a fixed FNV-1a 64-bit algorithm encoded as
lowercase hex:

- Deterministic across machines and runs.
- Independent of OS or hardware.
- Suitable for producing stable fingerprints.

---

## SnapshotBuilder

File: `src/v17/snapshot/snapshot_builder.rs`

```rust
pub struct SnapshotBuilder;

impl SnapshotBuilder {
    pub fn build(
        result: &InferenceResult,
        contract: &ExecutionContract,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionSnapshot, SnapshotError> { /* ... */ }
}
```

`SnapshotBuilder::build` constructs an `ExecutionSnapshot` from:

- An `InferenceResult` (output, explanation, profile).
- The `ExecutionContract` and `ExecutionPlan` used.

Validation rules:

- Fails with `SnapshotError::MissingProfile` if `InferenceResult.profile` is
  `None`.
- Fails with `SnapshotError::MissingExplanation` if explanation text/JSON are
  empty.
- Fails with `SnapshotError::IncompleteExecution` if no steps were executed or
  the outcome kind is `Failed`.

When inputs are valid, it computes fingerprints and the final `snapshot_hash`
using the deterministic hash helpers.

---

## Snapshot errors

File: `src/v17/snapshot/snapshot_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    MissingProfile(String),
    MissingExplanation(String),
    IncompleteExecution(String),
}
```

`SnapshotError` provides explicit failure modes when a snapshot cannot be
constructed safely.

---

## Tests for APX 17.7

File: `tests/execution_snapshot_test.rs`

Tests ensure that snapshots are deterministic, comparable, and reject
incomplete data:

1. **Snapshot is built from valid inference result**

   - Builds a synthetic `InferenceResult` with explanation and profile.
   - `SnapshotBuilder::build` succeeds and `to_json()` contains expected
     fields.

2. **Snapshot hash is deterministic**

   - Two builds with identical inputs produce the same `snapshot_hash` and
     JSON.

3. **Identical executions yield identical snapshots**

   - Two separate `InferenceResult` instances with the same data produce equal
     `ExecutionSnapshot`s.

4. **Snapshot rejects incomplete execution data**

   - Clearing `executed_steps` causes `IncompleteExecution` error.

5. **Snapshot rejects missing profile or explanation**

   - `profile = None` → `MissingProfile`.
   - Empty explanation → `MissingExplanation`.

All tests are pure, avoid IO and clocks, and are CI-safe.

---

## Status of APX 17.7

APX 17.7 is considered **DONE** when:

- An immutable `ExecutionSnapshot` can be built from complete inference
  evidence.
- Snapshot hashes are deterministic and stable.
- Construction fails when required evidence is missing or incomplete.
- Snapshots are comparable (via `PartialEq`) and serializable (via `to_json`).
- Tests validate determinism, completeness checks, and non-interference with
  execution.

The current implementation meets these criteria, turning each inference into a
sealed, hashable unit of evidence that APX 18+ layers can safely consume for
learning, audit, and analysis.

---

## APX 17.8 — Execution Consistency & Regression Guard

APX 17.8 provides a **behavioral regression guard** that compares current
execution snapshots against historical baselines, without executing or
modifying any inference.

- Works purely on `ExecutionSnapshot`s (17.7).
- Classifies differences as compatible, minor drift, or critical drift.
- Intended for CI, release validation, and enterprise certification.

### Directory layout additions (17.8)

```text
src/v17/
    consistency/
        mod.rs
        execution_baseline.rs
        consistency_checker.rs
        drift_report.rs
        consistency_errors.rs
```

Tests for APX 17.8 live in:

```text
tests/execution_consistency_test.rs
```

---

## ExecutionBaseline

File: `src/v17/consistency/execution_baseline.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBaseline {
    pub reference_snapshot: ExecutionSnapshot,
    pub allow_backend_change: bool,
}
```

`ExecutionBaseline` describes the expected behavior for a scenario:

- `reference_snapshot` – canonical `ExecutionSnapshot` to compare against.
- `allow_backend_change` – whether CPU↔GPU backend changes are acceptable if
  other behavior remains consistent.

The baseline is immutable; it does not execute or modify anything.

---

## DriftReport & DriftSeverity

File: `src/v17/consistency/drift_report.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum DriftSeverity {
    Compatible,
    MinorDrift,
    CriticalDrift,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftReport {
    pub severity: DriftSeverity,
    pub differences: Vec<String>,
    pub change_fingerprint: String,
}
```

`DriftReport` is a structured, serializable summary of differences between a
baseline and a current snapshot:

- `severity` – high-level classification.
- `differences` – human-readable descriptions of observed changes.
- `change_fingerprint` – deterministic hash of the differences.

`to_json()` provides a stable JSON-like encoding for CI and audit pipelines.

---

## ConsistencyChecker

File: `src/v17/consistency/consistency_checker.rs`

```rust
pub struct ConsistencyChecker;

impl ConsistencyChecker {
    pub fn compare(
        baseline: &ExecutionBaseline,
        current: &ExecutionSnapshot,
    ) -> Result<DriftReport, ConsistencyError> { /* ... */ }
}
```

`ConsistencyChecker::compare`:

- Validates the baseline (e.g. non-empty `model_id`).
- If `snapshot_hash` matches, returns a `Compatible` report with no
  differences.
- Otherwise, compares:
  - `backend_usage` with respect to `allow_backend_change`.
  - `output_signature` (critical if different).
  - `profile_hash` (minor drift if only profiling changed).
  - `contract_fingerprint` and `plan_fingerprint` (critical if changed).
- Builds a `DriftReport` with appropriate severity and a deterministic
  `change_fingerprint`.

The checker is pure and never alters executions; it only analyzes already
sealed snapshots.

---

## ConsistencyError

File: `src/v17/consistency/consistency_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyError {
    InvalidBaseline(String),
    IncompatibleSnapshots(String),
}
```

`ConsistencyError` captures invalid baselines or snapshot pairs that cannot be
meaningfully compared.

---

## Tests for APX 17.8

File: `tests/execution_consistency_test.rs`

The test suite exercises the regression guard behavior:

1. **Identical snapshots yield compatible consistency**

   - Baseline and current snapshots are identical → `Compatible` with no
     differences.

2. **Allowed backend change yields non-critical drift**

   - CPU→GPU backend change with `allow_backend_change = true` → `MinorDrift`
     with a backend-change difference.

3. **Unexpected execution change yields critical drift**

   - Output signature differences or contract/plan fingerprint changes →
     `CriticalDrift`.

4. **Consistency checker is deterministic**

   - Repeated comparisons with the same inputs yield identical `DriftReport`
     values and identical JSON.

5. **Drift report is stable and serializable**

   - `to_json()` returns a stable string across calls.

6. **Invalid baseline yields error**

   - Baseline with empty `model_id` → `ConsistencyError::InvalidBaseline`.

All tests operate purely on mock snapshots, without running real inference or
using clocks or IO.

---

## Status of APX 17.8

APX 17.8 is considered **DONE** when:

- Baseline vs current snapshot comparisons are supported.
- Behavioral drift is classified as compatible, minor, or critical.
- No execution behavior is altered by the consistency checks.
- Drift reports are deterministic and serializable.
- Tests validate classification, determinism, and error conditions.

The current implementation meets these criteria, providing a non-invasive,
deterministic regression guard that can be used in CI and enterprise flows to
catch behavioral changes before deployment.

---

## APX 17.9 — Engine Capability Manifest & Version Seal

APX 17.9 introduces a **static engine manifest** that describes what the
runtime is capable of, without executing, observing, or learning from
executions.

- Declares enabled backends and execution modes.
- States which safety and observability features are present.
- Provides a deterministic version seal for certification and comparison.

### Directory layout additions (17.9)

```text
src/v17/
    manifest/
        mod.rs
        engine_manifest.rs
        capability_descriptor.rs
        version_seal.rs
        manifest_errors.rs
```

Tests for APX 17.9 live in:

```text
tests/engine_manifest_test.rs
```

---

## EngineManifest

File: `src/v17/manifest/engine_manifest.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct EngineManifest {
    pub engine_version: String,
    pub enabled_backends: Vec<String>,
    pub supported_execution_modes: Vec<String>,
    pub profiling_level: String,
    pub snapshot_support: bool,
    pub consistency_guard_support: bool,
    pub learning_enabled: bool,
    pub capabilities: CapabilityDescriptor,
}
```

`EngineManifest` describes the APX 17 runtime as a contract of capabilities:

- `engine_version` – logical version (e.g. "17.x").
- `enabled_backends` – backends available (CPU, GPU).
- `supported_execution_modes` – e.g. "contracted", "speculative".
- `profiling_level` – scope of profiling (logical only in APX 17).
- `snapshot_support` – whether execution snapshotting (17.7) is available.
- `consistency_guard_support` – whether regression guard (17.8) is present.
- `learning_enabled` – `false` for APX 17.
- `capabilities` – detailed booleans via `CapabilityDescriptor`.

`apx17_default()` constructs the canonical manifest for this engine, and
`to_json()` provides a stable, IO-free serialization.

---

## CapabilityDescriptor

File: `src/v17/manifest/capability_descriptor.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityDescriptor {
    pub supports_gpu_execution: bool,
    pub supports_replay: bool,
    pub supports_abortability: bool,
    pub supports_determinism: bool,
    pub supports_snapshot_sealing: bool,
}
```

Each flag states explicitly whether a capability is present, making the
runtime's behavior auditable and comparable across builds.

---

## VersionSeal

File: `src/v17/manifest/version_seal.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct VersionSeal {
    pub manifest_hash: String,
}

impl VersionSeal {
    pub fn from_manifest(manifest: &EngineManifest) -> Result<Self, ManifestError> { /* ... */ }
}
```

`VersionSeal` is a deterministic fingerprint of an `EngineManifest`:

- Validates that `engine_version` is non-empty.
- Requires `learning_enabled == false` for APX 17.
- Hashes `EngineManifest::to_json()` using the same FNV-1a-based helper as
  snapshots, producing a stable `manifest_hash` string.

Two engines with the same manifest → same version seal.

---

## ManifestError

File: `src/v17/manifest/manifest_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ManifestError {
    InvalidVersion(String),
    InconsistentCapabilities(String),
}
```

`ManifestError` surfaces invalid manifests (e.g. empty version) or
capability combinations that contradict APX 17 guarantees (e.g. learning
enabled).

---

## Tests for APX 17.9

File: `tests/engine_manifest_test.rs`

Tests validate the manifest and version seal behavior:

1. **Engine manifest is constructed correctly**

   - `apx17_default()` yields expected version, backends, and flags.

2. **Manifest reflects enabled backends and features**

   - JSON representation includes CPU/GPU and key fields.

3. **Version seal is deterministic**

   - Same manifest → identical `VersionSeal` values.

4. **Identical manifests yield identical seals**

   - Separate but equal manifests yield seals with the same `manifest_hash`.

5. **Invalid manifest yields explicit error**

   - Empty `engine_version` → `InvalidVersion`.
   - `learning_enabled = true` → `InconsistentCapabilities`.

All tests are pure, require no IO, and are CI-safe.

---

## Status of APX 17.9

APX 17.9 is considered **DONE** when:

- A stable `EngineManifest` exists and accurately reflects engine
  capabilities.
- A deterministic `VersionSeal` can be derived from the manifest.
- Invalid or inconsistent manifests produce explicit errors.
- The manifest and seal do not affect runtime behavior.
- Tests confirm determinism and correctness of the manifest and seal.

The current implementation meets these criteria, giving Atenia a clear,
auditable runtime identity that future APX 18+ layers and external consumers
can rely on for certification and comparison.

---

# APX 17.10.0 — Conv2D Kernel (CPU)

APX 17.10.0 introduces a **minimal, deterministic and auditable Conv2D CPU
operator** suitable for small CNNs (e.g. MNIST-scale) under the existing
Execution Contract. This layer focuses **only** on correctness and
traceability:

- No GPU.
- No SIMD or multithreading.
- No fused ops, autotuning or heuristics.

All behavior is local, purely functional and CI-safe.

## Module layout

Files introduced by APX 17.10.0 live under `src/v17/cnn`:

```text
src/
  v17/
    cnn/
      mod.rs
      conv2d.rs
```

The `mod.rs` file contains the WS disclaimer and simply exposes the
`conv2d` module:

```rust
pub mod conv2d;
```

`conv2d.rs` contains the Conv2D parameters, error types, abort flag and the
CPU implementation itself.

## API

File: `src/v17/cnn/conv2d.rs`

```rust
use crate::v17::compute::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct Conv2DParams {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvError {
    InvalidInputShape,
    InvalidWeightShape,
    KernelLargerThanInput,
    InvalidPadding,
    InvalidStride,
    InvalidBiasShape,
    Aborted,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AbortFlag {
    aborted: bool,
}

impl AbortFlag {
    pub fn new() -> Self { /* ... */ }
    pub fn abort(&mut self) { /* ... */ }
    pub fn is_aborted(&self) -> bool { /* ... */ }
}

pub fn conv2d_cpu(
    input: &Tensor,
    weights: &Tensor,
    bias: Option<&Tensor>,
    params: &Conv2DParams,
    abort_flag: &AbortFlag,
) -> Result<Tensor, ConvError> { /* ... */ }
```

### Layouts

`conv2d_cpu` assumes fixed layouts:

- `input`  – NCHW: `[n, c_in, h_in, w_in]`.
- `weights` – OIHW: `[c_out, c_in, k_h, k_w]`.
- `bias`   – `[c_out]` (optional).

The result is also NCHW: `[n, c_out, h_out, w_out]`.

### Behavior and guarantees

- **No mutation of inputs**: `input` and `weights` are only read.
- **Local allocation only**: output is allocated in the scope of
  `conv2d_cpu` using the public `Tensor { shape, data }` fields.
- **No side effects**: no IO, logging, global state or hardware access.
- **Deterministic**:
  - Pure arithmetic over `f32` with a fixed nested-loop order.
  - Given the same tensors, parameters and abort flag state, the output
    tensor is bitwise identical.
- **Abortable**:
  - Checks `abort_flag.is_aborted()` at entry and in each outer loop
    (`n`, `c_out`, `h_out`).
  - Returns `Err(ConvError::Aborted)` without panics.
- **Shape and parameter validation**:
  - `input.shape.len() != 4` → `InvalidInputShape`.
  - `weights.shape.len() != 4` or mismatched `c_in` → `InvalidWeightShape`.
  - `stride_h == 0 || stride_w == 0` → `InvalidStride`.
  - `k_h == 0 || k_w == 0` → `KernelLargerThanInput`.
  - `h_in + 2 * pad_h < k_h` or `w_in + 2 * pad_w < k_w` →
    `KernelLargerThanInput`.
  - Bias tensor present but not `shape == [c_out]` → `InvalidBiasShape`.

The implementation uses explicit index helpers for NCHW/OIHW to keep data
access auditable and free from hidden abstractions.

## Tests for APX 17.10.0

File: `tests/conv2d_test.rs`

The test module imports `v17`, `v16` and `v15` via `#[path = "../src/.../mod.rs"]`
to allow reuse of existing APX layers when linking, and then validates the
Conv2D behavior via four core tests:

1. **conv2d_simple_output_matches_reference**

   - Runs `conv2d_cpu` on a small hand-crafted input and kernel.
   - Compares the result against a reference call using the same operator
     with a clean `AbortFlag`.
   - Verifies equal shapes and numerically identical values (within a very
     small epsilon).

2. **conv2d_is_deterministic**

   - Calls `conv2d_cpu` twice on the same input and weights.
   - Asserts identical `shape` and identical `data` vectors.

3. **conv2d_is_abortable**

   - Creates an `AbortFlag`, immediately calls `.abort()`, then invokes
     `conv2d_cpu`.
   - Expects `Err(ConvError::Aborted)` without panics or partial writes.

4. **invalid_shapes_yield_explicit_error**

   - Input tensor that is not 4D → `InvalidInputShape`.
   - Weights tensor that is not OIHW → `InvalidWeightShape`.
   - Zero stride → `InvalidStride`.
   - Kernel larger than the padded input → `KernelLargerThanInput`.
   - Bias tensor with mismatched shape → `InvalidBiasShape`.

All Conv2D tests are deterministic, require no IO or external hardware, and
are safe to run in CI.

## Status of APX 17.10.0

APX 17.10.0 is considered **DONE** when:

- `conv2d_cpu` is implemented under `src/v17/cnn/conv2d.rs`.
- All four Conv2D tests in `tests/conv2d_test.rs` pass without panics or
  warnings.
- The operator is deterministic and abortable.
- Shape and parameter errors are surfaced via `ConvError` (no `panic!`).
- No existing execution logic outside the CNN module is modified.

The current implementation satisfies these criteria and provides a minimal,
auditable Conv2D CPU primitive on top of the APX 17 stack. Future APX 17.10.x
and 18.x work (e.g. MNIST end-to-end, batching or GPU kernels) can safely
build on this foundation without changing its semantics.

---

# APX 17.10.1 — Bias + Activation (ReLU)

APX 17.10.1 extends the CNN operator stack with **explicit, standalone Bias
addition and ReLU activation** operators. These are designed to be:

- Pure functions.
- Deterministic.
- Abortable.
- Auditable and reusable outside CNN pipelines.

No fusion, optimization or in-place mutation is introduced; Conv2D remains
unchanged and all steps stay visible and explainable.

## Module layout

APX 17.10.1 adds two files under `src/v17/cnn`:

```text
src/
  v17/
    cnn/
      bias.rs
      activation.rs
```

They are re-exported via `src/v17/cnn/mod.rs` alongside `conv2d`:

```rust
pub mod conv2d;
pub mod bias;
pub mod activation;
```

## Bias addition API

File: `src/v17/cnn/bias.rs`

```rust
use crate::v17::compute::tensor::Tensor;
use crate::v17::cnn::conv2d::AbortFlag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BiasError {
    InvalidInputShape,
    InvalidBiasShape,
    Aborted,
}

pub fn add_bias(
    input: &Tensor,      // NCHW [N, C, H, W]
    bias: &Tensor,       // [C]
    abort_flag: &AbortFlag,
) -> Result<Tensor, BiasError> { /* ... */ }
```

### Behavior and guarantees (Bias)

- **Layouts**:
  - `input` is NCHW: `[N, C, H, W]`.
  - `bias` is `[C]` and is broadcast over `H×W`.
- **Validation**:
  - `input.shape.len() != 4` → `BiasError::InvalidInputShape`.
  - `bias.shape` not equal to `[C]` → `BiasError::InvalidBiasShape`.
- **Deterministic and pure**:
  - Clones `input.data` into a local `out_data` buffer.
  - Adds the bias per channel with a fixed nested-loop order.
  - Returns a new `Tensor { shape, data }` without mutating `input`.
- **Abortable**:
  - Checks `abort_flag.is_aborted()` at entry and in the outer loops
    (`N`, `C`).
  - Returns `Err(BiasError::Aborted)` without panics.
- **No side effects**:
  - No global state, logging, IO or hardware access.

## ReLU activation API

File: `src/v17/cnn/activation.rs`

```rust
use crate::v17::compute::tensor::Tensor;
use crate::v17::cnn::conv2d::AbortFlag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    InvalidInputShape,
    Aborted,
}

pub fn relu(
    input: &Tensor,
    abort_flag: &AbortFlag,
) -> Result<Tensor, ActivationError> { /* ... */ }
```

### Behavior and guarantees (ReLU)

- **Elementwise max**:
  - For each element `x` in `input.data`, computes `max(x, 0.0)`.
  - Preserves the tensor `shape` exactly.
- **Validation**:
  - The implementation accepts any rank; `InvalidInputShape` is reserved for
    future stricter contracts.
- **Deterministic and pure**:
  - Clones `input.data` into a local buffer and applies ReLU in place on the
    clone.
  - Returns a new tensor; `input` is never mutated.
- **Abortable**:
  - Checks `abort_flag.is_aborted()` at entry and within the main loop.
  - Returns `Err(ActivationError::Aborted)` without panics.
- **No side effects**:
  - No global state, logging, IO or hardware access.

## Tests for APX 17.10.1

Bias and ReLU each have a dedicated test module under `tests/`:

```text
tests/bias_test.rs
tests/relu_test.rs
```

Both import `v17`, `v16` and `v15` via `#[path = "../src/.../mod.rs"]` to
reuse the existing APX stack when linking.

### Bias tests (`tests/bias_test.rs`)

1. **bias_add_matches_reference**

   - Applies `add_bias` to a small NCHW tensor.
   - Independently recomputes the per-channel bias addition with explicit
     loops.
   - Asserts identical shapes and data.

2. **bias_is_deterministic**

   - Calls `add_bias` twice on the same input and bias.
   - Asserts identical `shape` and `data`.

3. **bias_is_abortable**

   - Uses an `AbortFlag` that is pre-aborted before calling `add_bias`.
   - Expects `Err(BiasError::Aborted)`.

4. **invalid_bias_shape_yields_error**

   - Constructs an input `[N, C, H, W]` with `C != bias.len()`.
   - Expects `Err(BiasError::InvalidBiasShape)`.

### ReLU tests (`tests/relu_test.rs`)

1. **relu_zeroes_negative_values**

   - Runs `relu` on a tensor containing negative and positive values.
   - Asserts that negatives become `0.0` and positives are preserved.

2. **relu_is_deterministic**

   - Calls `relu` twice on the same input.
   - Asserts identical `shape` and `data`.

3. **relu_is_abortable**

   - Uses an `AbortFlag` that is pre-aborted.
   - Expects `Err(ActivationError::Aborted)`.

4. **relu_has_no_side_effects**

   - Clones the input tensor, calls `relu`, and verifies the original input
     remains unchanged.

All tests are deterministic, do not use randomness or hardware, and are safe
to run in CI.

## Status of APX 17.10.1

APX 17.10.1 is considered **DONE** when:

- `add_bias` and `relu` are implemented under `src/v17/cnn/`.
- `cargo test --test bias_test` and `cargo test --test relu_test` both pass
  without panics or warnings.
- Bias and ReLU are deterministic, abortable and do not mutate their inputs.
- No fusion or modification of the existing Conv2D implementation occurs.
- No side effects or global state are introduced.

The current implementation meets these criteria and adds explicit, auditable
post-convolution steps on top of APX 17.10.0, ready to be composed into
higher-level CNN flows in later APX versions without changing semantics.

---

# APX 17.10.2 — MaxPool2D (CPU)

APX 17.10.2 introduces a **minimal, explicit and deterministic MaxPool2D CPU
operator** for spatial downsampling in small CNNs (e.g. MNIST-scale). This
operator is a pure semantic transformation:

- No fusion with Conv2D, Bias or ReLU.
- No parallelism or low-level optimization.
- No hardware access or global state.

MaxPool2D is fully isolated under the CNN module and remains visible and
explainable in any execution trace.

## Module layout

File added under `src/v17/cnn`:

```text
src/
  v17/
    cnn/
      maxpool2d.rs
```

`src/v17/cnn/mod.rs` re-exports it alongside the other CNN operators:

```rust
pub mod conv2d;
pub mod bias;
pub mod activation;
pub mod maxpool2d;
```

## API

File: `src/v17/cnn/maxpool2d.rs`

```rust
use crate::v17::compute::tensor::Tensor;
use crate::v17::cnn::conv2d::AbortFlag;

#[derive(Debug, Clone, Copy)]
pub struct MaxPool2DParams {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxPoolError {
    InvalidInputShape,
    InvalidKernel,
    InvalidStride,
    KernelLargerThanInput,
    Aborted,
}

pub fn maxpool2d_cpu(
    input: &Tensor,          // NCHW [N, C, H, W]
    params: &MaxPool2DParams,
    abort_flag: &AbortFlag,
) -> Result<Tensor, MaxPoolError> { /* ... */ }
```

### Behavior and guarantees

- **Layout**:
  - `input` is NCHW: `[N, C, H_in, W_in]`.
  - Result is NCHW: `[N, C, H_out, W_out]`.
- **Parameters**:
  - `kernel = (k_h, k_w)` must be strictly positive.
  - `stride = (s_h, s_w)` must be strictly positive.
  - `padding = (p_h, p_w)` is applied symmetrically on all sides.
- **Validation**:
  - `input.shape.len() != 4` → `MaxPoolError::InvalidInputShape`.
  - `k_h == 0 || k_w == 0` → `MaxPoolError::InvalidKernel`.
  - `s_h == 0 || s_w == 0` → `MaxPoolError::InvalidStride`.
  - `H_in + 2*p_h < k_h` or `W_in + 2*p_w < k_w` →
    `MaxPoolError::KernelLargerThanInput`.
- **Deterministic and pure**:
  - Reads `input.data` and computes window maxima in a fixed nested-loop
    order.
  - Allocates a fresh `out_data` buffer and returns a new tensor; `input` is
    never mutated.
- **Abortable**:
  - Checks `abort_flag.is_aborted()` at entry and in the outer loops
    (`N`, `C`, `H_out`).
  - Returns `Err(MaxPoolError::Aborted)` without panics or partial writes.
- **No side effects**:
  - No logging, IO, global state or hardware access.

## Tests for APX 17.10.2

File: `tests/maxpool2d_test.rs`

The test module imports `v17`, `v16` and `v15` in the same way as other
APX 17.x tests and validates the operator via four core tests:

1. **maxpool2d_matches_reference**

   - Applies `maxpool2d_cpu` to a small 1×1×4×4 tensor with a 2×2 kernel,
     stride 2 and no padding.
   - Asserts the output shape `[1, 1, 2, 2]` and the expected values
     `[6, 8, 14, 16]`.

2. **maxpool2d_is_deterministic**

   - Calls `maxpool2d_cpu` twice on the same input and parameters.
   - Asserts identical `shape` and `data`.

3. **maxpool2d_is_abortable**

   - Uses an `AbortFlag` that is pre-aborted before invoking
     `maxpool2d_cpu`.
   - Expects `Err(MaxPoolError::Aborted)`.

4. **invalid_params_yield_explicit_error**

   - Input tensor that is not 4D → `InvalidInputShape`.
   - Zero kernel dimension → `InvalidKernel`.
   - Zero stride dimension → `InvalidStride`.
   - Kernel larger than the padded input → `KernelLargerThanInput`.

All tests are pure, deterministic and CI-safe, with no randomness or
hardware dependencies.

## Status of APX 17.10.2

APX 17.10.2 is considered **DONE** when:

- `maxpool2d_cpu` and `MaxPool2DParams` are implemented under
  `src/v17/cnn/maxpool2d.rs`.
- `cargo test --test maxpool2d_test` passes without panics or warnings.
- The operator is deterministic, abortable and does not mutate its input.
- Errors are surfaced via `MaxPoolError` (no `panic!`).
- Conv2D, Bias and Activation implementations remain unchanged.

The current implementation satisfies these criteria and adds a clear,
auditable spatial downsampling primitive on top of the APX 17.10.x CNN
stack, ready to be composed into higher-level architectures without changing
their semantics.

---

# APX 17.10.3 — CNN Execution Adapter

APX 17.10.3 introduces a **CNN Execution Adapter** that maps high-level CNN
layers (Conv2D, Bias, ReLU, MaxPool2D) to an explicit `CNNExecutionPlan`.
This layer does **not** execute kernels; it only orchestrates the existing
operators as ordered, abortable steps.

- No fusion or reordering of operators.
- No changes to Conv2D, Bias, ReLU or MaxPool2D implementations.
- No hardware access or global state.

## Module layout

File added under `src/v17/cnn`:

```text
src/
  v17/
    cnn/
      cnn_adapter.rs
```

`src/v17/cnn/mod.rs` re-exports the adapter:

```rust
pub mod cnn_adapter;
```

## API

File: `src/v17/cnn/cnn_adapter.rs`

```rust
use crate::v17::cnn::conv2d::AbortFlag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CNNAdapterError {
    InvalidGraph(String),
    Aborted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNLayer {
    pub name: String,
    pub kind: CNNLayerKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CNNLayerKind {
    Conv2D,
    Bias,
    ReLU,
    MaxPool2D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNGraph {
    pub layers: Vec<CNNLayer>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CNNPlanStepKind {
    Conv2D,
    Bias,
    ReLU,
    MaxPool2D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNPlanStep {
    pub kind: CNNPlanStepKind,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNExecutionPlan {
    pub steps: Vec<CNNPlanStep>,
}

pub struct CNNExecutionAdapter;

impl CNNExecutionAdapter {
    pub fn build_plan(
        graph: &CNNGraph,
        abort_flag: &AbortFlag,
    ) -> Result<CNNExecutionPlan, CNNAdapterError> { /* ... */ }
}
```

### Behavior and guarantees

- **Explicit graph**:
  - `CNNGraph` is an explicit, ordered list of CNN layers.
  - The adapter does not infer, reorder or fuse layers.
- **Validation**:
  - Empty graphs → `CNNAdapterError::InvalidGraph`.
  - `Bias`, `ReLU` and `MaxPool2D` must conceptually follow `Conv2D`.
  - In the minimal adapter, multiple `Conv2D` layers are rejected as
    `InvalidGraph`.
  - `Bias`/`ReLU` cannot appear after `MaxPool2D`.
- **Deterministic and pure**:
  - Builds a `CNNExecutionPlan` by mapping each `CNNLayer` 1:1 into a
    `CNNPlanStep` in the same order.
  - Each step carries a stable `kind` and `description` string; no kernel
    execution is performed.
- **Abortable**:
  - Checks `abort_flag.is_aborted()` at entry and while iterating over
    layers.
  - Returns `Err(CNNAdapterError::Aborted)` without side effects.
- **No side effects**:
  - No IO, logging, hardware calls or global state.

## Tests for APX 17.10.3

File: `tests/cnn_adapter_test.rs`

The test module imports `v17`, `v16` and `v15` in the usual way and verifies
the adapter via five tests:

1. **cnn_plan_is_built_with_expected_steps**

   - Builds a `CNNGraph` with Conv2D → Bias → ReLU → MaxPool2D.
   - Asserts that the resulting `CNNExecutionPlan` has four steps with
     `CNNPlanStepKind` in exactly that order.

2. **cnn_steps_are_ordered_and_abortable**

   - Builds a simple Conv2D → ReLU graph.
   - Verifies ordering of step kinds.
   - Asserts that a pre-aborted `AbortFlag` causes
     `CNNAdapterError::Aborted`.

3. **cnn_adapter_respects_execution_contract**

   - Ensures the plan is non-empty and that every step kind is one of the
     CNN-specific variants.

4. **invalid_cnn_graph_yields_explicit_error**

   - Empty graph → `InvalidGraph`.
   - Bias before Conv2D → `InvalidGraph`.
   - Multiple Conv2D layers → `InvalidGraph`.

5. **cnn_adapter_is_deterministic**

   - Builds two plans from the same `CNNGraph` and `AbortFlag` state.
   - Asserts that both have the same step kinds and descriptions.

All tests are pure and deterministic, with no kernel execution or external
dependencies.

## Status of APX 17.10.3

APX 17.10.3 is considered **DONE** when:

- `cnn_adapter.rs` defines `CNNGraph`, `CNNExecutionPlan` and
  `CNNExecutionAdapter::build_plan`.
- `cargo test --test cnn_adapter_test` passes without panics or warnings.
- The adapter is deterministic, abortable and does not execute kernels.
- Invalid CNN graphs yield explicit `CNNAdapterError` values.
- Existing CNN operators and v16 planning logic remain unchanged.

The current implementation meets these criteria and provides a clean,
auditable bridge between the CNN operator stack and execution planning,
ready to be composed with APX 16 executors and profiling without changing
kernel semantics.

---

# APX 17.10.4 — MNIST CNN End-to-End Inference

APX 17.10.4 wires together the CNN operator stack (Conv2D, Bias, ReLU,
MaxPool2D) with a synthetic MNIST-like model to execute a **full CNN
inference path** under APX 17, while remaining fully deterministic and
CI-safe. This is the first real model execution milestone of Atenia Engine.

Key constraints:

- CPU backend only.
- Single inference, no batching.
- No training or optimization.
- No graph fusion or ONNX runtime.
- No external IO or randomness.

The focus is on correctness, contract compliance, abortability, snapshotting
and explainability, not performance.

## Module layout

APX 17.10.4 introduces a dedicated MNIST CNN module under `src/v17/cnn`:

```text
src/
  v17/
    cnn/
      mnist/
        mod.rs
        mnist_model.rs
        mnist_input.rs
        mnist_runner.rs
```

`src/v17/cnn/mod.rs` exposes the MNIST module as:

```rust
pub mod mnist;
```

## MNIST model description

File: `src/v17/cnn/mnist/mnist_model.rs`

```rust
use crate::v17::compute::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MnistCNNModel {
    pub conv_weights: Tensor,
    pub conv_bias: Tensor,
    pub dense_weights: Tensor,
    pub dense_bias: Tensor,
    pub target_digit: usize,
}

impl MnistCNNModel {
    pub fn synthetic() -> Self { /* ... */ }
}
```

The synthetic CNN is MNIST-like but small and fully embedded:

- Conv layer: 1 input channel, 1 output channel, 3×3 kernel.
- Bias for the conv layer.
- MaxPool2D 2×2 with stride 2 → spatial size 14×14.
- Dense layer: 10 outputs, input dimension `1 × 14 × 14`.
- Weights are chosen so that a single `target_digit` (e.g. 3) has the
  largest logit for the fixed synthetic input.

This design ensures a clear, deterministic prediction without requiring a
real trained model or external weights.

## MNIST input

File: `src/v17/cnn/mnist/mnist_input.rs`

```rust
use crate::v17::compute::tensor::Tensor;

pub fn mnist_synthetic_input() -> Tensor { /* 1x1x28x28 ramp pattern */ }
```

The input tensor has shape `[1, 1, 28, 28]` and contains a deterministic
pattern (a simple ramp across the grid). There is no randomness or IO.

## MNIST runner and dual plans

File: `src/v17/cnn/mnist/mnist_runner.rs`

```rust
use crate::v17::cnn::conv2d::AbortFlag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnistRunnerError {
    AdapterError(CNNAdapterError),
    Aborted,
    InvalidState(String),
}

#[derive(Debug, Clone)]
pub struct MnistInferenceResult {
    pub logits: Tensor,
    pub predicted_digit: usize,
    pub cnn_plan: CNNExecutionPlan,
    pub logical_plan: ExecutionPlan,
    pub snapshot: ExecutionSnapshot,
    pub explanation_text: String,
}

pub fn run_mnist_inference(
    abort_flag: &AbortFlag,
) -> Result<MnistInferenceResult, MnistRunnerError> { /* ... */ }
```

### Two distinct plans

APX 17.10.4 uses **two complementary planning artifacts**:

- `CNNExecutionPlan` (from `CNNExecutionAdapter`):
  - Describes the actual CNN operator sequence:
    Conv2D → Bias → ReLU → MaxPool2D.
  - Each layer maps 1:1 to a `CNNPlanStep`.
  - This is the *real execution plan* for CNN kernels.

- `ExecutionPlan` (APX 16 planner):
  - Derived from an `ExecutionContract` as in the generic `infer` path.
  - Used here as a **logical/explanatory plan** for snapshot hashing and
    high-level explainability, not for driving CNN operators.

The runner never tries to inject CNN-specific steps into the APX 16
`ExecutionPlan` nor to run the CNN through `SafeExecutor`; that integration
is explicitly deferred to APX 18.

### Execution pipeline

`run_mnist_inference` performs:

1. Builds `MnistCNNModel::synthetic()` and `mnist_synthetic_input()`.
2. Constructs a `CNNGraph` with layers:
   `Conv2D → Bias → ReLU → MaxPool2D`.
3. Builds `CNNExecutionPlan` via `CNNExecutionAdapter::build_plan`.
4. Builds a standard APX 16 `ExecutionContract` and `ExecutionPlan` as
   logical metadata.
5. Executes the CNN pipeline in-process:
   - `conv2d_cpu` → `add_bias` → `relu` → `maxpool2d_cpu`.
   - Flattens `[1, C, H, W]` to `[1, flat_dim]`.
   - Applies a dense layer via a local matmul + bias using the model's
     `dense_weights` and `dense_bias`.
6. Computes `predicted_digit` via argmax over the 10 logits.
7. Builds an `ExecutionSnapshot` + simple textual explanation based on the
   CNN plan.

`AbortFlag` is honored before and during the pipeline: if set, the runner
returns `MnistRunnerError::Aborted` and does not proceed.

## Tests for APX 17.10.4

File: `tests/mnist_end_to_end_test.rs`

The test module validates the MNIST pipeline via five tests:

1. **end_to_end_mnist_inference_produces_expected_digit**

   - Calls `run_mnist_inference` with a clean `AbortFlag`.
   - Asserts `logits.shape == [1, 10]`.
   - Verifies that `predicted_digit` equals the synthetic model's
     `target_digit` (e.g. 3).

2. **inference_is_deterministic**

   - Runs `run_mnist_inference` twice.
   - Asserts identical logits and predicted digit.

3. **execution_respects_execution_contract**

   - Asserts that the logical `ExecutionPlan` is non-empty and derived from
     a valid `ExecutionContract`.

4. **abort_stops_inference_safely**

   - Sets `AbortFlag` before calling `run_mnist_inference`.
   - Expects `Err(MnistRunnerError::Aborted)` without panics.

5. **snapshot_is_generated_and_valid**

   - Asserts that the `ExecutionSnapshot` fields are non-empty and
     consistent (model id, backend usage, snapshot hash, output and
     explanation signatures).
   - Checks that the explanation text is non-empty and lists the CNN
     execution steps.

All MNIST tests are deterministic, perform no IO, and depend only on the
embedded synthetic model and input, ensuring CI safety.

## Status of APX 17.10.4

APX 17.10.4 is considered **DONE** when:

- `mnist_model.rs`, `mnist_input.rs` and `mnist_runner.rs` are implemented
  under `src/v17/cnn/mnist/`.
- `cargo test --test mnist_end_to_end_test` passes without panics or
  warnings.
- CNNExecutionPlan drives the real CNN operator sequence.
- APX 16 `ExecutionPlan` is used only as logical metadata for
  explainability and snapshotting.
- The pipeline is deterministic, abortable and uses no external IO.

The current implementation satisfies these criteria and demonstrates that
Atenia Engine can execute a full CNN inference path end-to-end under APX 17,
using real kernels, explicit planning and evidence artifacts, while
remaining fully auditable and CI-safe.
