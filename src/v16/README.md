# APX v16.0 — Execution Contract Resolver

APX v16.0 defines the **execution contract layer**. It does not execute kernels, move
Tensors, or talk to hardware. Instead, it takes the **intention** coming from APX 15
(`DecisionBias`) and a passive snapshot of runtime-like state, and produces an
`ExecutionContract` that describes what would be **legally allowed** for a future
execution.

- No execution
- No planning
- No runtime integration
- No hardware calls

APX 15 decides *what we want*; APX 16.0 decides *what is allowed*.

---

## Directory layout (v16.0)

Location: `src/v16`

```text
src/
└── v16/
    ├── mod.rs
    └── contract/
        ├── mod.rs
        ├── execution_contract.rs
        ├── constraints.rs
        ├── contract_resolver.rs
        └── contract_errors.rs
```

Tests for APX 16.0 live in:

```text
tests/execution_contract_test.rs
```

APX v16 is fully isolated from the existing runtime. It only reuses v15's policy
types (`DecisionBias`) as inputs.

---

## Core concepts

### DecisionBias (from APX 15)

APX 16.0 does **not** redefine decision making. It consumes the `DecisionBias` from
APX 15 after policies, evidence, and user preferences have been applied.

```rust
pub struct DecisionBias {
    pub risk_weight: f32,
    pub latency_weight: f32,
    pub stability_weight: f32,
    pub memory_pressure_weight: f32,
    pub offload_cost_weight: f32,
}
```

In APX 16.0, `DecisionBias` expresses **intent**:

- Higher `latency_weight` means more willingness to favor low-latency choices.
- Higher `stability_weight` means preference for safer, more stable behavior.
- Higher `risk_weight` means willingness to accept more risk.
- `memory_pressure_weight` and `offload_cost_weight` express trade-offs related to
  memory and offload.

APX 16.0 does not modify these weights; it only uses them to constrain what is
legally allowed.

---

### RuntimeState (snapshot)

File: `src/v16/contract/constraints.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeState {
    /// Normalized memory headroom in [0.0, 1.0].
    pub memory_headroom: f32,
    /// Whether the system is currently considered stable.
    pub is_stable: bool,
    /// Whether a recovery happened recently.
    pub recent_recovery: bool,
    /// Whether offload-style backends are available at all.
    pub offload_supported: bool,
}
```

`RuntimeState` is a **passive snapshot** of relevant high-level conditions:

- It is **read-only** from the perspective of APX 16.0.
- It is not a live view and is never mutated.
- It does not expose concrete devices or kernel details.

This snapshot is used together with `DecisionBias` to derive constraints, but not
for executing any action.

---

### Constraints

File: `src/v16/contract/constraints.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintSeverity {
    Hard,
    Soft,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    ForbidOffload,
    LimitAggressiveness { max: f32 },
    RequireStability,
    RequireFallback,
    MemoryHeadroom { min: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub severity: ConstraintSeverity,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraints {
    pub items: Vec<Constraint>,
}
```

`Constraint*` types capture **derived restrictions** for future execution. They do
not execute anything; they are purely descriptive.

Examples:

- `ForbidOffload` – offload-style backends must not be used.
- `LimitAggressiveness { max }` – aggressiveness must stay below a threshold.
- `RequireStability` – the contract requires stability-oriented behavior.
- `RequireFallback` – any aggressive choice must have a safe fallback path.
- `MemoryHeadroom { min }` – execution must respect the observed headroom.

Each constraint has a `severity`:

- `Hard` – must be satisfied; otherwise execution is illegal.
- `Soft` – preferred, but may be relaxed by higher layers if safe to do so.

Constraints are **derived**, not invented: they come from combining `DecisionBias`
with `RuntimeState`.

---

### ExecutionContract

File: `src/v16/contract/execution_contract.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionBackend {
    Local,
    Offload,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionContract {
    pub bias: DecisionBias,
    pub runtime_snapshot: RuntimeState,
    pub allowed_backends: Vec<ExecutionBackend>,
    pub forbidden_backends: Vec<ExecutionBackend>,
    pub max_aggressiveness: f32,
    pub require_fallback: bool,
    pub require_stability: bool,
    pub constraints: Constraints,
}
```

`ExecutionContract` is the central artifact of APX 16.0. It is an **immutable
specification** of what is legally allowed for a future execution:

- `bias` – original intent from APX 15.
- `runtime_snapshot` – the `RuntimeState` used to derive the contract.
- `allowed_backends` / `forbidden_backends` – abstract backends, not real devices.
- `max_aggressiveness` – upper bound on how aggressive choices can be.
- `require_fallback` – whether aggressive paths must have a safe fallback.
- `require_stability` – whether stability-centric behavior is required.
- `constraints` – explicit list of derived `Constraint` values.

The contract **does not**:

- Contain actions or steps.
- Execute or schedule anything.
- Encode hardware-specific details.

It is a declarative boundary that later layers (16.1, 16.2, …) must respect.

---

### Contract errors

File: `src/v16/contract/contract_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ContractError {
    NoLegalExecution(String),
    IncompatibleIntent(String),
    InvariantViolation(String),
}
```

`ContractError` captures explicit failure modes during contract resolution:

- `NoLegalExecution` – under the given bias and state there is no legal execution.
- `IncompatibleIntent` – the requested intent is incompatible with observed state.
- `InvariantViolation` – basic invariants (e.g. normalization, ranges) are broken.

Errors are **descriptive only**:

- No actions are triggered.
- No recovery is performed.
- No state is mutated.

---

### ContractResolver

File: `src/v16/contract/contract_resolver.rs`

```rust
pub struct ContractResolver;

impl ContractResolver {
    pub fn resolve_contract(
        bias: &DecisionBias,
        state: &RuntimeState,
    ) -> Result<ExecutionContract, ContractError> { /* ... */ }
}
```

`ContractResolver` is the only component that:

- Reads `DecisionBias` (intent) and `RuntimeState` (snapshot).
- Validates invariants.
- Produces an `ExecutionContract` or a `ContractError`.

It does **not**:

- Execute or plan anything.
- Decide new policies.
- Touch the runtime or hardware.

#### Invariant checks

The resolver enforces basic invariants before producing a contract:

- `DecisionBias` must be normalized (`is_normalized() == true`).
- `memory_headroom` must be in `[0.0, 1.0]`.

If these invariants fail, it returns `ContractError::InvariantViolation`.

#### Deriving constraints

From `bias` and `state`, the resolver derives:

- A simple `base_aggressiveness` score:
  - `(latency_weight + risk_weight).min(1.0)`.
- When `memory_headroom` is extremely low and `base_aggressiveness` very high,
  the intent is considered incompatible:
  - Returns `ContractError::IncompatibleIntent`.
- `require_stability` is set when any of these hold:
  - `state.recent_recovery`
  - `!state.is_stable`
  - `stability_weight >= 0.8`
- If `require_stability` is true, a hard `RequireStability` constraint is added.
- When `memory_headroom` is low or the system is unstable, a soft
  `LimitAggressiveness { max: 0.5 }` constraint is added and
  `max_aggressiveness` is capped at `0.5`.
- `MemoryHeadroom` constraints:
  - `Hard` when headroom is < `0.2`.
  - `Soft` otherwise.

Backends are derived as:

- `Local` is always allowed.
- If `offload_supported` is true:
  - Low `offload_cost_weight` (<= 0.5) → allow `Offload`, require fallback,
    and add a soft `RequireFallback` constraint.
  - Otherwise → forbid `Offload` and add a hard `ForbidOffload` constraint.
- If `offload_supported` is false:
  - `Offload` is forbidden with a hard `ForbidOffload` constraint.

If the resulting `allowed_backends` set is empty, the resolver returns
`ContractError::NoLegalExecution`.

All of this logic is **pure** and depends only on the inputs.

---

## Tests for APX 16.0

File: `tests/execution_contract_test.rs`

The contract resolver is tested independently of the runtime and v15 policies. Tests
import v15 and v16 modules via `#[path = "../src/v15/mod.rs"]` and
`#[path = "../src/v16/mod.rs"]` without touching `src/lib.rs`.

The test suite covers:

1. **Contract is produced from valid bias**

   - With a normalized `DecisionBias` and a valid `RuntimeState`,
     `resolve_contract` returns `Ok(ExecutionContract)`.
   - The contract preserves the input `bias` and `runtime_snapshot` and has at
     least one allowed backend.

2. **Invalid bias/state yields explicit error**

   - Bias with weights outside `[0.0, 1.0]` → `InvariantViolation`.
   - `RuntimeState` with `memory_headroom` outside `[0.0, 1.0]` → `InvariantViolation`.

3. **Contract is deterministic**

   - Two calls to `resolve_contract` with identical `bias` and `state` produce
     identical `ExecutionContract` values.

4. **No side-effects during resolution**

   - Inputs (`DecisionBias` and `RuntimeState`) are cloned before resolution and
     compared afterwards to ensure they are unchanged.

5. **Constraints reflect bias priorities**

   - Strong stability bias with unstable state yields `require_stability = true`
     and a `RequireStability` constraint.
   - Very aggressive bias with low headroom yields a contract where
     `max_aggressiveness` is capped (<= 0.5) and a `LimitAggressiveness` constraint
     is present.

Running APX 16.0 tests:

```bash
cargo test --test execution_contract_test
```

All tests are deterministic, fast, and CI-safe, with no external dependencies and
no runtime or hardware interaction.

---

## Status of APX 16.0

APX 16.0 is considered **DONE** when:

- There is a clear `ExecutionContract` that describes what is allowed.
- The contract is derived from `DecisionBias` (APX 15) and `RuntimeState`.
- No execution or planning happens inside v16.0.
- Contract errors are explicit and descriptive.
- Tests validate determinism, invariants, and safety (no illegal forced actions).

The current implementation satisfies these criteria and provides a stable base for
future APX 16.x layers that will plan and execute within the boundaries defined
by the contract.

---

## APX 16.1 — Execution Planner

APX 16.1 introduces the **planning layer** on top of the execution contract. It
does not execute anything; it only produces a **descriptive, ordered
ExecutionPlan** from an `ExecutionContract`.

- No kernel execution
- No tensor moves
- No runtime or hardware access
- No new policy decisions

APX 16.0 defines *what is legal*.
APX 16.1 defines *how it would be attempted*, as a sequence of abortable steps.

### Directory layout additions (16.1)

New module under `src/v16`:

```text
src/v16/
    planner/
        mod.rs
        execution_plan.rs
        plan_step.rs
        execution_planner.rs
        planner_errors.rs
```

Tests for APX 16.1 live in:

```text
tests/execution_planner_test.rs
```

---

## PlanStep

File: `src/v16/planner/plan_step.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PlanStepKind {
    EnsureMemoryHeadroom,
    SelectBackendCandidate,
    PrepareFallback,
    MarkTensorsMovable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanStep {
    pub kind: PlanStepKind,
    pub description: String,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub abortable: bool,
    pub requires_verification: bool,
}
```

`PlanStep` is the **minimal unit** of the execution plan. Each step:

- Describes an abstract action type (`PlanStepKind`).
- Lists **preconditions** that must conceptually hold before the step.
- Lists **postconditions** expected after a successful step.
- Is always `abortable: true` in APX 16.1.
- Can require explicit verification (`requires_verification`).

Crucially, `PlanStep` does not:

- Contain executable logic.
- Touch runtime or hardware.
- Assume success.

It is a declarative description of what would be attempted in a later phase.

---

## ExecutionPlan

File: `src/v16/planner/execution_plan.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPlan {
    pub contract: ExecutionContract,
    pub steps: Vec<PlanStep>,
    pub globally_abortable: bool,
}
```

`ExecutionPlan` is an **immutable planning artifact** derived from an
`ExecutionContract`:

- `contract` – the exact contract used to derive the plan.
- `steps` – ordered sequence of `PlanStep` values.
- `globally_abortable` – whether the plan is considered abortable at any point.

The plan does not execute anything; it only captures **intentional structure**:

- What steps would be attempted.
- In what order.
- Under which conceptual conditions.

---

## Planner errors

File: `src/v16/planner/planner_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PlannerError {
    UnplannableContract(String),
    InvalidContract(String),
}
```

`PlannerError` represents explicit failure modes during planning:

- `InvalidContract` – the contract is structurally invalid for planning
  (e.g., no allowed backends).
- `UnplannableContract` – constraints are internally incompatible in a way that
  prevents a safe plan.

Errors are deterministic, descriptive, and have **no side effects**.

---

## ExecutionPlanner

File: `src/v16/planner/execution_planner.rs`

```rust
pub struct ExecutionPlanner;

impl ExecutionPlanner {
    pub fn build_plan(contract: &ExecutionContract) -> Result<ExecutionPlan, PlannerError> {
        /* ... */
    }
}
```

`ExecutionPlanner` is responsible for:

- Receiving a validated `ExecutionContract` (16.0).
- Checking minimal consistency conditions.
- Producing an `ExecutionPlan` or a `PlannerError`.

It does not:

- Execute kernels or move tensors.
- Access runtime or hardware.
- Make new policy decisions.

### Planning logic (APX 16.1)

Given a contract, the planner:

1. **Validates basic structure**

   - If `allowed_backends` is empty → `PlannerError::InvalidContract`.
   - If `require_stability` is true but there is no `ConstraintKind::RequireStability`
     in the contract’s constraints → `PlannerError::UnplannableContract`.

2. **Builds an ordered sequence of steps**

   In APX 16.1, the planner always produces the same sequence shape for a valid
   contract:

   - `EnsureMemoryHeadroom`
   - `SelectBackendCandidate`
   - `PrepareFallback` (only if `contract.require_fallback` is true)
   - `MarkTensorsMovable`

   Each step:

   - Is marked `abortable: true`.
   - Sets `requires_verification: true` to emphasize that future layers must
     verify conditions before acting.

3. **Produces the final plan**

   - The contract is cloned into the plan (`plan.contract`).
   - `globally_abortable` is set to `true`.

The function is **pure** and deterministic: for the same `ExecutionContract`,
`build_plan` always returns the same `ExecutionPlan` (or the same `PlannerError`).

---

## Tests for APX 16.1

File: `tests/execution_planner_test.rs`

The execution planner is tested independently of the runtime. Tests construct
synthetic `ExecutionContract` values using the same types as 16.0 and a simple
`DecisionBias` from v15.

The test suite covers:

1. **Plan is produced from valid contract**

   - For a well-formed contract with allowed backends and consistent
     constraints, `build_plan` returns `Ok(ExecutionPlan)`.
   - The plan preserves the contract and contains at least one step.
   - `globally_abortable` is `true`.

2. **Plan respects contract constraints**

   - When the contract requires stability and fallback, the resulting plan
     includes:
     - `EnsureMemoryHeadroom`
     - `SelectBackendCandidate`
     - `PrepareFallback` (if `require_fallback` is true)

3. **Planner is deterministic**

   - Two calls to `build_plan` with the same contract produce identical
     `ExecutionPlan` values.

4. **Invalid contract yields explicit error**

   - A contract with an empty `allowed_backends` list yields
     `PlannerError::InvalidContract`.

5. **Plan steps are ordered and abortable**

   - All steps in the plan have `abortable == true`.
   - The index of `EnsureMemoryHeadroom` is strictly before
     `SelectBackendCandidate`, enforcing a sensible high-level order.

Running APX 16.1 tests:

```bash
cargo test --test execution_planner_test
```

All tests are deterministic, fast, and CI-safe, with no external dependencies and
no runtime or hardware interaction.

---

## Status of APX 16.1

APX 16.1 is considered **DONE** when:

- There is a clear `ExecutionPlan` structure.
- Every plan is derived from an `ExecutionContract` and respects its
  constraints.
- No step executes anything or touches runtime/hardware.
- Planner errors are explicit and descriptive.
- Tests validate ordering, abortability, and determinism.

The current implementation meets these criteria and provides a stable planning
layer on top of APX 16.0, ready for future APX 16.2+ execution logic to consume
these plans safely.

---

## APX 16.2 — Safe Executor

APX 16.2 is the first layer that performs **real execution**. It takes an
`ExecutionPlan` from APX 16.1 and executes its steps **sequentially**, in a
validated and abortable way.

Key guarantees:

- No step is executed outside the plan.
- No step is executed without validation.
- Every step is abortable.
- Errors are explicit and do not improvise behavior.

### Directory layout additions (16.2)

New module under `src/v16`:

```text
src/v16/
    executor/
        mod.rs
        safe_executor.rs
        execution_context.rs
        step_executor.rs
        executor_state.rs
        executor_errors.rs
```

Tests for APX 16.2 live in:

```text
tests/safe_executor_test.rs
```

---

## ExecutionContext and RuntimeFacade

File: `src/v16/executor/execution_context.rs`

```rust
pub trait RuntimeFacade {
    fn ensure_memory_headroom(&mut self) -> Result<(), String>;
    fn select_backend_candidate(&mut self) -> Result<(), String>;
    fn prepare_fallback(&mut self) -> Result<(), String>;
    fn mark_tensors_movable(&mut self) -> Result<(), String>;
}

#[derive(Debug)]
pub struct ExecutionContext<R: RuntimeFacade> {
    pub runtime: R,
}
```

`RuntimeFacade` is a **minimal trait** that abstracts the underlying runtime
capabilities needed by APX 16.2. It can be implemented by real runtimes or by
test mocks.

`ExecutionContext` wraps a `RuntimeFacade` instance and is the **only access
point** for the executor to interact with the runtime:

- No direct hardware handles.
- No arbitrary execution entry points.
- No policy or evidence access.

---

## ExecutorState

File: `src/v16/executor/executor_state.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorStatus {
    Running,
    Aborted,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutorState {
    pub current_step: usize,
    pub status: ExecutorStatus,
    pub executed_steps: Vec<usize>,
}
```

`ExecutorState` tracks **minimal, in-memory execution state**:

- `current_step` – index of the next step in the plan to execute.
- `status` – overall executor status.
- `executed_steps` – indices of steps that have completed successfully.

The state is not persisted, is not exposed to policies, and does not modify the
plan.

---

## Executor errors

File: `src/v16/executor/executor_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorError {
    PreconditionFailed(String),
    UnsafeToExecute(String),
    StepFailed(String),
    Aborted(String),
}
```

`ExecutorError` captures explicit failure modes during execution:

- `PreconditionFailed` – preconditions for the step are not satisfied.
- `UnsafeToExecute` – the step would be unsafe under current conditions.
- `StepFailed` – the underlying runtime reported a failure.
- `Aborted` – execution was explicitly aborted.

Errors are typed, descriptive, and have no hidden side effects.

---

## StepExecutor

File: `src/v16/executor/step_executor.rs`

```rust
pub struct StepExecutor;

impl StepExecutor {
    pub fn execute_step<R: RuntimeFacade>(
        step: &PlanStep,
        ctx: &mut ExecutionContext<R>,
    ) -> Result<(), ExecutorError> {
        match step.kind {
            PlanStepKind::EnsureMemoryHeadroom => ctx
                .runtime
                .ensure_memory_headroom()
                .map_err(ExecutorError::StepFailed),
            PlanStepKind::SelectBackendCandidate => ctx
                .runtime
                .select_backend_candidate()
                .map_err(ExecutorError::StepFailed),
            PlanStepKind::PrepareFallback => ctx
                .runtime
                .prepare_fallback()
                .map_err(ExecutorError::StepFailed),
            PlanStepKind::MarkTensorsMovable => ctx
                .runtime
                .mark_tensors_movable()
                .map_err(ExecutorError::StepFailed),
        }
    }
}
```

`StepExecutor` is responsible for executing a **single** `PlanStep`:

- It does not modify the plan.
- It does not reorder or skip steps.
- It simply delegates to the corresponding `RuntimeFacade` method and maps
  errors into `ExecutorError`.

There is no hidden logic or speculative behavior: *one step = one runtime call*.

---

## SafeExecutor

File: `src/v16/executor/safe_executor.rs`

```rust
pub struct SafeExecutor<R: RuntimeFacade> {
    pub state: ExecutorState,
    pub context: ExecutionContext<R>,
    pub plan: ExecutionPlan,
}
```

`SafeExecutor` is the central component of APX 16.2. It:

- Receives an `ExecutionPlan` (16.1).
- Holds an `ExecutionContext<R>` with a `RuntimeFacade` implementation.
- Maintains `ExecutorState` while executing steps.

### API

```rust
impl<R: RuntimeFacade> SafeExecutor<R> {
    pub fn new(plan: ExecutionPlan, context: ExecutionContext<R>) -> Self { /* ... */ }

    pub fn status(&self) -> ExecutorStatus { /* ... */ }

    pub fn step(&mut self) -> Result<(), ExecutorError> { /* ... */ }

    pub fn abort(&mut self, reason: &str) -> ExecutorError { /* ... */ }
}
```

Behavior:

- `step()`:
  - Only runs when `status == Running`.
  - If `current_step` is beyond the last step, sets status to `Completed`.
  - Fetches the next `PlanStep` from the plan.
  - Validates that the step is `abortable`; otherwise marks `Failed` and
    returns `UnsafeToExecute`.
  - Delegates to `StepExecutor::execute_step`:
    - On success: records the step index, advances `current_step`, and marks
      `Completed` if no steps remain.
    - On error: marks `Failed` and returns the `ExecutorError`.
- `abort(reason)`:
  - Sets status to `Aborted` and returns an `ExecutorError::Aborted`.
  - Subsequent calls to `step()` do not execute further steps.

The executor never:

- Modifies the plan.
- Executes steps not present in the plan.
- Skips ahead or reorders steps.

Execution is strictly sequential and controlled.

---

## Tests for APX 16.2

File: `tests/safe_executor_test.rs`

Tests use a `MockRuntime` implementing `RuntimeFacade` to simulate runtime
behavior without touching real hardware. The mock records method calls and can
be configured to fail on specific calls.

The test suite covers:

1. **Executes steps in order**

   - Runs `step()` until the executor is no longer `Running`.
   - Asserts that status becomes `Completed`.
   - Checks that the sequence of runtime calls matches the order of plan steps.

2. **Stops execution on failed step**

   - Configures the mock to fail on `select_backend_candidate`.
   - First step succeeds, second fails, status becomes `Failed`.
   - No further runtime calls are made (e.g., no `prepare_fallback`).

3. **Abort stops execution safely**

   - Executes one step, then calls `abort(...)`.
   - Status becomes `Aborted`.
   - Further calls to `step()` do not trigger new runtime calls.

4. **No execution outside plan**

   - Runs until completion and asserts that the number of runtime calls does not
     exceed the number of steps in the plan.

5. **Executor state transitions are valid**

   - Verifies transitions `Running → Running → Aborted` and that subsequent
     calls to `step()` do not change the status.

Running APX 16.2 tests:

```bash
cargo test --test safe_executor_test
```

All tests are deterministic, CI-safe, and interact only with the `MockRuntime`
via `ExecutionContext`.

---

## Status of APX 16.2

APX 16.2 is considered **DONE** when:

- It executes real steps **one by one** based on an `ExecutionPlan`.
- It never executes outside the plan.
- It can abort at any time without corrupting state.
- Executor errors are explicit and descriptive.
- Tests validate order, abort behavior, and safety properties.

The current implementation meets these criteria and turns the APX 15–16 stack
into a concrete but still controlled execution path: intention → contract →
plan → safe execution.

---

## APX 16.3 — Runtime Feedback & Event Emission

APX 16.3 provides the **observability bridge** for the execution stack. It does
not execute anything or make decisions; it only converts the results of an
execution into **structured, deterministic events and outcomes** that can be
consumed by observability layers (e.g. APX 14).

- No execution
- No policy decisions
- No recovery
- No runtime access

APX 16.2 executes.
APX 16.3 observes.

### Directory layout additions (16.3)

New module under `src/v16`:

```text
src/v16/
    feedback/
        mod.rs
        execution_event.rs
        event_emitter.rs
        execution_outcome.rs
        feedback_collector.rs
        feedback_errors.rs
```

Tests for APX 16.3 live in:

```text
tests/runtime_feedback_test.rs
```

---

## ExecutionEvent

File: `src/v16/feedback/execution_event.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionEventKind {
    ExecutionStarted,
    StepStarted,
    StepSucceeded,
    ExecutionFailed,
    ExecutionAborted,
    ExecutionCompleted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionEvent {
    pub kind: ExecutionEventKind,
    pub step_index: Option<usize>,
    pub logical_timestamp: u64,
    pub severity: EventSeverity,
    pub message: String,
}
```

`ExecutionEvent` is a **structured event** representing something that happened
during execution:

- `kind` – event type (start, per-step, terminal conditions).
- `step_index` – optional index of the associated plan step.
- `logical_timestamp` – monotonic counter assigned by the emitter (no real
  clock).
- `severity` – `Info`, `Warning`, or `Error`.
- `message` – minimal human-readable description.

Events contain no logic, do not execute anything, and are suitable for
serialization and offline analysis.

---

## ExecutionOutcome

File: `src/v16/feedback/execution_outcome.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionOutcomeKind {
    Completed,
    Failed,
    Aborted,
    PartiallyCompleted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionOutcome {
    pub kind: ExecutionOutcomeKind,
    pub executed_steps: Vec<usize>,
    pub final_error: Option<String>,
}
```

`ExecutionOutcome` summarizes the **final result** of an execution:

- `kind` – global status.
- `executed_steps` – indices of steps that completed successfully.
- `final_error` – message describing the final error, if any.

It does not decide or correct anything; it simply describes what happened.

---

## Feedback errors

File: `src/v16/feedback/feedback_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackError {
    InconsistentEvents(String),
    InvalidEvent(String),
    LogicalOrderViolation(String),
}
```

`FeedbackError` represents errors in the **feedback layer itself**:

- `InvalidEvent` – for example, a step index out of range.
- `LogicalOrderViolation` – non-monotonic or logically impossible event
  sequences.
- `InconsistentEvents` – reserved for more complex inconsistencies.

These errors are deterministic and descriptive, and they never trigger
recovery or execution.

---

## EventEmitter

File: `src/v16/feedback/event_emitter.rs`

```rust
pub struct EventEmitter;

impl EventEmitter {
    pub fn emit_for_snapshot(
        plan: &ExecutionPlan,
        executed_steps: &[usize],
        status: &ExecutorStatus,
        final_error: Option<&str>,
    ) -> Result<(Vec<ExecutionEvent>, ExecutionOutcome), FeedbackError> { /* ... */ }
}
```

`EventEmitter` is responsible for transforming a **snapshot of execution
state** into a list of events and a final outcome:

- Validates `executed_steps`:
  - All indices are within the plan’s step range.
  - Indices are non-decreasing (no reordering or time travel).
  - Violations yield `FeedbackError::InvalidEvent` or
    `FeedbackError::LogicalOrderViolation`.

- Emission logic:
  - Emits a single `ExecutionStarted` event.
  - For each executed step index:
    - `StepStarted`
    - `StepSucceeded`
  - Emits a **terminal event** and derives an `ExecutionOutcomeKind` based on
    `ExecutorStatus` and how many steps were executed:
    - `Completed` → `ExecutionCompleted` + outcome `Completed`.
    - `Failed` → outcome `Failed` (if all steps ran) or `PartiallyCompleted`;
      terminal `ExecutionFailed` with `Error` severity.
    - `Aborted` → outcome `Aborted` (no steps) or `PartiallyCompleted`;
      terminal `ExecutionAborted` with `Warning` severity.
    - `Running` → outcome `PartiallyCompleted` and a warning snapshot event.

The emitter is pure and deterministic: same inputs → same events and outcome.
It does not touch the runtime or the executor.

---

## FeedbackCollector

File: `src/v16/feedback/feedback_collector.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackSnapshot {
    pub events: Vec<ExecutionEvent>,
    pub outcome: ExecutionOutcome,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct FeedbackCollector {
    events: Vec<ExecutionEvent>,
    outcome: Option<ExecutionOutcome>,
}
```

`FeedbackCollector` accumulates emitted events and the final outcome in memory
and exposes them as a snapshot:

- `record(events, outcome)` stores a new set of events and the corresponding
  outcome.
- `snapshot()` returns an optional `FeedbackSnapshot` with cloned data.

It does not mutate execution or persist anything; it is just a container for
observed results.

---

## Tests for APX 16.3

File: `tests/runtime_feedback_test.rs`

Tests simulate execution using a real `ExecutionPlan` and synthetic
`executed_steps` and `ExecutorStatus` values. No hardware or runtime is
involved.

The test suite covers:

1. **Events are emitted in execution order**

   - Logical timestamps are strictly increasing across events.
   - The first event is `ExecutionStarted`, the last is
     `ExecutionCompleted` for a completed execution.

2. **Outcome matches execution result**

   - Completed execution → `ExecutionOutcomeKind::Completed`.
   - Failed after partial execution → `PartiallyCompleted`.
   - Aborted with no steps → `Aborted`.

3. **Feedback is deterministic and pure**

   - Two calls to `emit_for_snapshot` with identical inputs produce identical
     `(events, outcome)`.
   - The `ExecutionPlan` and `executed_steps` inputs remain unchanged.

4. **FeedbackCollector records snapshot**

   - `FeedbackCollector::record` followed by `snapshot()` returns events and
     outcome identical to the originals.

5. **Invalid event sequences yield error**

   - Out-of-bounds step index → `FeedbackError::InvalidEvent`.
   - Non-monotonic indices → `FeedbackError::LogicalOrderViolation`.

Running APX 16.3 tests:

```bash
cargo test --test runtime_feedback_test
```

All tests are deterministic, fast, and CI-safe.

---

## Status of APX 16.3

APX 16.3 is considered **DONE** when:

- Every execution can be represented as a sequence of `ExecutionEvent`s.
- The final `ExecutionOutcome` matches the executor’s status and behavior.
- Feedback logic does not alter execution or touch the runtime.
- No decisions or recovery actions are taken based on feedback.
- Tests validate event ordering, determinism, and error handling.

The current implementation meets these criteria and closes the loop from
execution back to observability, enabling APX 14+ to consume real execution
evidence in a structured, auditable form.

---

## APX 16.4 — Adaptive Execution Guards

APX 16.4 adds **dynamic safety guards** on top of the execution stack. Guards
do not execute or replan; they only **observe conditions** and recommend safe
actions such as continuing, degrading, or aborting, always within the
constraints of the `ExecutionContract`.

- No policy decisions
- No plan modifications
- No step execution
- No runtime access

APX 16.2 executes.
APX 16.3 observes.
APX 16.4 protects.

### Directory layout additions (16.4)

New module under `src/v16`:

```text
src/v16/
    guards/
        mod.rs
        guard_conditions.rs
        guard_action.rs
        execution_guard.rs
        guard_manager.rs
        guard_errors.rs
```

Tests for APX 16.4 live in:

```text
tests/adaptive_guards_test.rs
```

---

## GuardConditions

File: `src/v16/guards/guard_conditions.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct GuardConditions {
    pub memory_pressure: f32,
    pub recent_failures: u32,
    pub latency_spike: bool,
    pub pre_oom_signal: bool,
}
```

`GuardConditions` is a **read-only snapshot** of observable signals that
guards can inspect during execution:

- `memory_pressure` – normalized in `[0.0, 1.0]` (higher → more pressure).
- `recent_failures` – count of recent failures in this execution context.
- `latency_spike` – whether latency exceeded an expected range.
- `pre_oom_signal` – whether a pre-OOM-style risk signal is present.

It does not expose hardware directly and is built from existing signals.

---

## GuardAction

File: `src/v16/guards/guard_action.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum GuardAction {
    Continue,
    Degrade,
    Abort,
}
```

`GuardAction` represents the **limited set of actions** a guard can recommend:

- `Continue` – proceed as planned.
- `Degrade` – proceed in a degraded or safer mode.
- `Abort` – stop execution to avoid unsafe conditions.

These are recommendations only; they do not execute or force illegal behavior.

---

## ExecutionGuard

File: `src/v16/guards/execution_guard.rs`

```rust
pub trait ExecutionGuard: Send + Sync {
    fn name(&self) -> &'static str;

    fn evaluate(&self, contract: &ExecutionContract, conditions: &GuardConditions) -> GuardAction;
}
```

`ExecutionGuard` is a **pure, evaluative** trait:

- Reads the `ExecutionContract` and `GuardConditions`.
- Returns a `GuardAction` recommendation.
- Does not touch runtime, execute steps, or modify intent or plans.

Implementations are deterministic: same contract + conditions → same action.

---

## Guard errors

File: `src/v16/guards/guard_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum GuardError {
    IllegalAction(String),
    InconsistentGuards(String),
}
```

`GuardError` captures problems inside the guard layer:

- `IllegalAction` – a recommended action would violate the `ExecutionContract`.
- `InconsistentGuards` – guards produce logically inconsistent recommendations
  (reserved for future use).

Errors are descriptive only; they do not trigger recovery or execution.

---

## GuardManager

File: `src/v16/guards/guard_manager.rs`

```rust
pub struct GuardManager {
    guards: Vec<Box<dyn ExecutionGuard>>,
}

impl GuardManager {
    pub fn new(guards: Vec<Box<dyn ExecutionGuard>>) -> Self { /* ... */ }

    pub fn evaluate(
        &self,
        contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> Result<GuardAction, GuardError> { /* ... */ }
}
```

`GuardManager` orchestrates multiple guards and resolves their recommendations
into a single `GuardAction` while enforcing contract legality.

Aggregation rules:

- `Abort` **dominates** everything.
- If no `Abort`, `Degrade` dominates `Continue`.
- If neither `Abort` nor `Degrade` is requested, the result is `Continue`.

Contract legality check (example in 16.4):

- If the final recommendation is `Continue` while `conditions.pre_oom_signal`
  is `true` and `contract.require_stability` is also `true`, the manager
  returns `GuardError::IllegalAction`.

The manager does not execute actions; it only produces recommendations.

---

## Tests for APX 16.4

File: `tests/adaptive_guards_test.rs`

Tests define concrete guard implementations to exercise the guard framework:

- `MemoryPressureGuard` – aborts when `memory_pressure` exceeds a threshold.
- `FailureCountGuard` – degrades when `recent_failures` exceed a limit.
- `AlwaysContinueGuard` – always recommends `Continue`.

The test suite covers:

1. **Guards detect risk and recommend action**

   - With high-risk conditions, `MemoryPressureGuard` recommends `Abort`.

2. **Abort dominates over degrade**

   - When both an abort and degrade condition are present, the final action is
     `Abort`.

3. **Guards never violate execution contract**

   - A guard that would always `Continue` under high pre-OOM risk with
     `require_stability == true` results in `GuardError::IllegalAction`.

4. **Guard evaluation is deterministic and pure**

   - Two evaluations with the same contract and conditions yield identical
     `GuardAction`.
   - Neither the contract nor the conditions are mutated.

5. **Low-risk conditions yield Continue**

   - Under low-risk `GuardConditions`, the combined action is `Continue`.

Running APX 16.4 tests:

```bash
cargo test --test adaptive_guards_test
```

All tests are deterministic, fast, and CI-safe, and they work entirely with
simulated conditions and contracts.

---

## Status of APX 16.4

APX 16.4 is considered **DONE** when:

- Guards detect risk correctly and recommend safe actions.
- Recommended actions are always legal under the `ExecutionContract`.
- No execution or planning occurs inside the guard layer.
- Abort and degrade decisions are explicit and contract-respecting.
- Tests validate safety, determinism, and absence of side effects.

The current implementation meets these criteria, adding a dynamic safety
envelope around APX 16.2 execution without changing intent, plans, or
contracts.

---

## APX 16.5 — Speculative Execution

APX 16.5 introduces **controlled speculative execution**. It allows executing
an `ExecutionPlan` speculatively under strict constraints:

- All speculative effects must be reversible.
- Speculative execution must remain isolated from the main execution context.
- The `ExecutionContract` and abortability guarantees must never be violated.

### Directory layout additions (16.5)

New module under `src/v16`:

```text
src/v16/
    speculative/
        mod.rs
        speculative_plan.rs
        speculative_context.rs
        speculative_executor.rs
        rollback_manager.rs
        speculative_errors.rs
```

Tests for APX 16.5 live in:

```text
tests/speculative_execution_test.rs
```

---

## SpeculativePlan

File: `src/v16/speculative/speculative_plan.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativePlan {
    pub base_plan: ExecutionPlan,
    pub speculative_only: bool,
}
```

`SpeculativePlan` is a marker wrapper around an `ExecutionPlan` used for
speculative runs:

- `base_plan` – the original plan (APX 16.1).
- `speculative_only` – marks that this plan is not the primary execution path.

It does not introduce new steps or illegal actions; it is a derived view of the
existing plan.

---

## SpeculativeContext

File: `src/v16/speculative/speculative_context.rs`

```rust
#[derive(Debug)]
pub struct SpeculativeContext<R: RuntimeFacade> {
    pub label: String,
    pub context: ExecutionContext<R>,
}
```

`SpeculativeContext` wraps an `ExecutionContext<R>` together with a label
describing the speculative run. It does not add capabilities; it only helps
identify and isolate speculative executions from the main one.

---

## RollbackManager

File: `src/v16/speculative/rollback_manager.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct RollbackManager<R: Clone> {
    original_runtime: R,
}
```

`RollbackManager` provides **reversibility** for speculative execution by
keeping a snapshot of the runtime facade:

- `new(&R)` stores a cloned copy of the runtime facade.
- `rollback(&mut R)` restores the original snapshot into a runtime instance.

In tests this operates on `MockRuntime`, but the design generalizes to any
`Clone`-capable facade.

---

## Speculative errors

File: `src/v16/speculative/speculative_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum SpeculativeError {
    ContractViolation(String),
    RollbackUnavailable(String),
    ExecutionFailed(String),
}
```

`SpeculativeError` captures failure modes specific to speculative execution:

- `ContractViolation` – e.g., attempting speculation on a non-abortable plan.
- `RollbackUnavailable` – reserved for cases where rollback cannot be
  guaranteed.
- `ExecutionFailed` – speculative execution of some step failed.

Errors are descriptive and do not perform recovery actions themselves.

---

## SpeculativeExecutor

File: `src/v16/speculative/speculative_executor.rs`

```rust
#[derive(Debug)]
pub struct SpeculativeExecutor<R: RuntimeFacade + Clone> {
    pub plan: SpeculativePlan,
    pub context: ExecutionContext<R>,
    rollback: RollbackManager<R>,
}
```

`SpeculativeExecutor` is responsible for executing a plan speculatively:

- Uses a dedicated `ExecutionContext<R>` and `RollbackManager<R>`.
- Does not modify the main executor or plan.

### API

```rust
impl<R: RuntimeFacade + Clone> SpeculativeExecutor<R> {
    pub fn new(base_plan: &ExecutionPlan, context: ExecutionContext<R>) -> Self { /* ... */ }

    pub fn run(&mut self) -> Result<(), SpeculativeError> { /* ... */ }
}
```

Behavior:

- `new(...)`:
  - Wraps the `ExecutionPlan` into a `SpeculativePlan`.
  - Captures an initial snapshot of the runtime via `RollbackManager`.

- `run()`:
  - If `base_plan.globally_abortable == false`, returns
    `SpeculativeError::ContractViolation`.
  - Iterates over `base_plan.steps` and executes each step using
    `StepExecutor::execute_step`.
  - On any error:
    - Calls `rollback.rollback(&mut context.runtime)` to restore the runtime.
    - Returns `SpeculativeError::ExecutionFailed` with a descriptive message.
  - On success: returns `Ok(())`.

Speculative execution thus mirrors the real plan but always under a
rollback-capable context.

---

## Tests for APX 16.5

File: `tests/speculative_execution_test.rs`

Tests use a `MockRuntime` implementing `RuntimeFacade` to simulate effects and
verify rollback and isolation.

The test suite covers:

1. **Speculative execution is isolated**

   - A main `SafeExecutor` and a `SpeculativeExecutor` use distinct runtimes.
   - Both runtimes record calls, but they are separate instances, so modifying
     one does not affect the other.

2. **Rollback restores original state**

   - When a speculative run fails (e.g., on `select_backend_candidate`), the
     runtime after rollback matches the original snapshot taken before
     speculation.

3. **Speculative execution is deterministic**

   - Two speculative runs with identical plans and initial runtimes produce
     identical call traces and counters.

4. **Speculation never violates contract abortability**

   - If the base plan is marked as not globally abortable,
     `run()` returns `SpeculativeError::ContractViolation`.

5. **RollbackManager restores runtime snapshot**

   - Independent test for `RollbackManager` ensuring that it fully restores
     a mutated runtime to its original state.

Running APX 16.5 tests:

```bash
cargo test --test speculative_execution_test
```

All tests are deterministic, mock-only, and CI-safe.

---

## Status of APX 16.5

APX 16.5 is considered **DONE** when:

- There is real speculative execution of an `ExecutionPlan`.
- Rollback is always available and correctly restores state on failure.
- Speculative runs do not contaminate the main execution state.
- The `ExecutionContract` is never violated.
- Tests validate isolation, reversibility, and determinism.

The current implementation meets these criteria, enabling Atenia to explore
alternative paths and anticipate outcomes without compromising safety or
contract guarantees.

---

## APX 16.6 — Execution Explainability

APX 16.6 adds **action-level explainability** on top of the execution stack.
It does not execute or modify anything; it only explains how:

- Intention (APX 15 `DecisionBias`)
- Contract (16.0 `ExecutionContract`)
- Plan (16.1 `ExecutionPlan`)
- Execution (16.2–16.5 events, guards, speculation)

combined to produce what actually happened.

### Directory layout additions (16.6)

New module under `src/v16`:

```text
src/v16/
    explain/
        mod.rs
        execution_explanation.rs
        explanation_builder.rs
        explanation_formatter.rs
        explain_errors.rs
```

Tests for APX 16.6 live in:

```text
tests/execution_explain_test.rs
```

---

## ExecutionExplanation

File: `src/v16/explain/execution_explanation.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct StepExecutionExplanation {
    pub step_index: usize,
    pub description: String,
    pub guard_action: Option<GuardAction>,
    pub speculative: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionExplanation {
    pub summary: String,
    pub decision_bias: DecisionBias,
    pub contract: ExecutionContract,
    pub plan_summary: String,
    pub steps: Vec<StepExecutionExplanation>,
    pub events: Vec<ExecutionEvent>,
    pub outcome: ExecutionOutcome,
    pub speculative_plan: Option<SpeculativePlan>,
}
```

`ExecutionExplanation` is an **immutable, structured explanation** of an
execution:

- `summary` – short human-readable description.
- `decision_bias` – original policy bias from APX 15.
- `contract` – `ExecutionContract` applied.
- `plan_summary` – textual summary of the plan.
- `steps` – per-step explanations including guard actions and whether they
  belong to a speculative plan.
- `events` – the raw `ExecutionEvent`s from 16.3.
- `outcome` – final `ExecutionOutcome`.
- `speculative_plan` – optional `SpeculativePlan` used during speculation.

It does not execute or alter anything; it only describes.

---

## Explain errors

File: `src/v16/explain/explain_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ExplainError {
    MissingInformation(String),
    InconsistentEvents(String),
}
```

`ExplainError` captures issues in building explanations:

- `MissingInformation` – essential inputs (e.g. events) are missing.
- `InconsistentEvents` – events are not in logical order or are contradictory.

These errors are deterministic and never trigger recovery or execution.

---

## ExplanationBuilder

File: `src/v16/explain/explanation_builder.rs`

```rust
pub struct ExplanationBuilder;

impl ExplanationBuilder {
    pub fn build(
        decision_bias: &DecisionBias,
        contract: &ExecutionContract,
        plan_summary: String,
        events: Vec<ExecutionEvent>,
        outcome: ExecutionOutcome,
        guard_actions: Vec<(usize, GuardAction)>,
        speculative_plan: Option<SpeculativePlan>,
    ) -> Result<ExecutionExplanation, ExplainError> { /* ... */ }
}
```

`ExplanationBuilder` correlates:

- Policy bias.
- Contract.
- A textual plan summary.
- Execution events and outcome.
- Guard actions per step.
- Optional speculative plan.

Behavior:

- Validates that there are events and that their logical timestamps are
  non-decreasing; otherwise returns `ExplainError::MissingInformation` or
  `ExplainError::InconsistentEvents`.
- Builds a `StepExecutionExplanation` for each guard action:
  - `description` is derived from the `GuardAction`.
  - `speculative` is `true` when the step index is within the `SpeculativePlan`.
- Derives a `summary` string from the `ExecutionOutcomeKind`.
- Returns a fully-populated `ExecutionExplanation`.

It does not re-interpret policies or modify events; it only organizes existing
data.

---

## ExplanationFormatter

File: `src/v16/explain/explanation_formatter.rs`

Two pure formatter functions are provided:

- `format_explanation_text(&ExecutionExplanation) -> String`
- `format_explanation_json(&ExecutionExplanation) -> String`

`format_explanation_text` renders a stable human-readable summary, listing:

- Summary and outcome kind.
- Policy bias and plan summary.
- Steps with indices, guard actions, and speculative flags.
- Events with logical timestamps and step indices.

`format_explanation_json` outputs a simple JSON-like representation with
controlled field order and without any external dependencies.

Both formatters are pure and deterministic.

---

## Tests for APX 16.6

File: `tests/execution_explain_test.rs`

Tests build explanations from synthetic contracts, plans, events, guard
actions, and speculative plans using only mock data.

The test suite covers:

1. **Explanation matches execution events**

   - The explanation retains all `ExecutionEvent`s and the exact
     `ExecutionOutcome`.

2. **Explanation reflects contract constraints and guard actions**

   - For aborted executions, the outcome kind indicates aborted or partially
     completed status, and the step explanation records an `Abort` guard
     action.

3. **Explanation includes speculation when present**

   - When a `SpeculativePlan` is provided, the explanation includes it and
     marks at least one step as speculative.

4. **Explanation is deterministic**

   - Two calls to `ExplanationBuilder::build` with identical inputs produce
     identical `ExecutionExplanation` values.

5. **Formatting output is stable**

   - Repeated calls to the text and JSON formatters yield identical strings for
     the same explanation.

Running APX 16.6 tests:

```bash
cargo test --test execution_explain_test
```

All tests are deterministic, fast, and CI-safe.

---

## Status of APX 16.6

APX 16.6 is considered **DONE** when:

- A complete `ExecutionExplanation` can be constructed from bias, contract,
  plan summary, events, outcome, guard actions, and speculative plans.
- Explanations faithfully match actual events and outcomes.
- No execution or planning occurs in this layer.
- Output is stable and deterministic.
- Tests validate fidelity and determinism.

The current implementation meets these criteria, completing the APX 16 stack
with a structured, auditable explanation of how intentions from APX 15 turned
into concrete actions.

---

## APX 16.7 — Execution Replay / Deterministic Re-run

APX 16.7 adds **deterministic replay** capabilities. It does not decide new
policies, generate new plans, or optimize anything. Instead, it:

- Replays an already-recorded execution.
- Uses existing `ExecutionEvent`s (16.3) and `ExecutionPlan` (16.1).
- Validates that replay is consistent with history.

### Directory layout additions (16.7)

New module under `src/v16`:

```text
src/v16/
    replay/
        mod.rs
        execution_replay.rs
        replay_context.rs
        replay_validator.rs
        replay_errors.rs
```

Tests for APX 16.7 live in:

```text
tests/execution_replay_test.rs
```

---

## Replay errors

File: `src/v16/replay/replay_errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    MissingInformation(String),
    InconsistentHistory(String),
    DivergentOutcome(String),
}
```

`ReplayError` captures issues that prevent a safe or faithful replay:

- `MissingInformation` – required inputs (events, plan) are missing.
- `InconsistentHistory` – events conflict with the plan or with each other.
- `DivergentOutcome` – replayed execution diverges from the recorded behavior.

Errors are descriptive and do not attempt recovery.

---

## ReplayContext

File: `src/v16/replay/replay_context.rs`

```rust
#[derive(Debug)]
pub struct ReplayContext<R: RuntimeFacade> {
    pub label: String,
    pub context: ExecutionContext<R>,
}
```

`ReplayContext` wraps an `ExecutionContext<R>` used exclusively for replay:

- Provides an isolated runtime facade for deterministic re-runs.
- Does not expose or mutate the main execution context.

---

## ReplayValidator

File: `src/v16/replay/replay_validator.rs`

```rust
pub struct ReplayValidator;

impl ReplayValidator {
    pub fn validate(
        plan: &ExecutionPlan,
        events: &[ExecutionEvent],
        _outcome: &ExecutionOutcome,
    ) -> Result<(), ReplayError> { /* ... */ }
}
```

`ReplayValidator` ensures that replay inputs are consistent and safe:

- Checks that there is at least one event.
- Verifies that event `logical_timestamp`s are non-decreasing.
- Extracts `StepSucceeded` events and ensures:
  - The number of succeeded steps matches `plan.steps.len()`.
  - Step indices follow the plan order (`0, 1, 2, …`).

If these checks fail, it returns `ReplayError::MissingInformation` or
`ReplayError::InconsistentHistory`.

---

## ExecutionReplay

File: `src/v16/replay/execution_replay.rs`

```rust
pub struct ExecutionReplay<R: RuntimeFacade> {
    pub contract: ExecutionContract,
    pub plan: ExecutionPlan,
    pub events: Vec<ExecutionEvent>,
    pub outcome: ExecutionOutcome,
    pub context: ReplayContext<R>,
}
```

`ExecutionReplay` is the central component for deterministic re-runs:

- Holds the original `ExecutionContract`, `ExecutionPlan`, `ExecutionEvent`s
  and `ExecutionOutcome`.
- Uses a `ReplayContext<R>` with a dedicated runtime facade.

### API

```rust
impl<R: RuntimeFacade> ExecutionReplay<R> {
    pub fn new(
        contract: ExecutionContract,
        plan: ExecutionPlan,
        events: Vec<ExecutionEvent>,
        outcome: ExecutionOutcome,
        context: ReplayContext<R>,
    ) -> Self { /* ... */ }

    pub fn replay(&mut self) -> Result<(), ReplayError> { /* ... */ }
}
```

Behavior:

- `new(...)` stores all replay inputs and the isolated `ReplayContext`.
- `replay()`:
  - Runs `ReplayValidator::validate` on `plan`, `events`, and `outcome`.
  - Derives the sequence of step indices from `StepSucceeded` events.
  - Replays each corresponding step in the same order using `StepExecutor` and
    the isolated context.
  - On any failure, returns `ReplayError::DivergentOutcome`.

This component does not change the contract, plan, or recorded events/outcome;
it only re-executes steps in a separate context.

---

## Tests for APX 16.7

File: `tests/execution_replay_test.rs`

Tests construct contracts, plans, events, and outcomes using the existing v16
stack, then use a `MockRuntime` to run replays.

The test suite covers:

1. **Replay reproduces original execution order and outcome**

   - Events and outcome inside `ExecutionReplay` remain identical to the
     originals after replay.

2. **Replay is deterministic**

   - Two replays with the same inputs produce identical runtime call traces.

3. **Replay aborts on inconsistent history**

   - Tampering with event timestamps yields `ReplayError::InconsistentHistory`.

4. **Replay has no side-effects on inputs**

   - Contract, plan, events, and outcome stored in `ExecutionReplay` remain
     equal to their pre-replay clones.

Running APX 16.7 tests:

```bash
cargo test --test execution_replay_test
```

All tests are deterministic, mock-only, and CI-safe.

---

## Status of APX 16.7

APX 16.7 is considered **DONE** when:

- An execution can be replayed faithfully from recorded events.
- Replay detects inconsistencies in history and aborts safely.
- No new decisions are made and no plans are modified.
- Replay has no side effects on the original data or main runtime.
- Tests validate fidelity, determinism, and isolation.

The current implementation meets these criteria, completing the APX 16 stack
with reproducible, auditable re-runs of past executions.
