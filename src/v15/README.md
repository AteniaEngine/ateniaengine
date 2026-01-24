# APX v15 — Execution Policy Layer

APX v15 is an **isolated, declarative policy layer** that lives entirely under `src/v15` and is tested from the crate-level `tests` directory. It does **not** depend on the engine runtime, hardware backends, or previous APX versions.

APX 15.0 defines the **formal base layer** on which future policy-related features (15.1+, 16.x, …) can build.

- No direct access to hardware
- No real evidence inputs yet
- No user-preference modeling yet
- No side effects, no state

Everything is expressed as **pure policies** that map abstract context into structured decision biases.

---

## Directory layout (v15)

Location: `src/v15`

```text
src/
└── v15/
    ├── mod.rs
    └── policy/
        ├── mod.rs
        ├── policy.rs
        ├── types.rs
        ├── registry.rs
        └── builtin/
            ├── mod.rs
            ├── stability_first.rs
            ├── throughput_first.rs
            ├── latency_first.rs
            ├── constrained_hardware.rs
            └── power_saving.rs
```

Tests for APX v15.0 live in:

```text
tests/policy_layer_test.rs
```

This keeps APX v15 **cold-testable** and versionable, and allows future APX 16 to consume v15 without tight coupling.

---

## Core concepts (APX 15.0)

### `PolicyInput`

File: `src/v15/policy/types.rs`

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyInput {
    // APX 15.0: intentionally empty. Future versions will extend this.
}
```

For APX 15.0, `PolicyInput` is an **empty context**. It exists so that later versions (15.1+: evidence, 15.2+: user preference, etc.) can extend it without changing the basic contract.

- No hardware details
- No real evidence
- No user data

### `DecisionBias`

File: `src/v15/policy/types.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionBias {
    pub risk_weight: f32,
    pub latency_weight: f32,
    pub stability_weight: f32,
    pub memory_pressure_weight: f32,
    pub offload_cost_weight: f32,
}
```

`DecisionBias` is a **pure data structure** that encodes the relative importance of several high-level criteria:

- `risk_weight`
- `latency_weight`
- `stability_weight`
- `memory_pressure_weight`
- `offload_cost_weight`

Rules for APX 15.0:

- All values are normalized in the range `[0.0, 1.0]`.
- No internal logic; this is not a planner, only a container for weights.

For testability, we add a helper:

```rust
impl DecisionBias {
    pub fn is_normalized(&self) -> bool {
        fn in_range(v: f32) -> bool {
            (0.0..=1.0).contains(&v)
        }

        in_range(self.risk_weight)
            && in_range(self.latency_weight)
            && in_range(self.stability_weight)
            && in_range(self.memory_pressure_weight)
            && in_range(self.offload_cost_weight)
    }
}
```

This is used only by tests to assert that policies respect the normalization contract.

---

## ExecutionPolicy trait

File: `src/v15/policy/policy.rs`

```rust
use super::types::{DecisionBias, PolicyInput};

pub trait ExecutionPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias;
}
```

This is the **base trait** for all APX 15 policies.

Properties:

- `Send + Sync` so policies can be safely shared across threads.
- `name()` returns a static identifier used by the registry.
- `evaluate()` is pure: for APX 15.0 it only depends on its arguments and internal constants.

Design constraints in 15.0:

- No mutation of external state
- No I/O or hardware calls
- No randomness
- No engine/runtime access

The result is that policies are **interchangeable**, purely by their outputs.

---

## Policy registry

File: `src/v15/policy/registry.rs`

```rust
use std::collections::HashMap;
use std::sync::Arc;

use super::policy::ExecutionPolicy;

pub struct PolicyRegistry {
    policies: HashMap<&'static str, Arc<dyn ExecutionPolicy>>, 
}
```

`PolicyRegistry` is a thin, explicit registry of available policies.

Public API:

```rust
impl PolicyRegistry {
    pub fn new() -> Self { /* ... */ }

    pub fn register<P>(&mut self, policy: P)
    where
        P: ExecutionPolicy + 'static,
    { /* ... */ }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ExecutionPolicy>> { /* ... */ }

    pub fn list(&self) -> Vec<&'static str> { /* ... */ }
}
```

Key points:

- Policies are stored as `Arc<dyn ExecutionPolicy>` for cheap cloning and shared ownership.
- `register()` is explicit; tests and future layers decide which policies to register.
- `list()` returns a sorted list of policy names for deterministic behavior.

No hidden behavior, no dynamic discovery, no side effects.

---

## Built-in policies (APX 15.0)

Location: `src/v15/policy/builtin/`

All built-in policies:

- Are **stateless**
- Return **fixed** `DecisionBias` values
- Use names that match their conceptual intent

Files:

- `stability_first.rs`
- `throughput_first.rs`
- `latency_first.rs`
- `constrained_hardware.rs`
- `power_saving.rs`

Each file defines a struct and implements `ExecutionPolicy`.

Example: `stability_first` (simplified)

```rust
use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

pub struct StabilityFirstPolicy;

impl ExecutionPolicy for StabilityFirstPolicy {
    fn name(&self) -> &'static str {
        "stability_first"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.1,
            latency_weight: 0.4,
            stability_weight: 1.0,
            memory_pressure_weight: 0.7,
            offload_cost_weight: 0.6,
        }
    }
}
```

Other policies follow the same pattern with different weight combinations, for example:

- **throughput_first**: favors latency/throughput
- **latency_first**: aggressively prioritizes latency with medium stability
- **constrained_hardware**: focuses on memory pressure and safer configurations
- **power_saving**: encodes a bias toward lower resource usage (higher offload cost weight)

None of these policies mention concrete devices (CPU/GPU) in the code; they only manipulate normalized weights.

---

## Tests for APX 15.0

File: `tests/policy_layer_test.rs`

The v15 layer is tested from the crate-level `tests` directory using:

```rust
#[path = "../src/v15/mod.rs"]
mod v15;
```

This pattern allows APX v15 to be compiled and tested **without modifying** `src/lib.rs`.

The test suite currently covers:

1. **Determinism**

   - For each built-in policy, the same `PolicyInput` yields the same `DecisionBias` every time.
   - `DecisionBias` values are asserted to be normalized (all weights in `[0.0, 1.0]`).

2. **Stability / no side effects**

   - Repeated calls to `evaluate()` on the same policy instance produce identical outputs.
   - There is no shared mutable state that changes across evaluations.

3. **Registry behavior**

   - All built-in policies are registered into a `PolicyRegistry`.
   - `list()` returns exactly the expected set of names.
   - `get(name)` returns a policy that produces a normalized `DecisionBias`.

Running only the APX 15.0 tests:

```bash
cargo test --test policy_layer_test
```

---

## Status of APX 15.0

APX 15.0 is considered **DONE** when:

- There is a clear abstraction for execution policies (`ExecutionPolicy`, `PolicyInput`, `DecisionBias`).
- Policies are interchangeable and discoverable via `PolicyRegistry`.
- There is no dependency on APX 14 or the engine runtime.
- No real evidence, hardware, or user data is consumed yet.
- Tests validate determinism, purity, registry coverage, and normalized outputs.

The current implementation satisfies these criteria and provides a stable base for:

- APX 15.1 (evidence-aware policies)
- APX 15.2 (user preference modeling)
- APX 15.3+ (policy switching, explanations, simulations) on top of this layer.

---

## APX 15.1 — Evidence-Aware Policies

APX 15.1 extends the v15 policy layer so that policies can **read passive evidence** produced by
other subsystems (e.g., APX 14), without executing actions or touching runtime state.

Design guarantees:

- Policies remain **pure** and **deterministic**.
- Evidence is **read-only** and fully **immutable** from the policy’s point of view.
- No tensor moves, no hardware access, no runtime calls.

APX 15.1 connects observability (APX 14) with intention (v15 policies) via structured snapshots.

### Directory layout additions (15.1)

New module under `src/v15/policy`:

```text
src/v15/policy/
    evidence/
        mod.rs
        snapshot.rs
        signals.rs
```

Tests for APX 15.1 live in:

```text
tests/evidence_policy_test.rs
```

This keeps the evidence layer cold-testable and independent of the engine runtime.

---

## Evidence signals (`PolicySignal*`)

File: `src/v15/policy/evidence/signals.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PolicySignalKind {
    RecentRecovery,
    HighMemoryPressure,
    FragmentationWarning,
    StableLatency,
    PreOomSignal,
}

/// A single, passive evidence signal derived from lower layers (e.g. APX 14).
/// All scores are normalized in [0.0, 1.0].
#[derive(Debug, Clone, PartialEq)]
pub struct PolicySignal {
    pub kind: PolicySignalKind,
    /// Normalized severity or strength of the signal.
    pub score: f32,
}

impl PolicySignal {
    pub fn is_normalized(&self) -> bool {
        (0.0..=1.0).contains(&self.score)
    }
}
```

Key properties:

- Signals are **descriptive only**; no actions or side effects.
- `score` is normalized to `[0.0, 1.0]`.
- Variants cover common patterns such as recoveries, memory pressure, fragmentation, latency
  stability, and pre-OOM risk.

---

## Evidence snapshots (`PolicyEvidenceSnapshot`)

File: `src/v15/policy/evidence/snapshot.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyEvidenceSnapshot {
    pub signals: Vec<PolicySignal>,
}

impl PolicyEvidenceSnapshot {
    pub fn new(signals: Vec<PolicySignal>) -> Self {
        Self { signals }
    }

    pub fn all_signals(&self) -> &[PolicySignal] {
        &self.signals
    }

    pub fn is_normalized(&self) -> bool {
        self.signals.iter().all(|s| s.is_normalized())
    }
}
```

`PolicyEvidenceSnapshot` is an **immutable, clonable snapshot** of passive evidence:

- No references to live runtime objects.
- No mutation or caching inside policies.
- Suitable for offline replay and auditing.

It aggregates multiple `PolicySignal` instances into a single, self-contained structure passed
into policies.

---

## Evidence-aware `ExecutionPolicy`

File: `src/v15/policy/policy.rs`

```rust
pub trait ExecutionPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    /// Base evaluation that does not consider evidence (APX 15.0 behavior).
    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias;

    /// Evidence-aware evaluation (APX 15.1).
    ///
    /// Default implementation falls back to the 15.0 behavior when no
    /// evidence is required, preserving existing policies.
    fn evaluate_with_evidence(
        &self,
        input: &PolicyInput,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> DecisionBias {
        let _ = evidence;
        self.evaluate(input)
    }
}
```

This extension keeps APX 15.0 semantics as the default:

- Policies that do not care about evidence can ignore it and still implement only `evaluate()`.
- Evidence is passed as an `Option<&PolicyEvidenceSnapshot>` and is always read-only.

The conceptual signature becomes:

```text
(PolicyInput, PolicyEvidenceSnapshot?) -> DecisionBias
```

without executing any actions or touching hardware.

---

## Example: stability-first policy reacting to pre-OOM evidence

File: `src/v15/policy/builtin/stability_first.rs`

```rust
impl ExecutionPolicy for StabilityFirstPolicy {
    fn name(&self) -> &'static str {
        "stability_first"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.1,
            latency_weight: 0.4,
            stability_weight: 1.0,
            memory_pressure_weight: 0.7,
            offload_cost_weight: 0.6,
        }
    }

    fn evaluate_with_evidence(
        &self,
        input: &PolicyInput,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> DecisionBias {
        let mut bias = self.evaluate(input);

        if let Some(snapshot) = evidence {
            let mut max_pre_oom_score = 0.0f32;

            for signal in snapshot.all_signals() {
                if matches!(signal.kind, PolicySignalKind::PreOomSignal) {
                    if signal.score > max_pre_oom_score {
                        max_pre_oom_score = signal.score;
                    }
                }
            }

            if max_pre_oom_score > 0.0 {
                let adjust = 0.2 * max_pre_oom_score;

                // Increase stability and reduce risk in the presence of
                // stronger pre-OOM evidence, clamping to [0.0, 1.0].
                bias.stability_weight = (bias.stability_weight + adjust).min(1.0);
                bias.risk_weight = (bias.risk_weight - adjust).max(0.0);
            }
        }

        bias
    }
}
```

Behavior:

- With stronger `PreOomSignal`, `stability_weight` increases and `risk_weight` decreases.
- No actions are triggered; only the **decision bias** is adjusted.
- The function remains deterministic for a given input and snapshot.

Other built-in policies can later be extended in a similar pattern.

---

## Tests for APX 15.1

File: `tests/evidence_policy_test.rs`

The evidence-aware layer is tested from the crate-level `tests` directory using
`#[path = "../src/v15/mod.rs"]` to import v15 without modifying `src/lib.rs`.

The test suite currently covers:

1. **Policy reacts to evidence**

   - Same policy + different `PolicyEvidenceSnapshot` → different `DecisionBias`.
   - For `StabilityFirstPolicy`, stronger `PreOomSignal` implies higher stability bias and/or
     lower risk bias.

2. **No-evidence fallback**

   - `evaluate_with_evidence(input, None)` is identical to `evaluate(input)`.
   - This guarantees that APX 15.1 does not break APX 15.0 semantics.

3. **Read-only guarantee**

   - The snapshot passed into `evaluate_with_evidence` is cloned before the call and compared
     after repeated evaluations to ensure it is not modified.

4. **Determinism and purity**

   - Repeated calls to `evaluate_with_evidence` with the same `PolicyInput` and snapshot
     produce identical `DecisionBias`.
   - No state is cached or mutated across calls.

Running APX 15.1 tests together with APX 15.0 tests:

```bash
cargo test --test policy_layer_test --test evidence_policy_test
```

All these tests are designed to be deterministic, fast, and CI-safe, with no external
dependencies or runtime side effects.

---

## APX 15.2 — Safe User Preferences

APX 15.2 adds a **human preference layer** on top of policies (15.0) and evidence (15.1).
User preferences are modeled as **soft constraints** that can bias `DecisionBias`, but they:

- Never become hard orders.
- Never force specific backends or disable safety mechanisms.
- Are always bounded, optional, and overridable by risk.

The guiding principle is: *the user suggests, Atenia decides*.

### Directory layout additions (15.2)

New module under `src/v15/policy`:

```text
src/v15/policy/
    preferences/
        mod.rs
        user_preferences.rs
        preference_weights.rs
```

Tests for APX 15.2 live in:

```text
tests/user_preference_test.rs
```

---

## User preferences (`UserPreferences`)

File: `src/v15/policy/preferences/user_preferences.rs`

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserPreferences {
    pub prefer_latency: bool,
    pub avoid_ssd: bool,
    pub prioritize_stability: bool,
    pub minimize_power: bool,
    pub prefer_gpu: bool,
}
```

These flags represent **high-level intentions**, not hardware-specific commands:

- `prefer_latency` – the user is willing to bias towards lower latency.
- `avoid_ssd` – the user prefers to avoid heavy reliance on slower tiers.
- `prioritize_stability` – favor more conservative, stable behavior.
- `minimize_power` – prefer less aggressive, less resource-intensive choices.
- `prefer_gpu` – abstract signal that the user is more willing to pay offload cost
  for performance (without naming GPU/CPU explicitly).

None of these preferences:

- Directly select devices.
- Disable recovery or safety.
- Override core policies.

---

## Applying preferences (`apply_user_preferences`)

File: `src/v15/policy/preferences/preference_weights.rs`

```rust
pub fn apply_user_preferences(
    base: &DecisionBias,
    prefs: &UserPreferences,
    evidence: Option<&PolicyEvidenceSnapshot>,
) -> DecisionBias { /* ... */ }
```

This function is **pure** and implements the final step in the logical chain:

```text
policy_bias
  → evidence_adjusted_bias
      → user_preference_adjusted_bias
          → final_bias
```

Key behaviors:

1. **Start from base bias**

   - `base` already reflects policy + evidence.
   - Preferences only apply small, bounded deltas on top.

2. **Aggregate risk from evidence**

   - Computes a scalar `risk_score` from `PolicyEvidenceSnapshot` by scanning signals:
     - `PreOomSignal`, `RecentRecovery` → strong contribution.
     - `HighMemoryPressure`, `FragmentationWarning` → medium contribution.
     - `StableLatency` → does not increase risk.

3. **Safety dominance**

   - If `risk_score >= 0.7` → **preferences are ignored entirely**, returning `base`.
   - If `0.3 <= risk_score < 0.7` → preferences are attenuated (scaled by 0.5).
   - If `risk_score < 0.3` → full preference effect (scale 1.0).

4. **Soft, bounded adjustments**

   - Adjustments are small and clamped into `[0.0, 1.0]`.
   - Examples:

     - `prefer_latency` increases `latency_weight` slightly, but **never reduces**
       `stability_weight` directly.
     - `prioritize_stability` increases `stability_weight` and decreases `risk_weight` a bit.
     - `minimize_power` reduces `offload_cost_weight` and slightly reduces `latency_weight`.
     - `prefer_gpu` is interpreted as more willingness to pay offload cost (without naming
       a specific backend) by modestly reducing `offload_cost_weight`.
     - `avoid_ssd` increases `memory_pressure_weight` and `risk_weight` slightly.

   - In all cases, preferences **modulate** the bias; they never force extreme values or
     violate safety constraints.

5. **No preference fallback**

   - When `UserPreferences::default()` (all flags false), the output is always exactly `base`,
     regardless of evidence.

---

## Tests for APX 15.2

File: `tests/user_preference_test.rs`

The preference layer is tested independently, using synthetic biases and evidence snapshots.

The test suite covers:

1. **Preferences are soft**

   - With multiple preferences enabled, all output weights remain within `[0.0, 1.0]` and
     do not saturate purely due to preferences.

2. **Preferences are overridable by risk**

   - With very high `PreOomSignal` risk, applying preferences yields a bias identical to `base`.

3. **No-preference fallback**

   - With `UserPreferences::default()`, the adjusted bias equals `base` both with and without
     evidence.

4. **Determinism**

   - Repeated calls to `apply_user_preferences` with the same inputs produce identical
     `DecisionBias`.

5. **Safety dominance under conflict**

   - When a user prefers latency but evidence indicates medium risk, the adjusted bias respects:
     - `stability_weight >= base.stability_weight`
     - `risk_weight <= base.risk_weight`

Running APX 15.2 tests:

```bash
cargo test --test user_preference_test
```

These tests ensure that user intent influences the final bias, but **never** overrides safety
or breaks the guarantees of APX 15.0 and 15.1.

---

## APX 15.3 — Runtime Policy Switching

APX 15.3 introduces a **runtime policy switching layer** that controls which execution policy is
currently active, without recompiling, restarting, or re-evaluating past decisions.

Key properties:

- Switching **only** changes the active policy (the decision lens).
- Evidence, history, user preferences, and runtime state remain intact.
- No hot reload, no scripting, no persistence; purely in-memory orchestration.

The intent is to support scenarios like:

- Training → `throughput_first`.
- Inference → `latency_first`.
- Unstable environments → `stability_first`.

All without touching the underlying engine or invalidating prior state.

### Directory layout additions (15.3)

New module under `src/v15/policy`:

```text
src/v15/policy/
    manager/
        mod.rs
        policy_manager.rs
```

Tests for APX 15.3 live in:

```text
tests/policy_switch_test.rs
```

---

## PolicyManager

File: `src/v15/policy/manager/policy_manager.rs`

```rust
pub struct PolicyManager {
    registry: PolicyRegistry,
    active_name: String,
}
```

The `PolicyManager` is the **single authority** for which policy is currently active. It does
**not**:

- Evaluate policies.
- Touch runtime or hardware.
- Inspect evidence or preferences.

It only orchestrates **which** `ExecutionPolicy` is used by higher layers.

### API

```rust
impl PolicyManager {
    pub fn new(registry: PolicyRegistry, initial_policy: &str) -> Self { /* ... */ }

    pub fn active_policy_name(&self) -> &str { /* ... */ }

    pub fn active_policy(&self) -> Arc<dyn ExecutionPolicy> { /* ... */ }

    pub fn list_available_policies(&self) -> Vec<&'static str> { /* ... */ }

    pub fn set_active_policy(&mut self, name: &str) -> bool { /* ... */ }
}
```

Behavior and guarantees:

- `new(...)` requires the initial policy to exist in the `PolicyRegistry`.
- `active_policy_name()` exposes the current active policy name.
- `active_policy()` returns an `Arc<dyn ExecutionPolicy>` for read-only use by callers.
- `list_available_policies()` returns a deterministic list of policy names.
- `set_active_policy(name)`:
  - If `name` exists in the registry, switches atomically and returns `true`.
  - If `name` does not exist, leaves the active policy unchanged and returns `false`.

There is no hidden state and no implicit fallback logic; all behavior is explicit.

---

## Tests for APX 15.3

File: `tests/policy_switch_test.rs`

The policy switching layer is tested using a registry populated with the built-in policies
(`stability_first`, `throughput_first`, `latency_first`).

The test suite covers:

1. **Initial policy is defined**

   - A `PolicyManager` created with a valid initial policy name reports that policy as active.

2. **Switch changes active policy**

   - Calling `set_active_policy` with a valid name updates the active policy.

3. **Invalid switch is ignored**

   - Switching to a non-existent policy name returns `false` and leaves the active policy
     unchanged.

4. **Switch does not reset state**

   - Sampling the bias from the active policy before and after a sequence of switches that
     returns to the original policy yields identical `DecisionBias`.
   - This shows that switching via `PolicyManager` does not reset or disturb policy behavior.

5. **Determinism under repeated switches**

   - Two `PolicyManager` instances built from identical registries and subjected to the same
     sequence of policy names end up in the same final state.

Running APX 15.3 tests:

```bash
cargo test --test policy_switch_test
```

These tests ensure that the active policy can be changed at runtime in a way that is explicit,
safe, and deterministic, without affecting existing engine behavior, evidence, or user
preferences.

---

## APX 15.4 — Policy Explainability

APX 15.4 adds an **explanation layer** on top of the policy engine. It does not change
decisions or weights; it only produces structured, deterministic explanations of why a given
`DecisionBias` was produced in the context of:

- The active policy (15.0).
- Evidence snapshots (15.1).
- User preferences (15.2).
- The current switching state (15.3).

Key guarantees:

- Explanations are **read-only views**.
- They do **not** modify `DecisionBias` or any state.
- Output is deterministic and auditable.

### Directory layout additions (15.4)

New module under `src/v15/policy`:

```text
src/v15/policy/
    explain/
        mod.rs
        explanation.rs
        formatter.rs
```

Tests for APX 15.4 live in:

```text
tests/policy_explain_test.rs
```

---

## PolicyExplanation

File: `src/v15/policy/explain/explanation.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PreferenceStatus {
    Applied,
    IgnoredDueToRisk,
    Inactive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceExplanation {
    pub name: &'static str,
    pub status: PreferenceStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalExplanation {
    pub kind: PolicySignalKind,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyExplanation {
    pub policy_name: String,
    pub final_bias: DecisionBias,
    pub considered_signals: Vec<SignalExplanation>,
    pub preference_explanations: Vec<PreferenceExplanation>,
    pub notes: Vec<String>,
}
```

`PolicyExplanation` is a **structured description** of why a particular `DecisionBias` was
produced. It can include:

- The policy name.
- The final bias (exact `DecisionBias`).
- Which evidence signals were considered.
- For each user preference: whether it was applied, inactive, or ignored due to risk.
- Notes summarizing whether preferences influenced the final bias or were overridden.

`PolicyExplanation::from_bias_and_context(...)` builds an explanation from:

- `policy_name`
- `final_bias`
- `base_bias_before_prefs` (bias before applying preferences)
- `UserPreferences`
- Optional `PolicyEvidenceSnapshot`

It mirrors the risk aggregation logic used in 15.2 to determine when preferences are ignored
due to high risk, but it does **not** reapply or recompute the bias.

---

## Explain formatters

File: `src/v15/policy/explain/formatter.rs`

Two pure formatter functions are provided:

- `format_explanation_text(&PolicyExplanation) -> String`
- `format_explanation_json(&PolicyExplanation) -> String`

`format_explanation_text` renders a stable, human-readable multiline string with:

- `policy_name`
- `final_bias` weights
- A list of signals (`kind`, `score`)
- A list of preferences and their `PreferenceStatus`
- A list of notes

`format_explanation_json` manually serializes a `PolicyExplanation` to JSON, controlling:

- Field order.
- Floating-point formatting (fixed precision).
- String escaping via a local helper.

Neither formatter reads external state, uses timestamps, or performs I/O.

---

## Tests for APX 15.4

File: `tests/policy_explain_test.rs`

The explainability layer is tested using synthetic biases, evidence, and preferences to remain
fully isolated from the runtime.

The test suite covers:

1. **Explanation matches bias**

   - `PolicyExplanation::from_bias_and_context` preserves the exact `final_bias` and
     `policy_name` passed in.

2. **Explanation is deterministic**

   - Repeated calls with identical inputs produce identical `PolicyExplanation` values.
   - Both text and JSON formatter outputs are identical across calls.

3. **No side-effects**

   - Inputs (`UserPreferences`, `PolicyEvidenceSnapshot`, base and final biases) remain
     unchanged after building an explanation.

4. **Ignored preferences are explained**

   - With high-risk evidence (e.g., strong `PreOomSignal`), active preferences are marked as
     `IgnoredDueToRisk`.

5. **Formatter output is stable**

   - Calling the text and JSON formatters multiple times on the same explanation yields
     identical strings.

Running APX 15.4 tests:

```bash
cargo test --test policy_explain_test
```

These tests ensure that every bias can be explained in a reproducible, auditable way without
changing engine behavior, policy logic, or decision weights.

---

## APX 15.5 — Policy Simulation (What-if Analysis)

APX 15.5 adds a **policy simulation layer** that can evaluate multiple policies in parallel
for the same inputs, evidence, and user preferences, **without changing** the active policy
or touching runtime state.

The goal is to support "what-if" questions such as:

- What would `latency_first` do under the same conditions as `stability_first`?
- How would preferences affect the decision if we used `throughput_first` instead?

All simulations are performed in a **pure, read-only** way and return structured results that
can be compared or inspected offline.

### Directory layout additions (15.5)

New module under `src/v15/policy`:

```text
src/v15/policy/
    simulation/
        mod.rs
        simulator.rs
        simulation_result.rs
```

Tests for APX 15.5 live in:

```text
tests/policy_simulation_test.rs
```

---

## SimulationResult

File: `src/v15/policy/simulation/simulation_result.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationResult {
    pub policy_name: String,
    pub bias: DecisionBias,
    pub explanation: PolicyExplanation,
    pub more_conservative_than_base: bool,
    pub more_aggressive_than_base: bool,
}
```

`SimulationResult` is the **unit of output** for policy simulations. For each simulated
policy it records:

- `policy_name` – name of the simulated policy.
- `bias` – final `DecisionBias` after policy, evidence, and preferences.
- `explanation` – `PolicyExplanation` describing how this bias was produced.
- `more_conservative_than_base` – comparison flag against a base bias.
- `more_aggressive_than_base` – comparison flag against a base bias.

In APX 15.5, the base bias is the bias of the **first** policy in the simulation list.

The constructor derives the comparison flags from the base bias as follows:

- A policy is considered **more conservative** than the base if:
  - `stability_weight >= base.stability_weight`, and
  - `risk_weight <= base.risk_weight`.

- A policy is considered **more aggressive** than the base if:
  - `latency_weight > base.latency_weight`, and
  - `risk_weight >= base.risk_weight`.

These rules provide a simple, deterministic way to compare simulated policies along the
stability/latency axis without encoding any runtime-specific semantics.

---

## PolicySimulator

File: `src/v15/policy/simulation/simulator.rs`

```rust
pub struct PolicySimulator;

impl PolicySimulator {
    pub fn simulate_for_policies(
        registry: &PolicyRegistry,
        policy_names: &[&str],
        input: &PolicyInput,
        prefs: &UserPreferences,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> Vec<SimulationResult> { /* ... */ }
}
```

`PolicySimulator` is a **pure, stateless helper** that runs simulations for a list of
policy names against the same logical context:

```text
(PolicyInput, PolicyEvidenceSnapshot?, UserPreferences) × {policy names} → [SimulationResult]
```

Behavior and guarantees:

- For each `policy_name` in `policy_names`:
  - Looks up the policy from the provided `PolicyRegistry`.
  - Computes a base bias via `ExecutionPolicy::evaluate_with_evidence`.
  - Applies `apply_user_preferences` to obtain the final bias.
  - Builds a `PolicyExplanation::from_bias_and_context` with the same inputs.
  - Constructs a `SimulationResult`, using the bias of the first simulated policy as the
    base for comparison flags.

- If a policy name does not exist in the registry, it is simply skipped; the function
  returns results only for resolvable policies.

- The simulator **does not**:
  - Modify the `PolicyRegistry`.
  - Touch `PolicyManager` or change the active policy.
  - Modify `PolicyInput`, `UserPreferences`, or `PolicyEvidenceSnapshot`.
  - Perform I/O or access hardware/runtime state.

The function is deterministic: the same registry, list of names, inputs, evidence, and
preferences always produce the same ordered `Vec<SimulationResult>`.

---

## Tests for APX 15.5

File: `tests/policy_simulation_test.rs`

The simulation layer is tested by combining the existing pieces of the policy stack
without any runtime integration.

The test suite covers:

1. **Multiple policy evaluation**

   - Calling `simulate_for_policies` with several policy names returns one
     `SimulationResult` per valid policy.
   - Each result carries the correct `policy_name` and a normalized `DecisionBias`.

2. **Simulation does not change the active policy**

   - A `PolicyManager` is used only to assert the active policy name before and
     after running simulations.
   - The simulator uses its own `PolicyRegistry` and does not interact with the
     manager, so the active policy remains unchanged.

3. **Determinism**

   - Running the same simulation twice with cloned inputs, preferences, evidence,
     and registry yields identical `Vec<SimulationResult>`.

4. **Comparable outputs**

   - All `SimulationResult` instances produced from the same inputs share
     consistent `considered_signals` and preference explanations, making them
     directly comparable across policies.

5. **No side effects**

   - The tests clone `PolicyInput`, `UserPreferences`, `PolicyEvidenceSnapshot`, and
     the `PolicyRegistry` before simulation and verify that they remain unchanged
     afterwards.

Running APX 15.5 tests:

```bash
cargo test --test policy_simulation_test
```

These tests ensure that policy simulation is a **pure, read-only what-if tool** that
respects the guarantees of APX 15.0–15.4 while enabling safe comparison between
alternative policies.
