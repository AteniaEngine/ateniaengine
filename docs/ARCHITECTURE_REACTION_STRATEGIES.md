# Architecture: Reaction Strategies

Status: design principle established, implementation planned for APX v21+.
Relevant prior work: SignalBus (v19), Guards (v16), Policies (v15).

## Core principle

**Execution resilience through intelligent path switching, not through
redundant execution.**

When Atenia detects a condition that threatens an in-flight execution
(memory pressure, failure accumulation, latency spikes, etc.), the
engine switches to a different execution path instead of aborting or
maintaining a redundant parallel execution.

This principle is grounded in a broader Atenia commitment: an
execution should not break for reasons that could be avoided by
choosing a different route, as long as the route change preserves
computational semantics.

## What this principle rejects

### Speculative execution with materialized alternatives

A common pattern in mission-critical systems is to keep two full
executions alive in parallel — primary and shadow. If the primary
fails, the shadow becomes primary.

Atenia rejects this pattern because the memory cost is prohibitive
for the hardware targets the engine is designed for (consumer and
prosumer machines with limited VRAM). Holding two fully-materialized
execution plans in memory would mean the engine itself is the
dominant memory consumer, which contradicts its purpose.

### Re-planning from scratch on failure

Another common approach is to let execution fail, catch the failure,
compute a new plan from scratch, and restart.

Atenia rejects this pattern because the latency of re-planning on
the critical path is unacceptable. Switching mid-execution should
take microseconds of decision cost, not seconds of plan
reconstruction.

### Static graceful degradation

A third pattern is to reduce aggressiveness uniformly under pressure
(smaller batches, lower precision, CPU fallback) without choosing
intelligently between alternatives.

Atenia rejects this pattern because it sacrifices opportunity.
A well-chosen path switch can preserve throughput better than
blanket degradation.

## What this principle accepts

**Pre-computed decision tree, materialized just-in-time.**

Before execution starts, Atenia constructs a decision tree that maps
observable runtime states to execution strategies. The strategies are
*decided* in advance — they are recipes, not materialized plans.

When a signal triggers during execution, the engine consults the tree
(microsecond cost), selects the strategy, and materializes it
just-in-time: allocates new tensors, moves existing ones, switches
backend, reduces batch, etc.

No duplicate plans live in memory. The cost of "having alternatives
ready" is the cost of having decided about them, not the cost of
having materialized them.

## Design decisions

The following decisions are current as of this document. They may
be revised as implementation proceeds toward v21+.

### Decision tree lifecycle: event-driven

The tree is constructed when execution starts, using the current
state reported by SignalBus. It remains fixed during stable conditions.
It is invalidated and reconstructed only when the SignalBus reports a
material change in state (not on a timer, not per operation).

This combines the determinism of per-execution construction with the
adaptability of continuous reconstruction, while avoiding arbitrary
polling intervals.

### Strategy scoring: single-score comparison

Each candidate strategy produces a single scalar score in `[0.0, 1.0]`
given the current state. The engine selects the highest-scoring
strategy. Multi-dimensional scoring (separate axes for safety, speed,
memory cost, etc.) is deferred; a scalar is sufficient for the initial
implementation and does not preclude a later migration.

### Strategy source: policies plus extensible trait

The built-in strategies of the engine correspond to the Policies
already defined in v15 (StabilityFirstPolicy, ThroughputFirstPolicy,
LatencyFirstPolicy, PowerSavingPolicy, ConstrainedHardwarePolicy).
Each policy is an execution strategy in this model.

External code can implement the `ExecutionStrategy` trait in its own
crate to extend the set of available strategies. This is a lightweight
extensibility mechanism: no dynamic loading, no plugin ABI, no
sandboxing. Advanced users who want custom behavior link their own
crate at compile time.

### Placement in the stack: new layer above Policies and Guards

The reaction system is a new layer that consumes the outputs of the
existing v15 Policies and v16 Guards. It does not replace them.

    Signals (SignalBus)
        ↓
    Guards (v16): "are we in crisis?" → GuardAction
        ↓
    Policies (v15): "what do we prioritize?" → DecisionBias
        ↓
    Decision Tree (v21+): "given both, which strategy?" → ExecutionStrategy
        ↓
    Just-in-time materialization: apply strategy to current plan

Guards remain imperative evaluators. Policies remain bias sources.
The decision tree is the orchestrator that composes them into a
strategy choice.

## Implications for existing code

None today. All v15 and v16 code remains unchanged. The SignalBus
already produces the inputs the decision tree will consume
(`GuardConditions` and `PolicyEvidenceSnapshot`).

The first code change to implement this architecture belongs to
v21, which will:

1. Define the `ExecutionStrategy` trait.
2. Implement built-in strategies as adapters over existing v15 Policies.
3. Define the `DecisionTree` that composes Guards + Policies into a
   strategy selection.
4. Wire the event-driven reconstruction to SignalBus state changes.

## Open items

- **Scoring function design:** how each strategy computes its score
  from current state. Probably policy-specific — StabilityFirstPolicy
  scores higher under high memory pressure, ThroughputFirstPolicy
  scores higher under low pressure, etc. To be specified in v21.

- **Materialization primitives:** what atomic operations the engine
  exposes for strategies to compose (offload tensor, reduce batch,
  switch backend, etc.). Depends on the maturity of model loading
  (v20) and execution integration.

- **Strategy conflict resolution:** what happens when two strategies
  tie on score. Current intuition: stable tiebreaker by priority
  order of built-in strategies. Non-critical until observed in practice.

- **Rollback of a triggered strategy:** if a strategy fails to
  materialize (e.g. tried to offload to RAM but RAM is also tight),
  what is the fallback. Deferred until first concrete strategies exist.

## Non-goals

- This document does not specify the API surface of the
  `ExecutionStrategy` trait.
- This document does not specify which strategies will exist in v21.
- This document does not commit to a timeline.

It establishes the architectural principle so that implementation
decisions in v21+ remain consistent with the design philosophy.
