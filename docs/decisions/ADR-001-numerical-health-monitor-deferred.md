# ADR-001: Numerical Health Monitor — Deferred Pending Empirical Data

## Status

Deferred

## Date

2026-04-26

## Context

M4.5 closed with empirical numerical validation of TinyLlama 1.1B against a PyTorch reference. The observed drift was within expected F32-vs-BF16 precision bounds:

- Max absolute difference: 0.732
- Mean absolute difference: 0.060
- Values with diff > 1.0: 0 (0.00%)
- Argmax disagreement at top-1: PyTorch and Atenia each picked the other's #2 candidate, with logit gap ≈ 0.03 between top-1 and top-2 in both engines (near-tie behavior)

Following this, the question was raised whether Atenia should include a runtime "Numerical Health Monitor" — a module that continuously verifies the engine's mathematics during execution and detects patterns of drift that might indicate either bugs or accumulated precision degradation.

The proposed module would include components such as:

- Continuous sanity checks (finite/range validation on intermediate outputs)
- Periodic self-test (mini-model with pre-computed expected values)
- Statistical drift detection (sliding window statistics on output distributions)
- On-demand deep validation (command-level utility comparing against reference)

This direction aligns with project principles: "stability before performance", "observable from the outside", "no semantic drift".

## Decision

Defer implementation of a Numerical Health Monitor module pending empirical drift data from upcoming milestones M4.6 (Llama-family expansion: Qwen 2.5, Llama 3.2, SmolLM2) and M4.7 (beyond-VRAM execution with reaction loop active).

## Rationale

The decision applies the same principle previously adopted for performance optimization: do not commit to corrective architecture without data demonstrating the problem is real and material.

Specifically:

1. **Single datapoint is insufficient.** TinyLlama alone does not tell us whether the observed drift is structural (model-class independent) or specific to TinyLlama's architecture / training. M4.6 will produce 3 additional datapoints (Qwen, Llama 3.2, SmolLM2) as a natural byproduct, with no extra work.

2. **M4.7 stress-tests the most likely drift source.** The reaction loop (M3-e) moves data between VRAM, RAM, and disk during execution. If accumulated numerical drift is going to manifest anywhere, it is most likely there. M4.6 cannot exercise this; M4.7 will.

3. **Designing without empirical data risks designing wrong.** Any monitoring architecture committed to today would be based on a single datapoint (TinyLlama). The right architecture for the actual drift profile observed across multiple models may be substantially different.

4. **The natural plan generates the data needed for the decision.** M4.6 includes per-model PyTorch comparison tests (M4.5-d.1 pattern). M4.7 will include its own numerical validation given the reaction loop's potential to introduce error. No additional work is needed to obtain the data.

## Consequences

### Positive

- M4.6 retains its current scope and timeline (~7 days, Plan B confirmed: Phase A + Qwen + Llama 3.2 + SmolLM2)
- M4.7 retains its scope as beyond-VRAM execution validation
- The eventual Numerical Health Monitor (if warranted) will be designed against real drift profiles from multiple models, not speculation
- Avoids premature commitment to an architecture that may not match the actual problem

### Negative

- Atenia continues without explicit runtime self-monitoring of numerical health for the duration of M4.6 + M4.7
- If a numerically severe issue exists between TinyLlama and other models, it will be caught at the test level rather than at runtime
- Some users may interpret the absence of a Numerical Health Monitor as a gap; this is mitigated by the explicit per-model PyTorch validation tests already planned

### Neutral

- The captured ideas (sanity checks, periodic self-test, drift detection, on-demand validation) are preserved in this ADR for future implementation if the decision is reversed

## Trigger to Revisit

This decision should be revisited once aggregate drift data is available from M4.6 and M4.7. The trigger conditions are:

- **No revision needed:** All 4 models (TinyLlama + Qwen + Llama 3.2 + SmolLM2) show drift in consistent ranges (mean abs diff ~0.05–0.15, max abs diff < 2.0). M4.7 reaction loop does not inflate drift significantly.

- **Targeted investigation:** One model shows dramatically higher drift than the others (e.g., 5× or more). Investigate root cause in that model specifically; revisit ADR if the root cause suggests systemic issue.

- **Numerical Health Monitor warranted:** Drift levels observed across models materially affect production usage (e.g., reproducibility complaints, benchmark divergence in user-facing contexts) OR M4.7 reaction loop introduces significant additional drift that suggests need for runtime monitoring.

## Alternatives Considered

### Alternative 1: Implement Numerical Health Monitor immediately

Rejected. The project would commit architecture against speculation rather than evidence. The same module designed against 4 models' real drift profiles is likely to differ from one designed against 1 model.

### Alternative 2: Implement only continuous sanity checks (Pattern A) now

Rejected. While Pattern A (finite/range checks) is the cheapest component and clearly useful, isolating it from the rest of the monitor design splits the architecture across milestones. Better to design the full monitor coherently when data justifies it.

### Alternative 3: No documentation, address only if reported

Rejected. The captured ideas (Pattern A through D) represent substantive design work that would be lost without documentation. This ADR preserves the design space for future implementation.

## Captured Ideas (for future implementation if triggered)

The following components were considered as part of a Numerical Health Monitor design. They are preserved here so they need not be re-derived.

### Component 1: Continuous Sanity Checks (Pattern A)

After each forward pass (or a sampled subset of passes), validate:

- All outputs finite (no NaN, no Inf)
- Magnitudes within configurable range
- Variance non-degenerate (not collapsed to ~0)

Cost: ~1% runtime overhead. Detects: catastrophic corruption, not subtle drift.

### Component 2: Periodic Self-Test (Pattern D)

At configurable intervals (e.g., every N forwards), execute:

- Synthetic deterministic input through a known mini-graph
- Compare output against pre-computed expected values
- Report exact drift, alert if exceeds threshold

Cost: low (small graph, runs infrequently). Detects: gradual engine degradation.

### Component 3: Statistical Drift Detection (Pattern C)

Maintain sliding-window statistics (max, mean, std) on outputs per NodeType. If distribution shifts significantly between runs, alert.

Cost: trivial. Detects: production-time changes indicating degradation.

### Component 4: On-Demand Deep Validation

Explicit command (e.g., `atenia validate-numerics`):

- Executes reference model (TinyLlama or other)
- Compares against PyTorch reference logits embedded in repo
- Reports complete drift profile
- Suitable for CI integration

Cost: high when invoked, invoked only on demand. Detects: comprehensive validation snapshot.

## References

- [HANDOFF M4.5](../HANDOFF_APX_V20_M4.5.md) — Empirical drift results that motivated this discussion
- [ROADMAP](../../ROADMAP.md) — M4.6 and M4.7 milestones that will produce the data triggering revisit
- Related principle in [ROADMAP](../../ROADMAP.md): "Observable and reproducible — Every behavior claimed by the engine must be verifiable through executable tests"
