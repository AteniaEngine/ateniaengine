# ADR-002: Mathematical Ground-Truth Validation Strategy

## Status

Accepted (implementation in M4.6.1)

## Date

2026-04-26

## Context

M4.5 closed with empirical numerical validation comparing Atenia's TinyLlama logits against a PyTorch reference. The observed drift (max abs 0.732, mean abs 0.060) was characterized as "within F32-vs-BF16 precision tolerance over 22 transformer blocks".

This characterization implicitly treats PyTorch as the ground truth against which Atenia's correctness is measured. A more rigorous framing recognizes that:

- Mathematics is an exact science. For given inputs, there is a single mathematically correct result.
- Computers compute approximations. Every framework (PyTorch, TensorFlow, JAX, Atenia, llama.cpp) introduces approximation error through floating-point arithmetic.
- "Atenia drifts 0.06 vs PyTorch" answers a question, but not the most important one. The more important question is: how far is each framework from mathematical truth?

Three possible interpretations of the M4.5 drift exist, and we cannot distinguish between them without additional measurement:

- **Case A**: Atenia (F32) is closer to mathematical truth than PyTorch (BF16). The observed drift is dominated by PyTorch's precision tradeoff. Atenia is mathematically correct.
- **Case B**: Atenia has a numerical bug that drifts away from truth. The fact that it differs from PyTorch is a symptom of the bug, not just precision difference.
- **Case C**: Both Atenia and PyTorch have correlated errors (improbable but possible). Both drift from truth in similar ways.

Without measuring against ground truth, we cannot distinguish these cases. This is significant because:

- Case A is acceptable; the project's stated principle of "stability before performance" is satisfied.
- Case B requires correction.
- Case C requires deeper investigation.

## Decision

Establish mathematical ground-truth validation as part of Atenia's empirical claims. Implement this in milestone M4.6.1, after M4.6 closes (Llama-family expansion to Qwen 2.5, Llama 3.2, SmolLM2).

The validation strategy combines three reference levels of increasing rigor:

### Level 1 — F64 reference (NumPy)

For each validated model, run forward pass in NumPy F64 with identical weights and inputs. F64 has 15–17 decimal digits of precision vs F32's 7, providing a "practical truth" against which both Atenia (F32) and PyTorch (BF16/F32) can be compared.

Cost: minutes per forward (NumPy F64 is fast). Suitable for full-model validation across all 4 models in M4.6.

### Level 2 — Cross-framework F32 comparison

Run identical forward in PyTorch with `model.float()` (forcing F32 instead of BF16). Compare:

- Atenia F32 vs PyTorch F32 (should be near-bit-exact if both are mathematically correct)
- Both vs NumPy F64 (truth reference)

If Atenia F32 ≈ PyTorch F32 ≈ NumPy F64, all three are mathematically correct. Differences indicate which is drifting from truth.

Cost: low (one extra PyTorch run per model). Suitable for inclusion in M4.6.1.

### Level 3 — Arbitrary precision reference (mpmath)

For isolated components (a single RMSNorm + Linear + RoPE block), compute the result with mpmath using ~100 digits of precision. This is "essentially mathematical truth" for any practical purpose.

Compute Atenia F32, PyTorch F32, NumPy F64, mpmath, and report errors against mpmath. Allows quantifying the error contribution of each precision level.

Cost: high per operation (mpmath is slow), low frequency (isolated components, not full models). Suitable for opt-in diagnostic, not routine validation.

## Rationale

This decision is motivated by epistemic rigor. The current M4.5-d.1 test answers "does Atenia agree with PyTorch?" The proposed validation answers "is Atenia mathematically correct?"

These are different questions. Agreement with PyTorch is sufficient for many practical purposes, but does not validate the fundamental claim implicit in the project: that Atenia's mathematical computations are correct.

Specifically:

1. **The current claim "drift within F32-vs-BF16 tolerance" is inferential, not measured.** We assume the drift is due to precision difference because the magnitude is consistent with that hypothesis. We have not eliminated alternative hypotheses (bug, correlated error).

2. **F64 reference eliminates the BF16 comparison artifact.** Comparing both Atenia F32 and PyTorch F32 against NumPy F64 removes the "PyTorch's BF16 is less precise" confound. Any remaining drift is precision-only or bug.

3. **The cost is bounded.** Level 1 (NumPy F64) is the bulk of the validation work and is computationally cheap. Level 2 is trivial (one extra PyTorch invocation). Level 3 is opt-in and surgical.

4. **It refines the empirical claim.** Instead of "drift within F32-vs-BF16 tolerance" (a claim that requires explanation), the validation produces "Atenia F32 within X ULPs of NumPy F64 ground truth across 4 models, consistent with expected F32 accumulation error" — a measurable, falsifiable claim.

## Consequences

### Positive

- Empirical claims about Atenia's correctness become measurable and verifiable rather than inferential
- Distinguishing precision drift from bugs becomes a routine part of validation
- The project gains a stronger foundation for claims of "mathematical correctness" in its public communication
- Future numerical issues can be diagnosed against established baselines

### Negative

- M4.6.1 is added as a sub-milestone (~1–2 days estimated)
- The validation pipeline grows to require NumPy F64 forward computation, which must be implemented for each model
- Level 3 (mpmath) requires careful selection of which components to validate; not exhaustive

### Neutral

- M4.5-d.1 (PyTorch BF16 comparison) is preserved as-is. The new validation is additive, not a replacement.
- The framework defined here can be reused for any future model added to Atenia, not just M4.6 models.

## Alternatives Considered

### Alternative 1: Continue with PyTorch-only validation

Rejected. Implicitly treats PyTorch as ground truth, which is a weaker claim than "mathematically correct". Insufficient to distinguish Cases A, B, C above.

### Alternative 2: Implement mpmath validation for full models

Rejected. mpmath at 100-digit precision is too slow for full forward passes (estimated days per forward for 22-layer models). Confined to Level 3 isolated-component validation where it is tractable.

### Alternative 3: Defer the entire question to a hypothetical future milestone

Rejected. The framing ambiguity (Cases A/B/C indistinguishable) is present today and grows as more models are validated. Establishing ground-truth methodology in M4.6.1, immediately after M4.6 generates the data, is the natural point.

## Trigger to Revisit

This decision may need revisiting if:

- M4.6.1 reveals that Atenia is materially distant from F64 truth (suggesting Case B, a bug). The decision then becomes about how to fix the identified bug, not the validation methodology itself.
- Computational cost of the F64 reference becomes prohibitive for some model class (unlikely but possible for very large models). The methodology may need to be selective.
- A more rigorous mathematical framework becomes available (e.g., interval arithmetic, formal verification of numerical kernels). Such a framework could supersede the current three-level approach.

## References

- [HANDOFF M4.5](../HANDOFF_APX_V20_M4.5.md) — empirical drift results that motivated this discussion
- [ADR-001](./ADR-001-numerical-health-monitor-deferred.md) — related decision on runtime self-monitoring; ADR-002 informs the "what to monitor against" question that ADR-001 deferred
- [ROADMAP](../../ROADMAP.md) — M4.6.1 sub-milestone where this is implemented
