# ADR-004: F64 Reference as Default Numerical Validation Methodology

## Status

Accepted

## Date

2026-04-28

## Context

ADR-002 (Mathematical Ground-Truth Validation Strategy, accepted 2026-04-26) proposed a three-level validation approach: NumPy F64 reference, cross-framework F32 comparison, and mpmath arbitrary precision. The ADR identified three possible interpretations of M4.5 numerical drift (Cases A, B, C) and committed to disambiguating them with empirical data when M4.6 generated additional datapoints.

During Phase A.1 of M4.6 (tied word embeddings support), the SmolLM2 1.7B model was validated end-to-end. The initial validation against PyTorch BF16 reference (M4.5-d.1 methodology) showed catastrophic drift concentrated in position 0:

- Max abs diff (Atenia vs PyTorch BF16): 14.0125
- Position 0 alone: 62.41% of logits drifted by more than 1.0
- TinyLlama (M4.5 baseline) showed only 0.73 max abs diff

This triggered a sequence of investigations to discriminate between bug hypotheses (H1–H5). All hypotheses were ruled out empirically: tied path (H1), rope_theta high (H2), vocab × tied (H3), SmolLM2-specific quirk in transformers code (H4), and softmax with -inf adversarial (H5).

Investigation F applied ADR-002's F64 reference methodology to the case directly. The triple comparison (Atenia F32 vs PyTorch BF16 vs PyTorch F64 truth) produced:

- Atenia F32 max drift vs F64: **0.001446**
- PyTorch BF16 max drift vs F64: **14.013928**
- Atenia is approximately **9692× more precise** than PyTorch BF16 on this model
- Argmax matches in all 4 positions Atenia-vs-F64
- Per-position drift Atenia-vs-F64 uniform (0.001 to 0.0001)

This is Case A of ADR-002 confirmed: Atenia F32 is mathematically correct; PyTorch BF16 is the source of observed drift. The "pos 0 catastrophic drift" entirely reflects BF16 accumulation patterns in 24-layer transformers, not Atenia's implementation.

## Decision

Establish F64 reference as the **primary validation methodology** for Atenia's numerical correctness from M4.6 onward.

Specifically:

1. Per-model numerical validation tests (M4.x-d.1 pattern) compare Atenia outputs against F64 reference (PyTorch loaded with `model.double()` or equivalent), not against PyTorch BF16 default.

2. PyTorch BF16 comparison is retained as **secondary informative metric**, useful for tracking how industry-standard BF16 inference drifts from mathematical truth on each model. This metric does not gate test pass/fail.

3. Tolerance threshold for Atenia-vs-F64 comparison is set at `max_abs_diff < 0.5`. Atenia empirically achieves ~0.001 on SmolLM2, providing ~300× safety margin against threshold while still detecting real numerical regressions.

4. New models added to Atenia (Llama 3.2, Qwen 2.5, future variants) follow the same pattern: F64 reference fixture is generated alongside BF16 reference, F64 is the assertion target, BF16 stats are reported informatively.

5. The TinyLlama M4.5-d.1 baseline (currently asserted against PyTorch BF16 with `max_abs_diff < 5.0` tolerance) is retained as-is for historical continuity. A new F64-based validation can be added in M4.6.1 retroactively without disturbing the existing M4.5 commit.

## Rationale

Empirical data from SmolLM2 demonstrates that the assumption "PyTorch BF16 is approximately ground truth" breaks for some models with deep transformer stacks. The TinyLlama case (where Atenia and PyTorch BF16 both happened to be similarly close to truth) was not generalizable. Without F64 reference, the SmolLM2 investigation would have had two possible conclusions:

- Lower threshold to accept observed drift (effectively hiding the question)
- Conclude there is a bug somewhere and continue investigating indefinitely without a way to falsify

Both outcomes are unacceptable. F64 reference resolves the question definitively: Atenia is mathematically correct within F32 precision; the observed BF16 drift is the industry's precision trade-off, not Atenia's bug.

This decision aligns with:

- ADR-002: validating against mathematical truth, not reference implementations
- ADR-003: questioning inherited methodologies (the convention of "PyTorch as ground truth" is now empirically demonstrated as misleading for some models)
- Project principle "stability before performance": F32 throughout validates as a correctness choice, not just a memory cost

## Consequences

### Positive

- Numerical validation in M4.6 onward has a methodologically sound reference (F64) instead of an empirically-shown-flawed one (PyTorch BF16)
- Atenia's correctness claims are now backed by direct comparison to mathematical truth, not by agreement with a less-precise reference implementation
- Future cases of "Atenia and PyTorch disagree" can be disambiguated quickly: regenerate F64 fixture, see which is closer to truth
- The methodology generalizes to any model addable to Atenia (PyTorch supports `model.double()` universally)

### Negative

- Generating F64 fixtures requires PyTorch to load model in F64, which roughly doubles RAM requirements during fixture generation. For 1–2 B models on 16 GB systems this is feasible (verified empirically with SmolLM2). For larger models (~7 B+), F64 fixture generation may require dedicated hardware or alternative strategies (layer-by-layer F64 conversion, F32 PyTorch reference as fallback).
- F64 fixture files are larger than BF16 (8 bytes vs 2 bytes per logit). For SmolLM2 with vocab 49 152 × 4 positions, this is 3.9 MB. Acceptable for current models; may need reconsideration for very large vocabularies or longer sequences.

### Neutral

- The BF16 reference fixtures are retained, not deleted. They serve as informative comparisons and historical continuity.
- M4.5-d.1 TinyLlama baseline is not retroactively changed. The M4.5 commit (`40fc4fa`) reflects the methodology current at that time. ADR-002 and ADR-004 explain the methodological evolution.

## Alternatives Considered

### Alternative 1: Lower the BF16 threshold to accept observed drift

Rejected. The drift observed in SmolLM2 (max 14.0) was an order of magnitude larger than TinyLlama (0.73). Lowering threshold to 20.0 or similar would hide the question rather than answer it. ADR-001 explicitly rejected "ignore the data" as a methodology.

### Alternative 2: Update ADR-002 retroactively rather than create ADR-004

Rejected. ADR-002 is well-formed as a strategy proposal under hypothesis. ADR-004 captures the operational decision that follows from empirical confirmation. Keeping ADR-002 as-is preserves the historical reasoning chain. ADR-004 references and builds on it.

### Alternative 3: Defer the methodology change until M4.6.1

Rejected. The SmolLM2 case is the first practical test of the new methodology, and waiting until M4.6.1 to formalize would mean Qwen 2.5 and Llama 3.2 (the next models) are validated under the inferior methodology. The cost of generating F64 fixtures alongside BF16 fixtures is incremental; the benefit of methodologically sound validation is significant.

## Trigger to Revisit

This decision should be revisited if:

- F64 fixture generation becomes intractable for some model scale or class. In that case, fallback methodologies (cross-framework F32 comparison per ADR-002 Level 2, or layer-isolated mpmath validation per Level 3) would need formalization.
- A model is encountered where Atenia F32 drift vs F64 exceeds 0.5 (the current threshold). This would indicate a real numerical issue requiring investigation, not a methodology problem.
- Industry methodology shifts in a way that makes BF16 reference meaningful again (e.g., if BF16 numerical stability research produces stable formulations). Currently this seems unlikely but should not be foreclosed.

## Application Note: Investigation Sequence

This ADR was preceded by a sequence of empirical investigations that ruled out alternative hypotheses before applying ADR-002's methodology. The investigation sequence is documented in commit history and is preserved as part of the project's epistemological record:

- Investigation A (stats granulares per position): identified drift concentrated in position 0
- Investigation C (rope_theta override): ruled out RoPE base frequency
- Investigation D-fast (TinyLlama tied forced): ruled out path A.1 as systemic source
- Softmax adversarial test: ruled out softmax numerical instability
- Lectura completa `modeling_llama.py`: ruled out SmolLM2-specific quirks
- Investigation F (F64 reference): confirmed Case A of ADR-002

The investigation sequence is preserved in test files (untracked or committed as appropriate) and documents the methodological discipline applied.

## References

- [ADR-001](./ADR-001-numerical-health-monitor-deferred.md) — applies "no decisions without data" principle
- [ADR-002](./ADR-002-mathematical-ground-truth-validation.md) — proposed three-level validation; this ADR confirms Case A and operationalizes the methodology
- [ADR-003](./ADR-003-methodology-questioning-framework.md) — questioning "PyTorch as ground truth" was an explicit application of this framework
- [HANDOFF M4.5](../HANDOFF_APX_V20_M4.5.md) — initial drift observation that motivated investigation
- [ROADMAP](../../ROADMAP.md) — M4.6 milestone where this methodology is first applied
- [ADR-007 — MoE certification by decomposition](./ADR-007-moe-certification-ladder.md) — extends this contract to Mixture-of-Experts models, where a single global F64 forward is infeasible (RAM) and incomplete (sparse routing); **reuses this ADR's `max_abs_diff < 0.5` + argmax bar verbatim** and adds the L0–L4 ladder rather than changing any threshold here. (This is the "F64 fixture generation becomes intractable for some model scale" trigger above, resolved for the MoE class.)
