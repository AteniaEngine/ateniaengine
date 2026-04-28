# ADR-003: Methodology Questioning Framework

## Status

Accepted

## Date

2026-04-28

## Context

Atenia Engine operates in an industry where most architectural decisions are inherited from established practice. The default posture for most ML runtime projects is to accept the dominant methodology and optimize within it. This ADR establishes an alternative posture for Atenia.

The insight motivating this ADR: the industry tends to respond to problems by adding resources rather than by questioning methodology.

Examples of this pattern:

- "Model does not fit in VRAM" → "Buy a larger GPU" rather than "Use available memory tiers more intelligently"
- "Inference is slow" → "Buy a faster GPU" rather than "Question whether GPU is the right baseline"
- "Larger models perform better" → "Train at greater scale" rather than "Question whether parameter count is the bottleneck"
- "Long context requires more compute" → "Quadratic attention is fundamental" rather than "Question whether attention is the right primitive"

Atenia Engine already implicitly questions methodology in one dimension (memory tiering via the M3-e reaction loop, addressing the first example above). This ADR formalizes the disposition to question methodology as a project principle, applicable to future decisions across all dimensions.

## Decision

Atenia Engine adopts methodology questioning as an explicit operating principle. When making architectural decisions, the project will:

1. Explicitly distinguish between conventions (implementational choices that can be revised) and fundamental properties (mathematical or physical truths that cannot).
2. For conventions, ask: "Is this the correct solution, or is it a legacy of constraints that no longer apply?" Resist the default of accepting industry practice without examination.
3. For fundamental properties, accept them as given. Do not spend project resources attempting to revise mathematics, thermodynamics, or other established truths.
4. When questioning a convention reveals a potentially better methodology, evaluate the cost of pursuing it. Some alternative methodologies are tractable within the project's scope; others require dedicated research efforts and should be captured as research questions, not project tasks.
5. Maintain technical discipline within the existing reality while exploring methodological alternatives. Atenia must continue to function as an engine that executes real models, even when its design embodies methodological choices that diverge from industry default.

## Rationale

This decision is motivated by several observations.

### Observation 1: Validation by adoption is not validation of correctness

Widespread adoption of a methodology is evidence that the methodology is workable, not that it is optimal. The history of science contains many examples of widely adopted methods being superseded by better methods that were dismissed during their periods of dominance. Geocentrism, phlogiston theory, and many other examples illustrate that "validated by many" does not imply "100% correct".

### Observation 2: The industry's default response pattern is suboptimal

The pattern "problem → more resources" works in environments where resources are cheap and abundant. In environments with constraints (consumer hardware, edge devices, environmental cost, accessibility budgets), "problem → better methodology" is more sustainable.

Atenia explicitly serves the constrained environment. This makes methodology questioning not just intellectually valuable but operationally necessary.

### Observation 3: Atenia's existing differentiation is methodology questioning

The M3-e reaction loop is, fundamentally, a refusal to accept the methodology "if it does not fit in VRAM, you cannot run it". This refusal is what makes Atenia distinct from other engines. The principle of methodology questioning, then, is not new to the project — it is the principle that already produced the project's most distinctive feature. Formalizing it ensures it applies consistently to future decisions.

### Observation 4: Distinguishing conventions from fundamentals is the discriminator

Without the convention/fundamental distinction, methodology questioning becomes either everything (paralyzing) or nothing (default acceptance). With the distinction, the project can focus its questioning energy where it has leverage: on implementational choices that are revisable, not on mathematical truths that are not.

## Consequences

### Positive

- The project's most distinctive features (already established in M3-e) are now connected to an explicit principle that can guide future work
- Future architectural decisions have a clear lens for evaluation: is this convention or fundamental? Is the convention worth questioning?
- The project's identity is articulated more clearly for both internal and external communication
- Researchers, contributors, and users encountering the project can understand what makes it distinct without having to infer from the architecture

### Negative

- Some decisions will take longer because they require explicit evaluation of "convention or fundamental?" before proceeding
- Resisting industry default may produce architectural choices that are unfamiliar to potential users or contributors, increasing the cognitive load of adopting Atenia
- The principle creates an obligation to apply it consistently. Failing to question methodology in some areas while questioning it in others produces inconsistent project identity

### Neutral

- This ADR does not commit to questioning any specific methodology. It commits to applying the questioning framework when appropriate, and to capturing the questioning explicitly when it occurs
- The project does not become "anti-industry" or contrarian by default. Industry methodologies are accepted when examination reveals they are correct, and questioned when examination reveals they are inherited rather than chosen

## Application Areas

The following are areas where Atenia already applies or could apply methodology questioning. These are not commitments to specific work; they are an inventory of where the framework operates.

### Currently applied

- **Memory tiering** (M3-e): rejecting "model must fit in VRAM" in favor of "model can use VRAM + RAM + disk intelligently"
- **Transparency** (project principle): rejecting "engine is a black box" in favor of "engine's decisions are auditable"
- **Numerical ground truth** (ADR-002): rejecting "PyTorch is ground truth" in favor of "mathematics is ground truth"
- **F32 over BF16 storage** (M4 decision): rejecting "smaller precision is always better for memory" in favor of "correctness validation precedes precision optimization"

### Potentially applicable in future scope

These are not commitments. They are examples of where the framework could be applied if the project chose to investigate.

- **Tokenizer separation**: questioning "tokenizer is a separate preprocessing stage" against research like ByT5 and Charformer that operate on bytes directly
- **Parameter scale**: questioning "more parameters = better capability" against evidence that smaller well-trained models (e.g., Phi-3) match larger ones on many benchmarks
- **GPU as inference baseline**: questioning "inference requires GPU" against modern CPU instruction sets (AVX-512, AMX, ARM SVE) that may be competitive for many workloads
- **Quantization for compression**: questioning "lower precision is purely a compression technique" against research on alternative number representations (posit, logarithmic)
- **Attention as primitive**: questioning "attention is the right computational primitive for sequence modeling" against state space models like Mamba

The framework is not committing the project to investigate any of these. It is committing to recognize them as questionable when relevant decisions arise.

### Out of scope

The framework is explicitly not a license to question mathematical or physical fundamentals. The following remain accepted as given:

- Floating-point arithmetic semantics (IEEE 754)
- Mathematical operations (addition, multiplication, etc.) and their properties
- Computability theory (what is computable in principle)
- Physical constraints (memory bandwidth, speed of light, etc.)

Questioning these is legitimate scientific activity but is not within Atenia's project scope.

## Alternatives Considered

### Alternative 1: Do not formalize the principle

Rejected. The principle is already implicit in the project's existing differentiation. Without explicit articulation, the principle risks being inconsistently applied as the project grows or as decisions accumulate over time.

### Alternative 2: Treat each convention questioning as a separate ADR rather than establishing a framework

Rejected. Individual ADRs (such as ADR-002 questioning the "PyTorch as ground truth" convention) capture specific decisions. A framework ADR captures the general operating principle that makes individual decisions consistent. Both have their place. This ADR is the framework; individual decisions remain captured in their own ADRs.

### Alternative 3: Adopt a more aggressive stance ("everything the industry does is questionable")

Rejected. Aggressive contrarianism produces noise and undermines the project's technical credibility. The framework explicitly distinguishes conventions (worth questioning) from fundamentals (not worth questioning), keeping the questioning focused on where it has leverage.

## Trigger to Revisit

This ADR is foundational and is not expected to be revised in the normal course of project development. Conditions that would trigger revision:

- The project's scope shifts substantially (e.g., from inference engine to training framework, or to a domain other than ML runtimes)
- A specific application of the framework consistently produces poor outcomes, suggesting the framework itself has a flaw
- Better articulations of the same principle become available (literature, similar projects' documentation), and adopting them improves clarity

## References

- [HANDOFF M4.5](../HANDOFF_APX_V20_M4.5.md) — Atenia's currently implemented methodology questioning (M3-e reaction loop)
- [ADR-001](./ADR-001-numerical-health-monitor-deferred.md) — applies the framework's "do not commit without data" implicit principle to a specific decision
- [ADR-002](./ADR-002-mathematical-ground-truth-validation.md) — applies the framework explicitly to question "PyTorch as ground truth"
- [ROADMAP](../../ROADMAP.md) — milestones reflect the project's scope under this framework
