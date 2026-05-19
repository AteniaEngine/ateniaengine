# ADR-006 — Adapter Toolkit: a declarative FamilyTensorSpec for the model-loading boundary

## Status

Accepted (AT-0). Implementation phased; this ADR governs the boundary, not a
shipped feature. AT-2 and AT-1 (below) are gated on this decision and on the
per-phase audit -> approve -> commit contract. AT-0/AT-2/AT-1 landed;
GAP-1/N1/N2/C1/C2/T1 closed (Gemma 2 GGUF correctness, post-AT-1 — see the
closure addendum at the end of this ADR). **AT-3 landed; Adapter Toolkit v1
closed; AT-4 remains explicitly deferred — see the v1 closure section at the
end of this ADR.**

## Context

Phases 11-16 progressively pulled family-specific *logic* out of the execution
core into the `src/model_adapters/` layer behind seven traits (`ModelAdapter`,
`HfWeightMapper`, `GgufWeightMapper`, `GgufNameMapper`, `StoreBackedGraphBuilder`,
`ResidencyHints`, `ConfigPolicy`). The *config* boundary closed in Phases 13-15;
the GGUF->HF weight-name boundary closed in Phase 16. Adapter *selection*
(`resolve_adapter`) and the trait surface are clean and well tested.

The Phi-3.5-mini-instruct Q4_K_M GGUF bring-up (commits `a712f28`, `e67f627`,
`b423f56`, `345482d`) exposed that, although the trait surface is clean, the
*data those traits dispatch over* is still imperative, duplicated, and
order-sensitive, scattered across at least four files outside the adapter
module:

- **GGUF->HF name mapping** — hand-written `match` arms in
  `v17/loader/gguf_to_hf_naming.rs` (`gguf_to_hf_name_common` +
  `phi3_gguf_extra` + `gemma2_gguf_extra`). The adapter *composes* them with
  `.or_else()`, and the composition order is semantically load-bearing: the
  Phi-3 fused `ffn_up -> gate_up_proj` bug (`345482d` / issue #5a) was an
  inverted composition order where the common table shadowed the family
  override.
- **Per-name load transforms** — three independent `if name.contains(...)`
  ladders: `weight_loading.rs::compute_transforms_for_name` (Llama/HF),
  `phi3.rs::phi3_transforms_for_name`, and
  `gguf_weight_loading.rs::gemma2_gguf_transforms_for_name`. The HF and GGUF
  transform tables carry the same information twice and silently diverged for
  Phi-3 (`b423f56` / issue #4: embed transposed when it must not be, Linear
  weights skipped when they must be transposed), surfacing only as a runtime
  shape mismatch.
- **Non-weight (config-input) tensors** — a hardcoded two-name `matches!` in
  `is_gguf_non_weight_tensor`, plus the GGUF completeness gate duplicated at
  two sites in `pipeline.rs` with a literal
  `!s.contains("rope_freqs") && !is_gguf_non_weight_tensor(s)`.
- **GGUF dtype coverage** — `gguf_decode.rs::decode_tensor` supports
  F32/F16/Q8_0/Q4_K/Q5_K/Q6_K; every other `GgufTensorType`
  (Q2_K/Q3_K/Q4_0/Q4_1/Q5_0/Q5_1/Q8_1/Q8_K/BF16/IQ*) fails as
  `UnsupportedDType` at decode time, not declared up front. The Phi-3.5 Q5_K
  gap (`e67f627` / issue #3) was discovered by a runtime failure, not by a
  coverage check.

The common thread: the trait *interface* is closed, but the family-specific
*data* it routes is imperative and unverified, so a new family is brought up by
copy-pasting ladders and discovering mistakes at runtime.

## Decision

Introduce an **Adapter Toolkit**: a declarative description of the
loading/mapping data each family needs, plus a conformance harness that
verifies it, plus a coverage validator. Specifically:

1. **`FamilyTensorSpec` is Rust data, not YAML/JSON.** A per-family
   struct/const that unifies, in one place: GGUF->HF name-mapping rules,
   per-name transform rules, the non-weight (config-input) tensor set, and
   fused-tensor descriptors. The free functions become lookups over the spec.
2. **The toolkit covers the loading/mapping layer only.** Graph builders
   (`build_llama`/`build_phi3`/`build_gemma2`), numeric policy, CUDA/CPU
   kernels, tiering/placement, and CI are explicitly out of scope. Those are
   genuinely different programs, not tables, and changing them is not the
   problem this ADR addresses.
3. **The refactor is behaviour-preserving.** AT-1 must not change any loaded
   weight, any transform order, any numeric output, or any control flow on the
   success path. The oracle is the existing test suite (382/382) plus numcert,
   plus the AT-2 conformance harness written *before* the refactor.
4. **Phase order is AT-0 -> AT-2 -> AT-1.** Docs (this ADR) first; then the
   conformance harness as an executable freeze of *current* behaviour
   (test-only, zero risk); then the declarative refactor, proven by the
   harness. Each phase is a separate audit -> approve -> commit unit.
5. **The adapter retains an imperative escape hatch.** If a future family
   needs mapping logic the spec cannot express, the adapter may still override
   imperatively. The spec models exactly today's three layouts
   (Llama-like / Phi-3 fused / Gemma 2 dual-norm); no speculative generality.

## What becomes declarative

Expressed as data in `FamilyTensorSpec`, looked up instead of branched:

- GGUF->HF tensor-name rules (top-level + per-block suffixes, including
  family fused/extra overrides) with the family-override-wins ordering encoded
  in the data, not in `.or_else()` call order.
- Per-name load-transform rules per tensor class (embed / norm / the Linear
  classes / fused QKV / fused gate_up / KV-tiled), with a single source of
  truth shared by the HF and GGUF paths (post-`.rev()` GGUF orientation equals
  HF safetensors orientation, the invariant `b423f56` established).
- The non-weight (config-input) tensor set, consumed by one completeness
  helper instead of two duplicated `pipeline.rs` gates.
- The declared GGUF dtype + tensor-class requirements of each adapter, so a
  coverage validator can flag a gap at registry/load time instead of at
  decode time.

## What stays imperative

- Graph construction (`build_*`): dual-norm, logit softcap, fused-tensor
  split, the LongRope `panic!` guard. These are model topology, not mapping
  tables.
- `ConfigPolicy` semantics (rope-scaling parsing, validation, defaults). Phases
  12-15 already closed this boundary; it is not re-opened here.
- Adapter selection (`resolve_adapter`) and the trait surface. Closed and
  tested; unchanged.
- The GGUF k-quant decoders themselves. The toolkit *declares* required
  coverage; it does not generate decoders.

## Why Rust data, not YAML/JSON (now)

- At five families the spec is small and entirely internal. YAML/JSON adds a
  parser, a schema, runtime parse-failure modes, and a serialization
  compatibility surface for zero functional gain.
- Rust data is type-checked by the compiler and refactored with the code; a
  malformed spec is a build error, not a runtime surprise — the opposite of
  the failure mode this ADR is closing.
- An external template format is an SDK surface. `model_adapters/mod.rs`
  states the layer is "deliberately not a public SDK yet"; committing to a
  serialized contract now would prematurely freeze it. Deferred to AT-4, only
  if a real new family or a public SDK requires it.

## Non-goal (explicit)

**The Adapter Toolkit does not provide automatic or "magic" support for new
models.** A new family still requires: a graph builder, a numeric validation
(F64 strict per ADR-004 or the functional GGUF schema), and explicit
review. The toolkit reduces boilerplate, removes the duplication between the
HF and GGUF mapping paths, and converts a class of runtime failures
(composition-order bugs, HF/GGUF divergence, undeclared dtype gaps) into
compile-time or test-time failures. It does not infer, auto-detect, or
auto-wire a model the engine has not been taught. Any external claim must
state this boundary.

## Phase order, risks, and mitigations

| Phase | Scope | Type | Risk | Oracle |
|-------|-------|------|------|--------|
| AT-0  | This ADR | docs-only | none | review |
| AT-2  | Conformance harness over the adapter registry | test-only | none | current suite 382/382 |
| AT-1  | `FamilyTensorSpec` + free fns reexpressed as lookups | refactor | high (numeric/loading contract) | AT-2 + suite + numcert |

Risks and mitigations:

- **Silent numeric corruption (high).** The transform/name tables *are* the
  loading contract; reordering a transform list changes loaded weights with no
  error. Mitigation: AT-2 is written and committed *before* AT-1 and freezes
  current behaviour as an executable oracle; AT-1 is behaviour-preserving and
  re-runs AT-2 + the full suite + numcert; no new family is introduced in the
  same phase as the refactor; transform output is diffed per tensor name.
- **Over-abstraction.** A spec too generic to express the next family is a
  leakier abstraction than the ladder it replaced. Mitigation: model exactly
  the three current layouts; keep the imperative escape hatch (Decision 5).
- **Scope creep into graph builders.** `build_phi3`/`build_gemma2` are
  genuinely different graphs. Mitigation: graph construction is explicitly out
  of scope (Decision 2); the toolkit touches mapping/loading only.
- **False expectation of auto-support.** Stated and bounded by the Non-goal
  section; AT-2 also asserts that an undeclared dtype/class is a *declared*
  failure, not a silent one.

## Phi-3.5 GGUF lessons that motivated this ADR

Each of the five layered Phi-3.5 GGUF failures maps to a specific leak this
toolkit closes:

- **LongRope from a tensor, not metadata** (`a712f28`) — config inputs can
  arrive as GGUF tensors; the non-weight set was a hardcoded global
  `matches!`. -> declarative non-weight set + one completeness helper.
- **Q5_K unsupported** (`e67f627`) — a dtype gap discovered by runtime
  `UnsupportedDType`. -> adapters declare required dtype coverage; the
  validator flags the gap early.
- **HF vs GGUF transform divergence** (`b423f56`) — embed/Linear inverted
  because the two tables drifted. -> single declarative transform source
  shared by both paths (the post-`.rev()` orientation invariant).
- **Fused gate_up name-map shadowed** (`345482d` / #5a) — `.or_else()`
  composition order let the common table shadow the family override.
  -> family-override-wins encoded in the data, asserted by AT-2.
- **Completeness gate duplicated** (`345482d` / #5b) — the same skip predicate
  hand-copied at two `pipeline.rs` sites. -> one helper driven by the spec.

## Trigger to revisit

- A real new model family lands -> the spec's expressiveness is validated
  against it; if it needs the imperative escape hatch, that is recorded and the
  spec is *not* speculatively generalized.
- The adapter layer is promoted to a public SDK -> revisit the Rust-data vs
  serialized-template decision; AT-4 (external template format / scaffolding
  generator) is reconsidered with the SDK requirements in hand.
- A new GGUF dtype is implemented in `decode_tensor` -> the coverage
  declaration is the single place adapters opt into it.

## Consequences

- The trait surface is unchanged; adapter behaviour is unchanged after AT-1
  (behaviour-preserving by contract, proven by AT-2 + numcert).
- New-family bring-up shifts from copy-pasting ladders to filling one
  type-checked spec, with a conformance harness that fails loudly on
  composition-order, HF/GGUF-divergence, and undeclared-coverage mistakes.
- The HF and GGUF mapping paths stop being two hand-synchronized copies.
- The toolkit is internal Rust data; no serialized contract is frozen and no
  automatic model support is promised. AT-2 and AT-1 do not begin without the
  per-phase audit and explicit approval.

## Update — Gemma 2 GGUF correctness (post-AT-1)

The ADR-006 "Trigger to revisit" (a real model exercising the spec) fired:
bringing up Gemma 2 GGUF (bartowski/gemma-2-2b-it Q4_K_M) validated the
FamilyTensorSpec against a non-Llama-layout family and surfaced six
pre-existing, never-validated defects in the Gemma 2 GGUF path, fixed in
isolated regression-tested layers:

- GAP-N1 (`907f1ca`): the real llama.cpp tensor `post_ffw_norm.weight` was
  unmapped (the name-extra table held guessed names that never matched a real
  checkpoint).
- GAP-C1/C2 (`455983c`): the GGUF config read the wrong metadata keys —
  head_dim from `attention.head_size` (never present) instead of
  `attention.key_length`; softcaps from `attention.logit_softcap` instead of
  `attn_logit_softcapping`. Behaviour-preserving for every prior GGUF
  (key_length == hidden/heads there, so the `!= inferred` filter still
  yielded None).
- GAP-N2 (`a33d44f`): Gemma 2's 4-norm layout maps `ffn_norm` to
  `pre_feedforward_layernorm`; the common Llama-layout table mapped it to
  `post_attention_layernorm`. Resolved by composing the Gemma 2
  GgufNameMapper extra-first (mirrors Phi3Adapter; the same composition-order
  class as Phi-3 #5a).
- GAP-T1 (`36bbfe0`, root cause): llama.cpp pre-folds `1 + gamma` into the
  Gemma 2 GGUF norm weights (measured exactly +1.0 element-wise vs the HF
  safetensors of the same model, across every norm class); re-applying the
  HF `+1` double-folded every RMSNorm to `(2 + gamma)`. Fixed with a
  GGUF-specific transform table (the HF table minus the norm `+1` fold).

The buggy verbatim `GEMMA2_GGUF_GAP1` table and the GAP-1-only `TileKvDim1`
recipe were removed; the spec field `gguf_gap1_transforms` was renamed
`gguf_transforms` (`Some` = a family GGUF table distinct from HF; `None` =
Llama / Phi-3). The imperative escape hatch (Decision 5) was **not** needed —
the spec expressed the corrected table. No `RopeUnpermute` is required
(Phi-3 bracket); Gemma 2 GGUF now generates text identical to the HF
reference. The AT-2 conformance snapshot and the AT-1a golden that pinned the
buggy table were updated (intentional, checkpoint-validated, documented in
the commit), not adapted to hide a regression. TinyLlama / Phi-3.5 GGUF
regressions are unaffected.

## v1 closure

AT-3 landed; the Adapter Toolkit v1 is closed. The full ADR-006 microplan
is delivered:

- AT-0 (`20bc651`) — this ADR.
- AT-2 (`d9634a6`) — adapter conformance harness, the executable freeze
  oracle for AT-1.
- AT-1a/b/c (`536a506`, `914ae2a`, `60311ea`) — `FamilyTensorSpec` data
  plus golden A/B oracle, then the GGUF->HF name maps and the load
  transforms rewired onto the spec, behaviour-preserving.
- AT-3a (`8a18cbf`) — the two hand-copied GGUF load-completeness gates in
  `pipeline.rs` collapsed into a single `is_unexpected_gguf_skip` helper
  (the duplication this ADR explicitly named).
- AT-3b (`30314b1`) — `FamilyTensorSpec::required_gguf_dtypes` plus the
  `every_adapter_required_dtypes_are_decodable` conformance test. A future
  adapter that declares an undecodable dtype (e.g. Q2_K, BF16) fails at
  test time, not at runtime as `UnsupportedDType` — the prevention loop
  for the Phi-3.5 Q5_K class of bug.

Empirical validation against two real model families is complete: Phi-3.5
GGUF and Gemma 2 GGUF both generate text identical to the HF reference,
and the imperative escape hatch (Decision 5) was **not** used for either —
the spec expressed both corrections. TinyLlama GGUF and Phi-3.5 GGUF
regressions remain green; lib 409/0/0; CI dual-blocking green.

What v1 deliberately does NOT include, consistent with this ADR:

- Graph builders (`build_llama` / `build_phi3` / `build_gemma2`) — Decision
  2; topology, not tables.
- GGUF dtype decoders themselves — toolkit *declares* required coverage,
  does not generate decoders.
- Adapter selection, `ConfigPolicy` semantics, tokenizer — out of toolkit
  scope by prior phases.

What stays explicitly deferred (AT-4 territory, no commitment in v1):

- A YAML / JSON external template format (Decision 1 + "Why Rust data,
  not YAML/JSON (now)"): no serialized contract is frozen.
- A scaffolding generator.
- A public Adapter SDK: the module is still "deliberately not a public
  SDK yet". The non-goal — **no automatic or magic support for new
  models** — holds in v1 and is restated here. A new family still
  requires a graph builder, numeric validation, and explicit review.

Triggers for revisiting (unchanged from the original "Trigger to revisit"
section): a new family that needs the imperative escape hatch (record it,
do not speculatively generalize); promotion of the adapter layer to a
public SDK (reconsider the Rust-data vs serialized-template decision);
new GGUF dtype in `decode_tensor` (the per-adapter declaration is the
single place adapters opt in).
