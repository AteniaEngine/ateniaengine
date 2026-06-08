# MoE Adapter Specification — Audit (MOE-INTEGRATE-1, design-only)

> **Update (MOE-INTEGRATE-1 implemented + MOE-ATK-DECL-1).** The v1 `moe` DSL section +
> `MoeFamilyKind` resolver + typed layout enums shipped in `src/adapter_toolkit/moe_spec.rs`
> (Mixtral/Qwen YAML, describe+validate). **MOE-ATK-DECL-1** then added a **declarative MoE
> family structural spec** (`src/adapter_toolkit/moe_family_spec.rs`) extending it with the
> DeepSeek/MLA + DeepSeek-V3 routing axes and `preset`s reproducing the **four**
> certified/mechanism families (Mixtral, Qwen-MoE, DeepSeek-V2-Lite, DeepSeek-V3 routing L0),
> with equivalence tests vs the authoritative runtime (`MoeFamily::descriptor`, `v3_router`).
> Still **describe + validate only — not replacing certified paths; no execution; no new
> family support claimed**. The resolver-bridge-to-`MoeRuntime` remains **MOE-INTEGRATE-2**.
> See `docs/HANDOFF_MOE_ATK_DECL_1.md`.

**Audit only — no code, no commits, no manifests, no execution.** Designs a
minimal **MoE Adapter Specification v1** for the Adapter Toolkit, grounded in
what already exists, so MOE-INTEGRATE-2 can wire the productive path. Sources:
`src/adapter_toolkit/{dsl,spec}.rs`, `src/nn/llama/moe_config.rs`,
`src/moe/{family,detect,binding,convention,runtime}.rs`,
`docs/ADAPTER_TOOLKIT_V2.md`, `docs/MOE_PRODUCTION_READINESS_AUDIT.md`,
`docs/MODEL_FAMILY_VALIDATION.md`, `docs/STATUS.md`.

## FASE 1 — How the dense Adapter Toolkit describes a family

ATK v2 is a **declarative, non-authoritative** front-end: `AdapterDsl`
(`dsl.rs`) is a serde schema (`deny_unknown_fields`) with optional sections —
`config` (rope), `weights` (fused QKV/MLP), `attention` (mha/gqa/mqa + kv_heads),
`tokenizer` (eos/turn-terminators) — and a required `family` string.
`spec.rs::resolve_family` maps that string to one of the **7 dense** v1 families
(`llama/qwen2/qwen3/gemma2/gemma3/phi3/mistral`); `generator.rs` produces a live
`AteniaModelAdapter`. The DSL **declares expectations** used for validation /
introspection; `config.json` stays the runtime source of truth.

**Key finding:** the toolkit has **zero MoE awareness**. `deny_unknown_fields`
means a MoE spec cannot even be authored today, and `resolve_family` has no MoE
branch — so ATK cannot currently describe or select a MoE family. (Confirmed:
no `num_experts` / `block_sparse_moe` / `router` / `shared_expert` anywhere in
`src/adapter_toolkit/`.)

## FASE 2 — What a MoE descriptor needs vs what already exists

| Information a MoE adapter needs | Already exists? | Where |
|---|---|---|
| **Family** (Mixtral / Qwen-MoE / DeepSeek-MoE) | ✅ recognised (not in ATK) | `moe/family.rs::MoeFamily` + `classify_family` (from tensor names) |
| **Routed expert count** | ✅ | `moe_config.rs::num_experts` (normalises `num_experts`/`num_local_experts`/`n_routed_experts`) |
| **Top-k (experts per token)** | ✅ | `moe_config.rs::experts_per_token` (+ `experts_per_token_or`, clamped) |
| **Shared experts** (presence/count/FFN size) | ✅ | `moe_config.rs::{has_shared_experts,num_shared_experts,shared_expert_intermediate_size}` |
| **Shared-expert gating** (sigmoid-gated vs ungated) | ⚠️ partial | encoded in `family.rs` descriptor + `convention.rs` (auto-select from `shared_expert_gate`); **not a declarative field** |
| **Routing convention** (renormalize top-k) | ✅ | `moe_config.rs::norm_topk_prob` + `family.rs::renormalizes_topk` (Mixtral default true) |
| **Expert FFN size** | ✅ | `moe_config.rs::expert_intermediate_size` (`moe_intermediate_size`) |
| **Tensor naming / layout** (classic `w1/w3/w2` vs `gate/up/down_proj` vs packed `gate_up_proj+down_proj`; router `block_sparse_moe.gate`/`mlp.gate`/`mlp.router`; shared `mlp.shared_expert.*`) | ⚠️ as strings | `family.rs::MoeFamilyDescriptor.{router_naming,expert_layout}` are **`&'static str` informational**; the real binding is in `detect.rs` + `binding.rs` |
| **Validation rules** (config↔tensors) | ✅ | `family.rs::validate_family_config` (expert count, top-k bound, shared agreement) |
| **A bridge ATK-spec → executing adapter** | ❌ | the dense ATK builds an `AteniaModelAdapter`; there is **no** equivalent that selects/produces the working `moe::runtime::MoeRuntime` |
| **Structured tensor spec** (like dense `FamilyTensorSpec`/`NameTable`/`TransformRule`) for experts | ❌ | dense has it (`model_adapters/tensor_spec.rs`); MoE naming is informational strings, not a structured, transform-bearing table |

**Conclusion:** ~80% of the *information* a MoE adapter needs already exists and
is battle-tested (the `MoeRuntime` consumes it to run a real 14.3 B Qwen-MoE
end-to-end). The gaps are (a) a **declarative DSL surface** to author it, (b) a
**MoE-family resolution path** in the toolkit, (c) a **structured tensor-layout
spec** (currently strings), and (d) the **resolver bridge** to the runtime.

## FASE 3 — MoE Adapter Specification v1 (proposed, declarative)

Mirror the dense pattern: one **optional** `moe` section on `AdapterDsl`,
declarative + non-authoritative (`config.json`/tensors remain the source of
truth; the section declares expectations for validation + selection). Authoring
example (YAML):

```yaml
family: qwen-moe            # NEW family selector: mixtral | qwen-moe | (deepseek-moe deferred)
architecture: Qwen2MoeForCausalLM
attention:
  type: gqa
  kv_heads: auto
moe:                        # NEW optional section
  experts: auto            # routed-expert count: integer or `auto` (defer to config.json)
  top_k: auto              # experts_per_token: integer or `auto`
  shared_expert:
    present: true          # or `auto` (infer from tensors/config)
    gating: sigmoid        # sigmoid (Qwen) | none/ungated (Mixtral) | auto
  routing:
    renormalize_topk: auto # bool or `auto` (Mixtral default true; Qwen norm_topk_prob)
  experts_layout: classic  # classic | packed | auto
  tensor_naming: qwen      # qwen | mixtral_block_sparse | mixtral_packed | auto
tokenizer:
  eos_tokens: [151643, 151645]
```

### The seven required descriptors (and their backing)

1. **family** — `MoeFamilyKind { Mixtral, QwenMoe }` (DeepSeek deferred). Resolves
   via a new `resolve_moe_family` mirroring `resolve_family`. Backed by
   `moe::family::MoeFamily`.
2. **experts** — routed count; `auto` ⇒ `moe_config::num_experts`. Validated
   against tensor-implied count (`family::validate_family_config`).
3. **top_k** — `auto` ⇒ `moe_config::experts_per_token_or`. Clamped to `[1,experts]`.
4. **shared_expert** — `{ present, gating }`; `auto` ⇒
   `moe_config::has_shared_experts` + `convention.rs` for gating.
5. **routing.renormalize_topk** — `auto` ⇒ `moe_config::renormalize_topk_or(family_default)`.
6. **tensor_naming / experts_layout** — a structured enum over the three known
   layouts (classic `w1/w3/w2`, classic `gate/up/down_proj`, packed
   `gate_up_proj+down_proj`) + router naming + shared naming. `auto` ⇒
   `family::classify_family`. **This is the one piece that needs a new structured
   type** (today it is `&'static str`).
7. **validation rules** — reuse `family::validate_family_config` verbatim:
   config↔tensor expert count, top-k ≤ experts, shared-expert agreement; a
   mismatch is a hard, family-named error (no silent acceptance).

### Non-goals for v1 (explicit)
- **Does not execute or lift fail-loud.** Like the dense ATK, the spec *names*
  and *validates*; selecting/running `MoeRuntime` is MOE-INTEGRATE-2.
- **DeepSeek-MoE / MLA excluded** (out of charter; MLA is a separate runtime).
- **No new config source of truth** — `auto` everywhere defers to
  `config.json` via `moe_config`; explicit values are *expectations* checked
  against the model, never injected into it.

## FASE 4 — Wiring vs real engineering, and risks

**Mostly wiring (known, incremental):**
- Add the optional `MoeSection` serde struct to `AdapterDsl` (+ keep
  `deny_unknown_fields`).
- Add `resolve_moe_family` + a `MoeFamilyKind` selector (mirror `resolve_family`).
- A resolver that folds `MoeSection` + `moe_config::MoeConfig` into the inputs the
  existing `MoeRuntime` already takes (the runtime is unchanged).
- Reuse `family::validate_family_config` for the validation rules.

**Real (but small) engineering:**
- A **structured MoE tensor-layout spec** — promote `MoeFamilyDescriptor`'s
  `&'static str` naming into typed enums (`ExpertLayout`, `RouterNaming`,
  `SharedExpertNaming`) with the per-layout transform knowledge that today lives
  imperatively in `binding.rs`. This is the only piece that is an abstraction
  design, not a copy-through. (~1 new module, M.)
- Typed **gating/convention** fields (sigmoid vs ungated; renorm) instead of
  string descriptors — small, mirrors `convention.rs`.

**Risks:**
- **Drift / dual source of truth.** The `moe` section must stay declarative: if
  authors can set `experts: 8` and it silently overrides `config.json`, that is
  a correctness trap. Keep `auto` the default and explicit values *validation
  only* (the dense ATK already enforces this discipline — follow it exactly).
- **Convention heterogeneity.** Mixtral (renorm, no shared) vs Qwen (sigmoid
  shared, no renorm) vs DeepSeek (multiple shared, MLA) must map exactly to
  `family.rs`/`convention.rs` or the spec and runtime diverge. Mitigation: the
  spec resolves to the *same* `MoeFamily`/convention the runtime already uses; no
  parallel logic.
- **Scope creep into execution.** v1 must not lift the dense loader's fail-loud
  or route to the runtime — that is MOE-INTEGRATE-2, gated + tested separately.
- **`deny_unknown_fields` migration.** Adding the section is backward-compatible
  (optional), but every existing spec/test must still parse — covered by the
  existing ATK test battery.

## FASE 5 — Deliverable

### Proposed design (summary)
A single optional `moe` section on `AdapterDsl` + a `MoeFamilyKind` resolver,
declarative and `auto`-defaulting to the existing `moe_config`/`family.rs`, with
a new typed tensor-layout spec replacing today's informational strings, and the
existing `validate_family_config` as the validation rule set. It *describes and
validates* a MoE family; it does **not** execute (that is the next milestone).

### Gaps found
1. No MoE surface in the ATK DSL (`deny_unknown_fields` blocks authoring).
2. No MoE-family resolution in `spec.rs` (only 7 dense families).
3. MoE tensor naming/layout is informational `&'static str`, not a structured
   `FamilyTensorSpec`-grade table.
4. No resolver bridge from an ATK spec to the working `MoeRuntime`.
(Config plane, family recognition, and validation **already exist and are
proven**.)

### Estimated complexity
- DSL `MoeSection` + `MoeFamilyKind` resolver + validation reuse: **S–M, high
  confidence** (data exists; mirror the dense path).
- Structured tensor-layout spec (typed enums + transforms from `binding.rs`):
  **M, medium confidence** (the one real abstraction).
- Total MOE-INTEGRATE-1 (the *spec*, no execution): **M**. The execution wiring
  (fail-loud lift + `generate`→`MoeRuntime` routing) is **MOE-INTEGRATE-2**, a
  separate M.

### Recommended roadmap
1. **MOE-INTEGRATE-1 (this spec, then implement):** add the `moe` DSL section +
   `MoeFamilyKind` resolver + the typed tensor-layout spec + reuse
   `validate_family_config`. **Qwen-MoE first** (the family that already runs
   end-to-end), Mixtral second. Declarative + validating only; fail-loud
   unchanged; tests mirror the dense ATK battery. **No execution.**
2. **MOE-INTEGRATE-2:** the resolver bridge + dense-loader fail-loud lift behind
   the opt-in + `generate`→`MoeRuntime` routing. *This* is where MoE reaches the
   normal runtime.
3. **MOE-REAL-MIXTRAL / MOE-CERT-SCALE:** real Mixtral weights + the scale
   certification methodology (the open research item).
4. DeepSeek-MoE / MLA and performance (interactive speed) remain later/ongoing.

**Strategic note:** MOE-INTEGRATE-1 is **low-risk, mostly wiring**, and it is the
correct first step because it makes MoE *describable* by the same toolkit that
already governs the 7 dense families — without touching execution, fail-loud, or
the runtime. The certified math + the working `MoeRuntime` are the assets this
spec front-ends; it does not re-derive them.

*Audit only — no source/manifests changed, no commits, no execution.*
