# HANDOFF — MOE-ATK-DECL-1: declarative MoE family structural spec

Milestone: **MOE-ATK-DECL-1** — add a **declarative spec layer** to the Adapter
Toolkit that describes MoE families across every structural axis and **reproduces the
certified families** from typed presets, **parallel to the handwritten paths** and
**not replacing the certified paths yet**. **No runtime/loader/manifest change; no new
family support claimed.**

## Result

A new declarative module `src/adapter_toolkit/moe_family_spec.rs` expresses the
**structural shape** of a MoE family as typed data and certifies, by test, that each
**preset reproduces the authoritative handwritten conventions**. It extends the
MOE-INTEGRATE-1 `moe_spec.rs` (Mixtral/Qwen YAML front-end) with the DeepSeek/MLA + V3
axes — describe + validate **only**, no execution.

It can express the **four certified-or-mechanism families**:

| Arch (`MoeArch`) | Bridges to | Routing | Shared | Attention | Dense-first | Status |
|---|---|---|---|---|---|---|
| `Mixtral` | `MoeFamily::Mixtral` | softmax top-k, renorm | none | GQA | 0 | MoE-certified **L3** |
| `QwenMoe` | `MoeFamily::QwenMoe` | softmax top-k, no renorm | sigmoid-gated | GQA + qkv bias | 0 | MoE-certified **L3** |
| `DeepSeekV2Lite` | `MoeFamily::DeepSeekMoe` | softmax top-k, renorm | ungated | MLA + YaRN | 1 | MoE-certified **L3** |
| `DeepSeekV3Route` | `MoeFamily::DeepSeekMoe` | **V3 noaux + group-limited + scaling** | ungated | MLA + YaRN | 3 | routing **L0 mechanism only** |

## The spec (axes covered)

`MoeStructuralSpec` is a typed record over: expert layout (`ClassicW1W3W2` /
`ClassicGateUpDown` / `Packed`), router naming, shared-expert presence + naming +
gating (`Sigmoid` / `Ungated`), routing scheme (`SoftmaxTopK` / `SigmoidTopK` /
`V3NoAuxGroupLimited`), top-k renormalisation, **V3 router params** (`n_group`,
`topk_group`, `routed_scaling_factor`, `norm_topk_prob`), attention (`Mha` / `Gqa` /
`Mla`), qkv bias, YaRN, dense-first layer count, and disk-tier policy. `MoeArch::preset`
returns the canonical spec per family; `from_label`/`label` round-trip.

**Reuse, no duplication:** the typed enums `ExpertLayout` / `RouterNaming` /
`SharedExpertNaming` / `SharedGating` come from `moe_spec.rs`; the family bridge is
`crate::moe::family::MoeFamily`; the V3 params build a real
`crate::moe::v3_router::V3RouterConfig` (the certified mechanism) for validation.

## Equivalence (FASE 4) — the spec cannot silently diverge

The tests assert each declarative preset matches the **authoritative runtime sources**:

- `arch.to_moe_family()` equals the expected `MoeFamily` (the runtime's enum).
- `preset().renormalize_topk` **==** `MoeFamily::descriptor().renormalizes_topk`, and
  `preset().shared_present` **==** `descriptor().has_shared_expert` — i.e. the spec's
  convention equals the handwritten convention, for all four archs.
- MLA appears **iff** the arch is DeepSeek; Mixtral/Qwen are GQA.
- The `DeepSeekV3Route` preset's V3 params build a **valid `V3RouterConfig`** and route
  through `v3_router::v3_route` (256 experts / top-8) — the declarative spec agrees with
  the L0-certified routing mechanism.

## Fail-loud (FASE 5)

`MoeStructuralSpec::validate()` rejects: V3 routing without V3 params (or vice versa),
MLA on a non-DeepSeek arch (or a DeepSeek arch missing MLA), sigmoid shared-gating with
no shared expert, and structurally invalid V3 router params (delegated to the
authoritative `v3_router` validator). Unknown arch labels error.

## What this does NOT do (scope guards)

- **Not replacing the certified paths.** The handwritten `MoeRuntime` assembly remains
  the execution source of truth; this layer only *describes and validates*.
- **No execution / routing / loader / fail-loud lift / manifest change.**
- **No new family support claimed.** A preset is a structural description. DeepSeek-V3
  as a **model** is **not supported** (real-weight provisioning-blocked); only its
  **routing mechanism** is L0. `MoeArch::is_runnable_model()` returns `false` for it.

## Validation

- 13 new unit tests in `moe_family_spec.rs` (presets validate, label round-trip,
  family/renorm/shared equivalence to the runtime descriptor, MLA-deepseek-only, V3
  preset ↔ v3_router, 4 fail-loud cases).
- `cargo test --lib adapter_toolkit::` green (MOE-INTEGRATE-1 `moe_spec` + dense ATK
  untouched); full `cargo test --lib` green.

## Files

- `src/adapter_toolkit/moe_family_spec.rs` (new), `src/adapter_toolkit/mod.rs` (+module
  + re-exports; `AttentionKind` deliberately not re-exported to avoid colliding with the
  dense `spec::AttentionKind`).
- docs: this handoff + `ADAPTER_TOOLKIT_V2.md`, `MOE_ADAPTER_SPEC_AUDIT.md`, `STATUS.md`,
  `POST_MIXTRAL_L3_ROADMAP_AUDIT.md`.

## Next recommended step

**MOE-INTEGRATE-2 (deferred, separate milestone):** the resolver bridge from a
declarative spec to the working `MoeRuntime` + the dense-loader fail-loud lift behind the
opt-in + `generate`→`MoeRuntime` routing. *That* is where MoE reaches the productive
runtime. Until then this layer stays describe-and-validate only.

## Risks / caveats

- The `DeepSeekV2Lite` / `DeepSeekV3Route` structural values (dense-first counts, YaRN,
  V3 knobs) are taken from the public configs / the L0 mechanism; they are *structural
  descriptions*, validated against the runtime where an authoritative source exists
  (family descriptor, v3_router) and documented where it is config-derived.
- This is a parallel layer; it must be kept in sync if the handwritten conventions ever
  change — the equivalence tests are the guard that fails if they drift.
