# HANDOFF — MOE-INTEGRATE-2: opt-in resolver bridge (declarative spec → runtime)

Milestone: **MOE-INTEGRATE-2** — add an **opt-in resolver bridge** from the declarative
`MoeStructuralSpec` (MOE-ATK-DECL-1) to a runtime plan, and, behind the opt-in, to the
**unchanged certified `MoeRuntime`**. **Handwritten certified paths remain default**; no
numerics / threshold / manifest change; **no new family support claimed**; **V3 routing is
mechanism-only, non-runnable**.

## Result

`src/adapter_toolkit/moe_resolver.rs` — `MoeSpecResolver`:

- `resolve(&MoeStructuralSpec) -> ResolvedMoeRuntimePlan` (pure, always safe): turns a spec
  into a typed runtime plan — `MoeFamily`, execution **convention** (`Atenia` /
  `HuggingFaceQwen`), `RoutingPlan` (`SoftmaxTopK{renorm}` / `SigmoidTopK{renorm}` /
  `V3NoAuxGroupLimited`), attention, expert layout, shared mode, dense-first, YaRN,
  disk-tier hint, a representative `V3RouterConfig` (V3 only), the manifest **cert scope**
  (informational), and a **`runnable`** flag.
- `runtime_gate(&plan, opt_in) -> Result<()>`: a non-runnable arch is refused; a runnable
  arch without the opt-in is refused; runnable + opt-in passes.
- `load_runtime(spec, dir) -> Result<MoeRuntime>`: the **opt-in bridge** — resolve, enforce
  the gate (`runnable` + `ATENIA_ENABLE_MOE=1`), then delegate to the **unchanged**
  `MoeRuntime::load_from_dir`. Fail-loud (before touching disk) for a non-runnable arch or a
  missing opt-in.

## Specs it resolves

| Arch | Family | Convention | Routing | Attention | Runnable |
|---|---|---|---|---|---|
| Mixtral | `Mixtral` | `Atenia` | softmax top-k, renorm | GQA | ✅ |
| Qwen-MoE | `QwenMoe` | `HuggingFaceQwen` | softmax top-k, no renorm | GQA | ✅ |
| DeepSeek-V2-Lite | `DeepSeekMoe` | `Atenia` | softmax top-k, renorm | MLA + YaRN, dense-first 1 | ✅ |
| DeepSeek-V3-routing | `DeepSeekMoe` | `Atenia` | **V3 noaux/group-limited** | MLA + YaRN, dense-first 3 | ❌ **mechanism-only** |

The resolved **convention** matches the handwritten `RealMoeLayer::resolve_convention`
rule exactly (HuggingFaceQwen iff a sigmoid-gated shared expert, else Atenia).

## Equivalence (FASE 4) — cannot silently diverge

The resolver's **equivalence guard** cross-checks the spec against the authoritative
handwritten `MoeFamily::descriptor()`: if `renormalize_topk` or `shared_present` disagree
with the runtime convention, `resolve` returns `EquivalenceMismatch` (hard error). Tests
assert each preset resolves to the handwritten family/convention/routing/attention/
dense-first/YaRN/disk-tier/runnable expectations, and that a deliberately divergent spec
(Mixtral without renorm) is rejected.

## Guardrails (FASE 5)

- **V3 routing cannot run as a real model:** `runnable=false` → `runtime_gate` /
  `load_runtime` return `NonRunnable` even with the opt-in, before any I/O.
- **Opt-in required:** a runnable arch without `ATENIA_ENABLE_MOE=1` → `OptInRequired`.
- **Unsupported / invalid specs fail loud:** an invalid spec (e.g. MLA on Mixtral) is
  rejected by `resolve` via the `Spec` error.
- **Default runtime unchanged:** the productive routing
  (`crate::moe::production::decide_route`), the dense loader fail-loud, the numerics, and
  the certification manifest are untouched. The bridge is an additive, opt-in layer.

## Validation

- 10 new resolver unit tests (resolve-equivalence for the 4 archs, convention rule, V3
  non-runnable, opt-in gate, divergent-spec + invalid-spec rejection,
  load_runtime-fails-loud-before-disk).
- `cargo test --lib adapter_toolkit::` + `moe::` green; full `cargo test --lib` green; the
  three certified families and the productive path unchanged.

## Files

- `src/adapter_toolkit/moe_resolver.rs` (new), `src/adapter_toolkit/mod.rs` (+module +
  re-exports).
- docs: this handoff + `ADAPTER_TOOLKIT_V2.md`, `MOE_ADAPTER_SPEC_AUDIT.md`, `STATUS.md`,
  `POST_MIXTRAL_L3_ROADMAP_AUDIT.md`.

## What this does NOT do

- It does **not** wire MoE into the productive CLI / `generate` default; the productive
  `decide_route` still routes Mixtral/Qwen only and is unchanged. `load_runtime` is an
  internal opt-in entry, not the default path.
- It does **not** lift the dense loader fail-loud, change numerics, lower thresholds, or
  enable any new family. DeepSeek-V2-Lite resolving as runnable reflects its existing L3
  certification + runtime; productive routing of DeepSeek remains deferred (separate work).

## Next recommended step

**MOE-PRODUCT-1 (future):** wire a resolver-selected runnable plan into the productive
`generate` path (extend `decide_route` to consult the resolver) behind the opt-in, with
CLI surfacing — the actual productization, gated + tested separately. Performance (CPU
forward throughput / bounded-cache disk-tier) and DeepSeek productive routing are parallel
tracks.

## Risks / caveats

- The bridge is a parallel layer; if the handwritten conventions change, the equivalence
  guard + tests are what fail. Kept in lockstep with `MoeFamily::descriptor` and
  `RealMoeLayer::resolve_convention`.
- `load_runtime` for a runnable arch with the opt-in delegates to the real
  `MoeRuntime::load_from_dir`, which needs a real checkpoint dir (not exercised in unit
  tests; the gate logic is tested deterministically via `runtime_gate`).
