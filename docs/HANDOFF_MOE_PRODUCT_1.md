# HANDOFF — MOE-PRODUCT-1: opt-in MoE in the productive `generate` (resolver-backed)

Milestone: **MOE-PRODUCT-1** — wire MoE into the productive `atenia generate` flow
**opt-in**, with the routing decision taken **through the declarative resolver bridge**
(`MoeSpecResolver`). The **dense path stays intact**, **fail-loud is the default** when
MoE is not enabled, and **no new family support is claimed**.

## Result

`atenia generate <model>` now routes a recognised MoE checkpoint through the resolver:

- **Dense checkpoint** → the existing dense pipeline, **unchanged** (`route → Dense`).
- **MoE, no opt-in** (`ATENIA_ENABLE_MOE` unset) → **fail-loud**, exit 2, with the
  family-aware message from `diagnose_moe` (e.g. "set `ATENIA_ENABLE_MOE=1` to run").
- **MoE runnable + opt-in set** → the controlled MoE runtime (`controlled_moe_generate`,
  the same gated entry `atenia moe-generate` uses) — for the **productively-routable**
  families **Mixtral / Qwen-MoE**.
- **DeepSeek-MoE** → **refused** (certified L3, but productive MLA generate routing is
  deferred — no new family support claimed).
- **DeepSeek-V3 routing** → **non-runnable mechanism-only**, never reachable as a model.
- **Unknown / unsupported / uncertified** → **refused** (fail-loud).

The decision is the new `MoeSpecResolver::route(&MoeDiagnosis) -> MoeRoute`, which maps the
detected family to a `MoeArch` (`arch_for_productive_routing`), **resolves it through the
declarative resolver** (`resolve` → equivalence guard + `runnable` flag), then applies the
opt-in. It is behaviour-equivalent to the lower-level `moe::production::decide_route` for
the productively-routable families (asserted by test), but the runnable decision now flows
through the spec resolver, so a spec that diverges from the handwritten convention or a
mechanism-only arch can never be routed.

## What changed

- `src/adapter_toolkit/moe_resolver.rs`:
  - `MoeSpecResolver::arch_for_productive_routing(MoeFamily) -> Option<MoeArch>` (Mixtral /
    Qwen-MoE → arch; DeepSeek → `None`, productive routing deferred).
  - `MoeSpecResolver::route(&MoeDiagnosis) -> MoeRoute` — the resolver-backed productive
    routing decision (pure; fail-loud by default).
- `src/cli_generate.rs`: the MoE probe in `run()` now calls `MoeSpecResolver::route` instead
  of `moe::production::decide_route` (same `MoeRoute` outcomes for Mixtral/Qwen; dense
  passthrough and the fail-loud message are unchanged). `run_moe_text` +
  `controlled_moe_generate` are unchanged.

## Guardrails (FASE 5)

- **Dense generate untouched:** `route → Dense` falls straight through to the dense
  pipeline; no numerics / thresholds / loader change.
- **Fail-loud default:** a MoE checkpoint without `ATENIA_ENABLE_MOE=1` exits 2 with a clear
  family-aware message; nothing runs.
- **V3 cannot run as a real model:** the V3 routing mechanism is non-runnable and is never
  reachable from a checkpoint family (DeepSeek → `None`), and the resolver's `runnable`
  flag would refuse it regardless.
- **No new family support claimed:** the productive runnable set stays **Mixtral / Qwen-MoE**
  (the families already routed in MOE-INTEGRATE-2). DeepSeek stays refused.
- **Certified paths + manifest + ADR-007 untouched** (docs-only notes where relevant).

## Validation

- 4 new resolver unit tests (`route` dense/needs-opt-in/run/refused, `arch_for_productive_routing`)
  + a new integration test `tests/moe_product_routing_test.rs` that drives
  `MoeSpecResolver::route` on the real Mixtral/Qwen/dense fixtures and asserts it **agrees
  with `decide_route`** on the runnable set.
- Regressions: `moe_integrate_routing_test` + `moe_production_test` green;
  `cargo test --lib adapter_toolkit::` + full `cargo test --lib` green.

## Files

- `src/adapter_toolkit/moe_resolver.rs` (+`route` + `arch_for_productive_routing` + tests),
  `src/cli_generate.rs` (route through the resolver), `tests/moe_product_routing_test.rs`
  (new).
- docs: this handoff + `CLI.md`, `ADAPTER_TOOLKIT_V2.md`, `STATUS.md`.

## Operating it

```
# Dense models: unchanged.
atenia generate <dense-model> --prompt "..."

# MoE (Mixtral / Qwen-MoE): opt-in, experimental.
ATENIA_ENABLE_MOE=1 atenia generate <mixtral-or-qwen-moe> --prompt "..."
# Without the flag: fail-loud with a clear message (exit 2).
```

Caveats: MoE generate is **experimental / opt-in**, CPU-only, slow (bounded-cache disk-tier
forward); **MoE-certified L3 is active-path-certified, NOT the dense ADR-004 `CERTIFIED`**;
L4 reserved/unreachable.

## Next recommended step

- **MOE-PRODUCT-2:** productive DeepSeek-V2-Lite (MLA) generate routing (un-defer DeepSeek
  in `arch_for_productive_routing`) with its own EOS/tokenizer validation, behind the opt-in.
- **Performance:** CPU forward throughput / bounded-cache disk-tier (MoE generate is slow).
- **CLI UX:** surface the MoE status (`atenia diagnose`-style) and the opt-in hint in help.

## Risks / caveats

- `MoeSpecResolver::route` is behaviour-equivalent to `decide_route` for the current set; if
  the productive set is later extended (e.g. DeepSeek), the equivalence assertion must be
  updated deliberately.
- The opt-in env (`ATENIA_ENABLE_MOE`) is process-global; the routing tests are sequential to
  avoid toggling it under a concurrent assertion.
