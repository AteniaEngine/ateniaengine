# HANDOFF — MOE-8: conditional expert subgraph execution

Milestone: **MOE-8** (per-expert conditional execution driven by the
scheduler). Correctness-first, CPU-only, experimental, synthetic-fixture-
only. The MOE-5 fused op, MOE-6 primitives and MOE-7 dynamic dispatch are
all **kept intact**. No real Mixtral, no checkpoint loading (MOE-2
fail-loud preserved), no loaders, no adapters, no CLI, no generation, no
CUDA, no tier-planner, no batching, no optimisation. Predecessors:
MOE-0..7 (`8e9d55d`).

## What changed vs MOE-7

MOE-7 kept the conditionality *inside one fused op* (`MoeDynamicDispatch`):
the scheduler saw a single node and did not know which experts ran. MOE-8
makes each expert its **own graph node** (`ConditionalExpert`), so the
scheduler executes **N separate expert steps**, each of which gates itself
on the routing selection. The MoE output is the sum of those per-expert
contribution nodes. The scheduler now *drives* expert-level gating.

```text
MOE-7:  … → [MoeDynamicDispatch]                 (1 node, internal dispatch)
MOE-8:  … → [ConditionalExpert e0] ┐
            [ConditionalExpert e1] ┤→ tree-sum → output   (N nodes, per-expert gating)
            [ConditionalExpert e2] ┤
            [ConditionalExpert e3] ┘
```

## How gating works

`NodeType::ConditionalExpert { layer_id, expert_id, d_model }` takes two
inputs: the model vector and the `MoeTopK` selection tensor
`[idx0, w0, idx1, w1, …]`. Its executor calls
`crate::moe::execute_conditional_expert(layer_id, expert_id, input,
selection)`, which:

- scans the selection for `expert_id` (`expert_weight_in_selection`);
- **if selected** → runs `layer.experts[expert_id].forward(input)` and
  scales it by the routing weight; returns `(contribution, executed=true)`;
- **if not selected** → **never calls `forward`** and returns a zero
  contribution `[d_model]` with `executed=false`.

The forward of an unselected expert is genuinely skipped — gating is real
compute elision, not a multiply-by-zero. The sum of all contribution nodes
equals the MoE output (the zeros from skipped experts add nothing).

## How the scheduler decides execute vs skip

The decision is made **inside the executed step**, driven by the standard
`run_plan` → `execute_single` loop (no scheduler redesign — rule 17 not
triggered). Each `ConditionalExpert` node is an ordinary plan step; when
the scheduler runs it, the node inspects the selection tensor produced
upstream by `MoeTopK` and either runs its expert or emits zeros. The
scheduler thus executes one node per expert and the gating is observable
per node (`executed` flag), unlike MOE-7's opaque single op.

The executed/skipped accounting is verified directly:
`executed_and_skipped_counts_are_correct` asserts exactly `k` experts
execute and `num_experts − k` are skipped for a top-k selection.

## Validation chain

```text
conditional pipeline (N gated nodes + sum)
  == MoeDynamicDispatch (MOE-7)
  == forward_sparse (MOE-4)
  == dense restricted oracle (MOE-3)
```

All within 1e-5. Tests:
`conditional_pipeline_matches_dynamic_dispatch`,
`conditional_pipeline_matches_sparse_reference`,
`conditional_pipeline_matches_dense_oracle`,
plus `conditional_expert_executes_when_selected`,
`conditional_expert_skips_when_not_selected`,
`executed_and_skipped_counts_are_correct`,
`conditional_pipeline_is_deterministic`,
`conditional_expert_rejects_unknown_layer_and_expert`,
`existing_fused_tests_still_pass`.

## What is still missing for real Mixtral

- **Real weights**: experts come from the synthetic registry, not a loaded
  checkpoint. MOE-2 still fails loud on real MoE files.
- **A multi-layer / multi-token model graph**: this is single-layer,
  single-token. Real inference needs the MoE layer wired into the full
  transformer block across all layers and a sequence loop.
- **Loader + adapter integration** (expert tensor mapping → graph nodes):
  untouched here; that is the data-plane work (a later milestone).
- **Optimisation / batching / GPU**: out of scope throughout.

## What was NOT implemented

- No general conditional-subgraph control flow for arbitrary node types —
  only the MoE expert-gating node. No `if`/branch nodes in the scheduler.
- No removal of `MoeDynamicDispatch` / fused op / primitives — all kept.
- No real model, loader, adapter, CLI, generation, CUDA/ROCm/Metal,
  tier-planner, batching, optimisation, or autograd for the new node.

## How this prepares MOE-9

MOE-9 can wire this conditional-expert pipeline into a full transformer
MoE block (multi-layer) and/or begin the data-plane connection (loader →
expert nodes) — now that per-expert scheduler-driven gating is certified
against the whole reference chain.

## Files modified

* `src/amg/nodes.rs` — `ConditionalExpert` variant.
* `src/amg/graph.rs` — forward exec arm + arity-validator arm.
* `src/amg/builder.rs` — `moe_conditional_expert` helper.
* `src/moe/graph_op.rs` — `execute_conditional_expert` +
  `expert_weight_in_selection`.
* `src/moe/mod.rs` — re-exports.
* `tests/moe_conditional_expert_test.rs` — new (9 tests).
* `docs/HANDOFF_MOE_8.md` — this file.

Local validation: `cargo test --lib --release` → **669 passed / 0 failed /
1 ignored**. MOE-5/6/7 integration suites still green.

No loader, adapter-toolkit, CLI, generation, CUDA, ROCm, Metal, or
tier-planner changes. Fused op + primitives + dynamic dispatch intact.
Fail-loud preserved.
