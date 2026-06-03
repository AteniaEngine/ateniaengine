# HANDOFF — MODEL-INTAKE-1: "Say Yes More Often, Safely"

The executive coverage audit ([docs/MODEL_COVERAGE_EXECUTIVE_AUDIT.md]) found
the **#1 coverage blocker** was the **hard reject by architecture string**: any
`architectures[0]` not among the seven registered families was refused at load,
so many genuinely Llama-compatible checkpoints never even got a chance to run.
This milestone raises real coverage **without adding families, and without
touching Numeric Policy / CUDA / MoE** — by inserting an explicit, auditable
compatibility layer that decides *yes (compatible, uncertified)* or *no (with an
actionable reason)* instead of a blanket reject. **Safety > coverage**: certified
families are byte-for-byte unchanged, and nothing clearly-incompatible runs.

## Where the decision was, and where it is now

The dense load path (`nn/llama/pipeline.rs`) already reads the authoritative
`architectures[0]` and resolves an adapter via
`model_adapters::resolve_adapter(&metadata)`. The old code did
`resolve_adapter(...).ok_or_else(reject)?` — a **hard reject** on `None`.

Now: native resolution is **unchanged** (a registered adapter is used exactly as
before — certified/supported families never touch the new code). Only on `None`
do we consult the new **compatibility layer** (`model_adapters::compat`), which
returns `Accept { adapter, status, warnings }` or `Reject { message }`.

## FASE 2 — Compatibility layer (`src/model_adapters/compat.rs`)

`resolve_intake(architecture, model_type, config, generic_opt_in) -> IntakeOutcome`,
consulted only for non-native architectures:

1. **Allowlist hit** → run topology checks → Accept (status `Allowlisted`) or
   Reject (checks failed).
2. **Unknown + no opt-in** → Reject with the `ATENIA_INTAKE_GENERIC=1` hint.
3. **Unknown + opt-in** → run topology checks → Accept (status `Generic`,
   loudly UNCERTIFIED) or Reject.

Purely additive: the native path is a separate `match` arm, so certified
families are guaranteed identical behaviour.

## FASE 3 — Known-compatible allowlist (evidence-gated)

`LLAMA_COMPATIBLE_ALLOWLIST: &[AllowlistEntry { architecture, base_architecture,
evidence }]`. **Deliberately small** and **evidence-only** (never by intuition).
Key insight: the vast majority of Llama-compatible models (Vicuna, NousHermes,
SmolLM, OpenLLaMA, TinyLlama, **Yi's HF releases**, …) already declare
`architectures: ["LlamaForCausalLM"]` and therefore resolve **natively** — they
never reach the allowlist. The list only covers *distinct* arch strings that are,
by documented topology, the identical Llama decoder. Seeded entries:

| architecture | → base | evidence |
|---|---|---|
| `LLaMAForCausalLM` | `LlamaForCausalLM` | legacy capitalisation (original LLaMA / older converters) — byte-identical decoder + tensor names |
| `YiForCausalLM` | `LlamaForCausalLM` | Yi (01.AI) Technical Report adopts the Llama architecture (RMSNorm+RoPE+SwiGLU+GQA), standard Llama tensor names; trust_remote_code variant exposes this distinct string |

Adding an entry is a one-line edit and **must** carry a concrete topology
reference. A unit test asserts every allowlist base resolves natively and every
allowlisted arch is itself non-native (so the layer can never shadow a family).

## FASE 4 — Generic Llama-compatible path (opt-in)

For an **unknown** architecture, `ATENIA_INTAKE_GENERIC=1` opts into running it
through the Llama adapter **iff** the config passes the topology checks. It is:
- **off by default** → current reject behaviour preserved;
- **loud** → every acceptance logs `architecture "X" accepted via … (UNCERTIFIED)`
  plus each non-fatal warning, on `stderr`, at load;
- **fail-loud downstream** → weight loading still errors on any tensor-name /
  shape mismatch, so an accept never silently corrupts — a wrong guess fails
  cleanly at bind time.

## FASE 5 — Compatibility checks (`check_llama_topology`)

Returns `Ok(warnings)` (plain Llama decoder; warnings are non-fatal divergences)
or `Err(reasons)` (every hard incompatibility). **Hard failures** (refuse — a
generic Llama build cannot reproduce them faithfully):
- non-positive `hidden_size` / heads / kv-heads / `intermediate_size` /
  `vocab_size` / `num_hidden_layers` / `rope_theta`; non-finite/≤0 `rms_norm_eps`;
- `hidden_size % num_attention_heads != 0` (without explicit `head_dim`);
- `num_attention_heads % num_key_value_heads != 0` (GQA grouping undefined);
- **specialised-family fields present** → require a dedicated adapter, never
  mis-run as Llama: `attn_logit_softcapping`, `final_logit_softcapping`,
  `query_pre_attn_scalar` (Gemma 2); `rope_local_base_freq`,
  `sliding_window_pattern` (Gemma 3); `partial_rotary_factor` (Phi-4).

**Non-fatal warnings** (accept, but surface): `sliding_window` (generic path uses
full attention → diverges past the window); `rope_scaling` (Llama-family scaling
rules applied — verify against source).

## FASE 6 — Failure modes

Every rejection is an explicit `PipelineError::Loader(InvalidFormat(message))`
naming the architecture and the reason(s); unknown-without-opt-in names the
`ATENIA_INTAKE_GENERIC` escape hatch and the supported list. **No silent
fallback** anywhere.

## FASE 7 — Tests

- `model_adapters::compat` units (9): allowlist accept (no opt-in); unknown
  reject (with env hint); generic accept (clean config); generic reject
  (indivisible heads / GQA misgrouping); allowlisted-but-soft-cap reject;
  sliding-window warning-not-fatal; plain-Llama topology passes; allowlist bases
  native + allowlisted arch non-native.
- `tests/model_intake_compat_test.rs` (5, crate-boundary + real `from_json_str`):
  allowlisted accept; unknown reject w/ hint; generic accept; native arch never
  in allowlist; plain-Llama topology passes.

## FASE 8 — Visibility (`atenia capabilities`)

The capabilities report now lists the **known-compatible** allowlist
(`arch -> base (uncertified)`) and the `ATENIA_INTAKE_GENERIC` opt-in, sourced
from the compat layer (single source of truth) — JSON and human output.

## Deliverable answers

1. **New models that now pass:** checkpoints declaring an allowlisted distinct
   arch string (`LLaMAForCausalLM`, `YiForCausalLM` trust_remote_code variant);
   **and**, under `ATENIA_INTAKE_GENERIC=1`, any unknown architecture whose
   `config.json` is a structurally-plain Llama decoder (RMSNorm + RoPE + SwiGLU +
   GQA, standard tensor names) — subject to fail-loud weight binding. (Models
   that already declared `LlamaForCausalLM` were already passing natively.)
2. **Still rejected:** unknown architectures without the opt-in; any config with
   specialised-family fields (Gemma soft-caps, Gemma 3 dual-RoPE, Phi-4 partial
   rotary) — they need a dedicated adapter; structurally-invalid configs
   (indivisible heads, GQA misgrouping, non-positive dims); and, unchanged,
   MoE / DeepSeek / multimodal / encoder-decoder (out of this milestone's scope).
3. **Allowlist:** a small, evidence-gated table `arch -> base + evidence`,
   consulted only for non-native archs, gated by the topology checks, extensible
   one line at a time with a required topology reference.
4. **Generic path:** opt-in (`ATENIA_INTAKE_GENERIC=1`), runs an unknown arch as
   Llama iff topology checks pass, loudly UNCERTIFIED, fail-loud at weight bind.
5. **Risks:** an allowlisted/generic model with matching tensor names but subtly
   different numerics (e.g. norm placement) would run **uncertified** — mitigated
   by (a) the topology checks rejecting known-divergent shapes, (b) loud
   UNCERTIFIED logging, (c) the default-off opt-in, and (d) fail-loud weight
   binding. No numerical certificate is implied by acceptance.
6. **Estimated coverage gained:** modest-but-real on the residual distinct-string
   Llama-compatible tail (the bulk was already native); the larger structural win
   is converting a **silent/blanket reject** into an **explicit, opt-in,
   topology-checked** decision — the audit's #1 blocker is removed without
   loosening safety.
7. **Tests:** 9 unit + 5 integration, all green.
8-9. **Commit / CI:** see git log + the push; CI must be green on both blocking
   jobs.
10. **Recommended next:** **CERTIFY-BREADTH-1** — extend the f64 reference
   pipeline to Phi-3 + Gemma 2 so "supported" families are also *certified*; then
   `FORMAT-INTAKE-1` (`.bin` / GPTQ-AWQ read path). Both raise trustworthy
   coverage and make the just-landed intake layer pay off across more of the
   catalog. (Not recommended: exotic new families or MoE-perf — measured/Out of
   scope.)

## Files

- `src/model_adapters/compat.rs` (new) — allowlist + generic path + topology
  checks + 9 unit tests.
- `src/model_adapters/mod.rs` — `pub mod compat;`.
- `src/nn/llama/pipeline.rs` — additive intake decision on the `resolve_adapter`
  `None` branch (native path unchanged).
- `src/cli/diagnostics.rs` — `atenia capabilities` surfaces the allowlist + opt-in.
- `tests/model_intake_compat_test.rs` (new) + `docs/HANDOFF_MODEL_INTAKE_1.md`
  (this) + `docs/STATUS.md`.
