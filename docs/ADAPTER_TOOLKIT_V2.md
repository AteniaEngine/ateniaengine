# Adapter Toolkit v2 — User & Engineering Manual

This manual documents Adapter Toolkit v2 (ATKv2) as it is actually
implemented in `src/adapter_toolkit/`. It describes real behaviour,
not intentions: every command, field, rule and limitation below
corresponds to code that exists and is covered by tests.

---

## 1. Overview

> **MoE note (MOE-INTEGRATE-1 / MOE-ATK-DECL-1).** Beyond the 7 dense families, the
> toolkit carries a **declarative MoE spec layer** that *describes and validates* MoE
> families **parallel to the handwritten runtime paths** — it does **not** execute or
> route. `src/adapter_toolkit/moe_spec.rs` is the YAML `moe` section (Mixtral/Qwen);
> `src/adapter_toolkit/moe_family_spec.rs` (`MoeStructuralSpec` / `MoeArch::preset`) adds
> the DeepSeek/MLA + DeepSeek-V3 routing axes and reproduces the four
> certified/mechanism families (Mixtral, Qwen-MoE, DeepSeek-V2-Lite, DeepSeek-V3 routing
> L0), equivalence-tested against the runtime. **MOE-INTEGRATE-2** adds the **opt-in resolver
> bridge** (`moe_resolver.rs`, `MoeSpecResolver`): a spec resolves to a `ResolvedMoeRuntimePlan`
> and, behind `ATENIA_ENABLE_MOE=1`, delegates to the unchanged certified `MoeRuntime` —
> **handwritten certified paths remain default**, V3 routing is **mechanism-only /
> non-runnable**, no new family support claimed. **Not replacing the certified paths.** The
> productive CLI/`generate` wiring is the next step (MOE-PRODUCT-1).
> See `docs/MOE_ADAPTER_SPEC_AUDIT.md`, `docs/HANDOFF_MOE_ATK_DECL_1.md`,
> `docs/HANDOFF_MOE_INTEGRATE_2.md`.

### What it is

Adapter Toolkit v2 is a **declarative layer on top of** the v1
model-adapter system (`src/model_adapters/`). It lets a model be
described by a small YAML/JSON document — an *adapter spec* —
instead of a hand-written Rust adapter.

### What problem it solves

In v1, supporting a model family means writing a Rust adapter: a
struct implementing seven traits, registered in a static array.
That is the right tool for a genuinely new architecture, but it is
heavyweight when a model is just a known family with a different
EOS set, a different tokenizer detail, or a per-checkpoint quirk.

ATKv2 covers that gap. For any model that belongs to one of the
seven families the engine already supports, you can describe it in
a few lines of YAML, validate it, and inspect the resulting adapter
— with no Rust, no recompilation, and no risk to the runtime core.

### Philosophy: simple on the outside, powerful on the inside

- **Simple on the outside.** The minimum valid spec is one line:
  `family: llama`. A first-time user never sees the trait system.
- **Powerful on the inside.** A v2 adapter is a v1 adapter
  *parameterised by data*. It delegates every graph-building and
  weight-mapping operation to the hand-written v1 adapter for the
  family. Nothing is reimplemented; nothing is generated as Rust
  source; the runtime core and the graph builders are never
  touched.

### Hard boundaries

ATKv2 **never**:

- modifies the runtime core or the graph builders;
- duplicates v1 adapter logic;
- generates Rust code at runtime;
- invents a model architecture the engine cannot run.

The DSL *parameterises* the seven existing families. It does not
add new ones. A model outside those families (e.g. classic Falcon)
is a typed error, never a forced fallback — see [Limitations](#11-limitations).

---

## 2. Quick Start

**1. Write a spec.** Create `my-model.yaml`:

```yaml
family: llama
format: safetensors
```

**2. Load it:**

```text
atenia load my-model.yaml
```

**3. Expected output:**

```text
Adapter Toolkit v2 — generated adapter
======================================
  family          : Llama
  architecture    : LlamaForCausalLM
  model_type      : llama
  v1 base adapter : llama
  declared features (validated, not applied — config.json is authoritative):
    rope          : standard
    attention     : mha
    kv_heads      : auto (from config.json)
    fused_qkv     : false
    fused_mlp     : false
  tokenizer:
    eos_tokens    : (from config.json / GGUF)
    turn_terminators: (none declared)
  overrides       : (none)

  status          : OK (adapter constructed; generation not run)
```

`load` parsed the spec, validated it, resolved the family to the v1
`llama` adapter, and built the v2 adapter. It did **not** run
generation — that is by design (see [§4.1](#41-atenia-load)).

`load` is one step of a larger flow. The full path from a model on
disk to generated text is **`inspect` → `load` → `generate`**:
`inspect` bootstraps a spec, `load` **validates** it, and the
existing `atenia generate` command **executes** generation. ATKv2
owns the first two steps; generation stays exactly where it was.

---

## 3. Typical Workflow

This is the real, end-to-end path for bringing up a model.

**1. You have a model.** A directory with a `config.json` plus
safetensors, or a single `*.gguf` file.

**2. Bootstrap a spec with `inspect`:**

```text
atenia inspect ./models/my-model
```

`inspect` auto-detects the family, attention shape, EOS set and
(for HF) the RoPE variant, and prints a YAML spec. Save the YAML
block to a file, e.g. `my-model.yaml`.

**3. Adjust the YAML if needed.** For most HF models the generated
spec is complete. For a GGUF of a long-context model you may need
to add `config.rope: longrope` by hand — `inspect` emits a note
when it could not detect something (see [§7](#7-auto-detection-inspect)).

**4. Validate with `load`:**

```text
atenia load my-model.yaml
```

A clean run ends with `status: OK`. `load` does not run
generation — it parses, validates and builds the adapter.

**5. If something fails, investigate with `debug`:**

```text
atenia debug my-model.yaml
```

`debug` prints the verbose report — the resolved v1 adapter, its
capabilities, the tensor-name mapping — plus any warnings.

**6. Generate text with the existing command:**

```text
atenia generate --prompt "..." --model ./models/my-model --max-tokens 100
```

Generation is **not** an ATKv2 command. `load` validates the
adapter; `generate` executes the model. The two are deliberately
separate.

---

## 4. CLI Commands

ATKv2 adds three subcommands to the `atenia` binary. All three
print to stdout on success and exit `0`; on any toolkit error they
print `error: ...` to stderr and exit `2`.

### When to use each command

| Command | Use it to… |
|---------|------------|
| `atenia load`    | **Validate** an adapter spec — parse, check the rules, build the adapter, confirm `status: OK`. |
| `atenia debug`   | **Investigate issues** — the verbose report shows the resolved v1 adapter, its capabilities, the tensor-name mapping, and any warnings. |
| `atenia inspect` | **Bootstrap a new spec** — auto-detect a model directory and emit a starting YAML. |

None of the three runs generation. For that, use `atenia generate`
(see [§3](#3-typical-workflow) for the full `inspect → load →
generate` flow).

### 4.1 `atenia load`

```text
atenia load <FILE>
```

**What it does:**

1. Parses the DSL file (`.yaml`, `.yml`, or `.json`).
2. Runs the declarative validators (see [§9](#9-validation-rules)).
3. Resolves the spec to a v1 family and base adapter.
4. Builds the v2 adapter.
5. Prints a summary report.

**What it does NOT do:** `load` never runs text generation, never
loads model weights, and never builds a compute graph. It is a
spec-construction and validation command. To generate text, use
the existing `atenia generate` command — it is unchanged.

**Example:**

```text
atenia load config/adapters/qwen.yaml
```

```text
Adapter Toolkit v2 — generated adapter
======================================
  family          : Qwen2
  architecture    : Qwen2ForCausalLM
  model_type      : qwen2
  v1 base adapter : qwen2
  declared features (validated, not applied — config.json is authoritative):
    rope          : standard
    attention     : gqa
    kv_heads      : auto (from config.json)
    fused_qkv     : false
    fused_mlp     : false
  tokenizer:
    eos_tokens    : [151643, 151645]
    turn_terminators: ["<|im_end|>"]
  overrides       : 1 declared
    - deepseek-distill     eos_tokens=[151643]

  status          : OK (adapter constructed; generation not run)
```

If validation produces warnings (not errors), `load` still
succeeds and prints them under a `warnings:` section.

### 4.2 `atenia debug`

```text
atenia debug <FILE>
```

`debug` runs the same pipeline as `load` but prints a **verbose**
report that additionally includes:

- the v1 capability flags the adapter inherits (`hf_safetensors`,
  `gguf`, `store_backed_generation`, `fused_qkv_weight_mapping`,
  `fused_gate_up_mapping`, `gemma2_softcaps`);
- a sample of the GGUF→HF tensor-name mapping the adapter exposes.

**When to use it:** use `debug` when you need to confirm *which*
v1 adapter a spec resolved to and *what it can do* — for example,
to verify that a `family: phi` spec really inherits Phi-3's fused
QKV/gate-up weight mapping, or to see how a GGUF tensor name is
rewritten to its HF equivalent.

**Example (excerpt):**

```text
atenia debug config/adapters/phi.yaml
```

```text
  v1 capabilities (delegated):
    hf_safetensors          : true
    gguf                    : true
    store_backed_generation : true
    fused_qkv_weight_mapping: true
    fused_gate_up_mapping   : true
    gemma2_softcaps         : false
  GGUF -> HF tensor-name sample (v1 mapping, delegated):
    token_embd.weight        -> model.embed_tokens.weight
    output_norm.weight       -> model.norm.weight
    blk.0.attn_norm.weight   -> model.layers.0.input_layernorm.weight
    blk.0.attn_q.weight      -> model.layers.0.self_attn.q_proj.weight
    blk.0.attn_qkv.weight    -> model.layers.0.self_attn.qkv_proj.weight
    blk.0.ffn_down.weight    -> model.layers.0.mlp.down_proj.weight
    blk.0.ffn_up.weight      -> model.layers.0.mlp.gate_up_proj.weight
```

### 4.3 `atenia inspect`

```text
atenia inspect <MODEL_DIR>
```

`inspect` reads a model directory and **auto-detects** a valid
adapter spec for it. It is the fastest way to produce a starting
YAML for a model you have on disk.

**How it works:**

1. Detects the format — a `*.gguf` file ⇒ GGUF; a `config.json`
   ⇒ HF safetensors.
2. Reads the family/architecture, attention shape, EOS set, and
   (for HF) the RoPE variant.
3. Emits a YAML spec, followed by a resolved-spec preview.

The emitted YAML is **self-checked**: `inspect` validates and
resolves it before printing, so its output is always loadable by
`atenia load`.

**Example:**

```text
atenia inspect ./models/gemma-2-2b-it
```

```text
# Adapter Toolkit v2 — generated by `atenia inspect`
# This YAML is loadable directly with `atenia load`.

family: gemma2
architecture: Gemma2ForCausalLM
model_type: gemma2
format: safetensors
config:
  rope_theta: 10000.0
attention:
  type: gqa
  kv_heads: 4
tokenizer:
  eos_tokens:
  - 1
  - 107

Adapter Toolkit v2 — resolved spec
==================================
  family          : Gemma2
  ...
```

To save the YAML to a file, keep the lines from `family:` down to
the blank line before `Adapter Toolkit v2 — resolved spec` (the
preview block is not YAML). See [§7](#7-auto-detection-inspect)
for auto-detection details and limitations.

---

## 5. DSL Reference

One schema, [`AdapterDsl`], covers all three authoring levels.
Every field except `family` is optional. Unknown keys are a hard
parse error (`deny_unknown_fields`) — a typo never passes silently.

### 5.1 Basic level

```yaml
family: llama          # required
format: safetensors    # optional, informational
quant: Q4_K_M          # optional, informational
```

| Field    | Required | Meaning |
|----------|----------|---------|
| `family` | yes | One of: `llama`, `qwen` / `qwen2`, `qwen3`, `gemma` / `gemma2`, `gemma3`, `phi` / `phi3`, `mistral`. |
| `format` | no | `safetensors` or `gguf`. Informational — the loader detects the real format from the model directory. |
| `quant`  | no | Quantisation tag (e.g. `Q4_K_M`). Informational only. |

A one-line `family: llama` file is a complete, valid spec.

### 5.2 Intermediate level

```yaml
family: qwen
architecture: Qwen2ForCausalLM
model_type: qwen2
tokenizer:
  eos_tokens: [151643, 151645]
  turn_terminators:
    - "<|im_end|>"
```

| Field          | Meaning |
|----------------|---------|
| `architecture` | Explicit HF `architectures[0]` string. Overrides the family-derived default. Must be one v1 recognises. |
| `model_type`   | Explicit HF `model_type`. Must be one v1 recognises — an unknown value is a hard error, not a silent fallback. |
| `tokenizer.eos_tokens` | Explicit multi-EOS token id set. |
| `tokenizer.turn_terminators` | Turn-terminator token strings (resolved to ids by the generation pipeline). |

### 5.3 Advanced level

```yaml
family: phi
architecture: Phi3ForCausalLM

config:
  rope: longrope
  partial_rotary_factor: 0.75      # only with rope: partial
  rope_theta: 10000.0

weights:
  fused_qkv: true
  fused_mlp: true
  split_strategy: phi_qkv

attention:
  type: gqa
  kv_heads: auto                   # an integer, or the keyword `auto`

tokenizer:
  turn_terminators:
    - "<|end|>"

overrides:
  deepseek-distill:
    tokenizer:
      eos_tokens: [1, 106]
```

#### `config`, `weights`, `attention` are declarative

**This is the single most important thing to understand about the
DSL.** The `config`, `weights`, and `attention` sections are
**declarative, not authoritative**:

- They describe what the model is *expected* to be.
- The toolkit uses them for **validation** (catching inconsistent
  combinations) and for **introspection** (the `load` / `debug`
  reports).
- They do **NOT** replace or mutate the model's `config.json` /
  GGUF metadata. The model's own config remains the single source
  of truth for the runtime.

For example, `attention.kv_heads: 8` does not *set* the KV-head
count — the real count comes from `config.json`. Declaring it lets
the validator check your expectation and lets `debug` show it. The
introspection output labels this explicitly:

```text
  declared features (validated, not applied — config.json is authoritative):
```

This is a deliberate design choice: the YAML must never appear to
control runtime behaviour it does not actually control. If you
need to change a model's real config, edit its `config.json`.

#### `overrides`

`overrides` is a map of per-checkpoint adjustments, keyed by an
opaque label. Each entry layers its `tokenizer` / `config` block on
top of the base spec. Use it when one checkpoint of a family needs
a different EOS set — the canonical case is a DeepSeek-R1 distill
that shares the Qwen architecture but uses a different stop token.

---

## 6. Real Examples

Five working examples ship under `config/adapters/`. Each one loads
directly with `atenia load config/adapters/<name>.yaml`.

### `llama.yaml` — simple level

```yaml
family: llama
format: safetensors
```

The minimal spec. Architecture, attention shape, RoPE and EOS are
all read from the model's own `config.json` at load time. Resolves
to the v1 `llama` adapter.

### `qwen.yaml` — intermediate level

```yaml
family: qwen
architecture: Qwen2ForCausalLM
format: safetensors
attention:
  type: gqa
  kv_heads: auto
tokenizer:
  eos_tokens: [151643, 151645]
  turn_terminators:
    - "<|im_end|>"
overrides:
  deepseek-distill:
    tokenizer:
      eos_tokens: [151643]
```

Declares the GQA attention shape, the multi-EOS set, the Qwen turn
terminator, and a per-checkpoint override for a DeepSeek distill.
Resolves to the v1 `qwen2` adapter.

### `gemma.yaml` — intermediate level

```yaml
family: gemma
architecture: Gemma2ForCausalLM
format: safetensors
config:
  rope: standard
attention:
  type: gqa
  kv_heads: auto
tokenizer:
  turn_terminators:
    - "<end_of_turn>"
```

Gemma 2 text model. Soft-caps and the dual-norm topology are owned
by the v1 `gemma2` adapter and need no DSL declaration. Resolves to
the v1 `gemma2` adapter.

### `phi.yaml` — advanced level

```yaml
family: phi
architecture: Phi3ForCausalLM
format: safetensors
config:
  rope: longrope
weights:
  fused_qkv: true
  fused_mlp: true
  split_strategy: phi_qkv
attention:
  type: gqa
  kv_heads: auto
tokenizer:
  turn_terminators:
    - "<|end|>"
```

Phi-3. Declares LongRoPE and the fused QKV / gate-up weight layout
with its split strategy. These declarations *describe* what the v1
`phi3` adapter already implements — the DSL does not change the
builder, it documents and validates the model's shape.

### `mistral.yaml` — intermediate level

```yaml
family: mistral
architecture: MistralForCausalLM
format: safetensors
config:
  rope: standard
attention:
  type: gqa
  kv_heads: auto
tokenizer:
  eos_tokens: [2]
```

Mistral 7B dense. Pure Llama topology with GQA. Resolves to the v1
`mistral` adapter, which delegates graph build and weight mapping
to the Llama path. Mixtral / Mistral-MoE are out of scope.

---

## 7. Auto-Detection (`inspect`)

`atenia inspect <dir>` builds a spec from a model directory.

### What it detects

| Property | HF safetensors (`config.json`) | GGUF (`*.gguf`) |
|----------|-------------------------------|-----------------|
| Format | `config.json` present | `*.gguf` present |
| Family / architecture | `architectures[0]` / `model_type` | `general.architecture` |
| Attention shape | `num_attention_heads` vs `num_key_value_heads` ⇒ MHA / GQA / MQA | `<arch>.attention.head_count` vs `head_count_kv` |
| EOS set | `eos_token_id` (scalar or array) | `tokenizer.ggml.eos_token_id` |
| RoPE variant | `rope_scaling` / `partial_rotary_factor` | **not detectable — see below** |
| `rope_theta` | `rope_theta` | `<arch>.rope.freq_base` |

### Limitations

**GGUF cannot expose the RoPE variant.** llama.cpp folds LongRoPE
and partial-rotary scaling into the precomputed `rope_factors`
tensors; it does not store the variant as a metadata key. So when
`inspect` reads a GGUF file it **cannot** tell a plain-RoPE model
from a LongRoPE model. It does **not** guess. It leaves `rope`
unset (the standard path) and emits an explicit note as a comment
in the generated YAML:

```text
# note: GGUF metadata does not expose the RoPE variant; `rope` was left
# note: unset (standard). Long-context models (e.g. Phi-3 LongRoPE) may
# note: need `config.rope: longrope` added to the YAML by hand —
# note: auto-detection cannot recover this from a GGUF file.
```

### How to correct it manually

If you are inspecting a GGUF of a long-context model (a Phi-3
128K variant, for example), add the RoPE declaration by hand after
running `inspect`:

```yaml
family: phi
format: gguf
config:
  rope: longrope        # <-- added manually; inspect cannot detect this from GGUF
```

For HF safetensors models this is not an issue — `config.json`
fully specifies the RoPE variant, so safetensors auto-detection is
lossless.

---

## 8. Debugging & Introspection

### Seeing the generated adapter

`atenia load` shows the summary; `atenia debug` shows the verbose
report. Use `debug` to confirm:

- **which v1 adapter** the spec resolved to — the `v1 base adapter`
  line. A `family: qwen` spec must show `qwen2`; a `family: phi`
  spec must show `phi3`.
- **what the adapter can do** — the `v1 capabilities` block.

### Understanding the tensor mapping

The `debug` report prints a GGUF→HF tensor-name sample. This is the
v1 mapping the adapter delegates to. It is the quickest way to see
a family-specific quirk:

- A `phi` adapter maps `blk.0.attn_qkv.weight` to
  `model.layers.0.self_attn.qkv_proj.weight` (fused QKV) and
  `blk.0.ffn_up.weight` to `...mlp.gate_up_proj.weight` (fused
  gate/up).
- A `llama` adapter shows `blk.0.attn_qkv.weight -> (not mapped by
  this family)` — the Llama layout has no fused QKV tensor.

### Verifying overrides

The `load` / `debug` report lists every declared override under the
`overrides` section, with the EOS set each one resolves to:

```text
  overrides       : 1 declared
    - deepseek-distill     eos_tokens=[151643]
```

An override whose `tokenizer` block is absent shows
`eos_tokens=(inherits base)` — confirming it layered nothing and
falls back to the base spec.

---

## 9. Validation Rules

`atenia load` and `atenia debug` validate the spec before building
the adapter. Validation is **fail-loud**: a blocking error stops
the command with exit code `2` and a clear message.

Validation produces two severities, and the distinction is strict:

- **Errors → execution stops.** The command prints the error and
  exits with code `2`. The adapter is **not** built. The spec is
  inconsistent and must be fixed.
- **Warnings → execution continues.** The command still builds the
  adapter and exits `0`. Warnings are printed under a `warnings:`
  section so you can decide whether they matter; they flag an
  unusual-but-not-broken spec.

### Errors (blocking)

| Rule | Message trigger |
|------|-----------------|
| GQA requires `kv_heads` | `attention.type: gqa` without `attention.kv_heads`. |
| Fused QKV requires a split strategy | `weights.fused_qkv: true` without `weights.split_strategy`. |
| `partial_rotary_factor` must not contradict `rope` | `config.partial_rotary_factor` set while `config.rope` is `standard` or `longrope`. |
| `partial_rotary_factor` range | Value not in `(0, 1]`. |
| MQA implies one KV head | `attention.type: mqa` with an explicit `kv_heads` ≠ 1. |
| Unknown family | `family` is not one of the seven supported families. |
| Unknown architecture | An explicit `architecture` v1 does not recognise. |
| Unknown `model_type` | An explicit `model_type` v1 does not recognise. |

### Warnings (non-blocking)

| Rule | Meaning |
|------|---------|
| Fused weights on a non-Phi family | `fused_qkv` / `fused_mlp` declared on a family whose v1 builder has no fused path. The flag changes nothing. |
| LongRoPE on a non-Phi family | `config.rope: longrope` declared where v1 has no LongRoPE parser. |
| Orphan `split_strategy` | `split_strategy` set without `fused_qkv: true` — nothing to split. |
| Empty `eos_tokens` | `tokenizer.eos_tokens: []` — generation would have no EOS stop. |

Warnings do not stop `load`; they are printed under a `warnings:`
section so you can decide whether they matter.

---

## 10. Troubleshooting

This section maps real symptoms to diagnoses. ATKv2 commands
(`load` / `debug` / `inspect`) help with spec-level problems;
weight-level and generation-level problems are diagnosed with the
existing `atenia generate` command.

### The model does not load

**With `atenia load`:** the message names the stage. A parse error
(`adapter DSL parse error: ...`) means the YAML/JSON is malformed
or has an unknown key. A validation error
(`adapter DSL validation error: ...`) means a rule in [§9](#9-validation-rules)
failed. A resolution error (`adapter DSL resolution error: ...`)
means the family / architecture is not one v1 supports.

**With `atenia generate`:** if `load` succeeds but generation fails
at model load, the problem is in the model directory, not the spec
— a missing `config.json`, missing weight files, or a corrupt
GGUF. ATKv2 does not load weights, so `load` cannot catch this.

### Architecture error

`error: adapter DSL resolution error: unknown family ...` or
`... cannot auto-detect a supported family ...` means the model is
not one of the seven supported families. This is expected for
classic Falcon, MoE models, multimodal models — see
[Limitations](#11-limitations). It is not a bug; the toolkit
refuses to pretend it can run an architecture the engine has no
builder for.

### Incorrect tokens / garbled output

This is a generation-level problem, diagnosed with `atenia
generate`, not `load`. Check, in order:

1. **Tokenizer.** Is the model directory's `tokenizer.json` the
   one that matches the weights?
2. **EOS / stop tokens.** If the model rambles past where it should
   stop, the EOS set is wrong. Declare the correct one with
   `tokenizer.eos_tokens`, or use an `overrides` entry for the
   specific checkpoint.
3. **Chat template.** A wrong or missing chat template produces
   coherent-but-off output. Try `atenia generate --no-chat-template`
   to isolate it.

### Empty output

Empty output usually means the very first generated token is an
EOS token — the EOS set or the chat template is wrong. Check the
EOS ids against the model's `tokenizer_config.json`. ATKv2's
`tokenizer.eos_tokens` is the declarative place to record the
correct set.

### Incorrect EOS

Use `atenia debug` to confirm what EOS set the spec declares, and
the `overrides` mechanism if only one checkpoint of a family
differs. Remember: a declared `eos_tokens` is metadata the
generation pipeline consumes — it does not override the model's
`config.json` EOS unless the pipeline is wired to prefer it.

### Incomplete GGUF

If `inspect` fails on a GGUF with `GGUF: missing general.architecture`
or a metadata read error, the GGUF file is truncated or corrupt.
Re-download it. A GGUF that downloaded partially is the most common
cause — verify the file size against the source.

### Small models vs large models

Small instruct models (≤ ~360M) are genuinely more repetitive and
more prone to verbose or off-target output under greedy decoding.
This is a model-capacity property, not an engine or adapter defect.
The same spec and adapter that work for a 7B model work for a 135M
model; the *output quality* differs because the model differs. Do
not chase an "adapter bug" when the first sentence is correct and
the model is simply small.

---

## 11. Limitations

ATKv2 is honest about what it cannot do.

- **Classic Falcon is out of scope.** `FalconForCausalLM` /
  `RWForCausalLM` use LayerNorm (not RMSNorm), parallel attention,
  and a multi-query fused-QKV layout — a distinct architecture with
  no v1 graph builder. A spec or `inspect` targeting classic Falcon
  fails loud with a resolution error. Modern Falcon3 *is* supported:
  it declares `LlamaForCausalLM` and resolves to the `llama` family.
- **MoE is not supported.** There is no mixture-of-experts builder
  in v1. Mixtral / Mistral-MoE are out of scope; the DSL has no MoE
  family.
- **Multimodal / vision / encoder-decoder are not supported.**
  ATKv2 covers dense causal language models only.
- **LongRoPE is not detectable from GGUF.** See
  [§7](#7-auto-detection-inspect). `inspect` emits a note; the user
  must add `config.rope: longrope` by hand for GGUF long-context
  models.
- **`atenia load` does not run generation.** It is a
  spec-construction and validation command by design. Generation
  stays in `atenia generate`.
- **The DSL parameterises; it does not redefine.** `config` /
  `weights` / `attention` are declarative — they do not mutate the
  model's `config.json`.

### Dependency note

The YAML backend uses `serde_yaml 0.9`, which is deprecated
upstream. This is accepted, contained debt: YAML parsing is used
only by the DSL front-end, never on the inference hot path, and the
JSON backend (`serde_json`, fully maintained) is a complete
substitute. A migration TODO is recorded in `Cargo.toml`.

---

## 12. Best Practices

- **Start simple.** Use a one-line `family:` spec first. Add
  sections only when you have a concrete reason — a known EOS
  quirk, a per-checkpoint override.
- **Prefer `inspect` for a new model.** Run `atenia inspect` on the
  model directory and use its output as your starting YAML. It is
  correct by construction for HF models.
- **Use `overrides` for per-checkpoint quirks, not per-family.** If
  *every* checkpoint of a family needs a setting, put it in the
  base spec. Reserve `overrides` for the one distill / fine-tune
  that differs.
- **Treat the declarative sections as documentation + a safety
  net.** Declaring `attention.type: gqa` does not change runtime
  behaviour, but it makes the validator catch a future mistake and
  makes the spec self-documenting.
- **Validate a new model before trusting it.** Run `atenia load`,
  read the warnings, then `atenia debug` to confirm it resolved to
  the v1 adapter you expected.
- **For GGUF long-context models, always check the RoPE note** that
  `inspect` emits, and add `config.rope` manually if needed.
- **Do not fight `config.json`.** If a real config value is wrong,
  fix `config.json`. The DSL is not the place for that.

---

## 13. Design Tradeoffs

A few deliberate decisions shape ATKv2. They are tradeoffs, not
accidents — each one trades a capability for safety or simplicity.

- **The DSL is declarative, not authoritative.** A spec describes
  and validates a model; it does not mutate the model's
  `config.json`. This gives up "configure the model from YAML" in
  exchange for a single, unambiguous source of truth and no silent
  overrides.
- **`load` does not run generation.** It builds and validates the
  adapter, nothing more. This gives up a one-command "load and
  run" in exchange for a fast, side-effect-free command that is
  safe in scripts and CI. Generation stays in `atenia generate`.
- **No new architectures are introduced.** The DSL parameterises
  the seven existing families. This gives up "describe any model
  in YAML" in exchange for never pretending to run an architecture
  the engine has no builder for — unsupported models fail loud.
- **v2 delegates to v1; it does not replace it.** A
  `GeneratedAdapter` wraps a hand-written v1 adapter and forwards
  every trait method. This gives up an independent v2 execution
  path in exchange for zero duplication, zero core changes, and a
  v2 adapter that is behaviourally identical to v1 on the v1
  surface.

---

## 14. Architecture (for engineers)

ATKv2 lives in `src/adapter_toolkit/`. The pipeline:

```text
  .yaml/.json  ──dsl──▶  AdapterDsl
  AdapterDsl   ──spec─▶  ResolvedAdapterSpec
  spec         ──gen──▶  GeneratedAdapter   (impl AteniaModelAdapter)
  adapters     ──reg──▶  AdapterRegistry    (v2-first, v1-fallback)
```

### `AdapterDsl` (`dsl.rs`)

The serde schema. One struct, all sections optional except
`family`. Parses `.yaml`/`.yml` (serde_yaml) and `.json`
(serde_json) into the same type. `deny_unknown_fields` turns a
misspelled key into a hard error.

### `ResolvedAdapterSpec` (`spec.rs`)

The intermediate representation: the validated, normalised form of
an `AdapterDsl`. It holds:

- the DSL `family` mapped to a v1 `ModelFamily` and a base
  architecture string;
- a `FeatureSet` — the normalised pattern catalog (`RopeKind`,
  `AttentionKind`, `KvHeadsResolved`, fused-QKV/MLP flags);
- the per-checkpoint overrides, resolved and layered.

Resolution is also a validator: an unknown family / architecture /
`model_type` is a typed error here.

### `GeneratedAdapter` (`generator.rs`)

A v2 adapter. It holds a `&'static dyn AteniaModelAdapter` — the v1
hand-written adapter for the family — and implements the v1 7-trait
supertrait (`ModelAdapter`, `HfWeightMapper`, `GgufWeightMapper`,
`GgufNameMapper`, `StoreBackedGraphBuilder`, `ResidencyHints`,
`ConfigPolicy`) by **pure delegation** to it.

This is the key design point: a v2 adapter is *behaviourally
identical* to the v1 adapter it wraps on the v1 surface. Graph
topology, weight mapping, and GGUF naming are all v1's. The DSL
adds metadata (EOS set, turn terminators, overrides) exposed via
`GeneratedAdapter::spec()` — it never smuggles anything into the
graph build.

`GeneratedAdapter::from_spec` resolves the base adapter through v1's
own `resolve_adapter` and asserts that the resolved adapter's family
matches the spec's family. A mismatch (e.g. an explicit
`architecture` inconsistent with `family`) fails loud.

### `AdapterRegistry` (`registry.rs`)

A runtime registry of v2 adapters. Resolution is **v2-first,
v1-fallback**:

1. Try the registered v2 adapters (first registered match wins).
2. Fall back to v1's static `resolve_adapter`.
3. If neither resolves, return `Unresolved`.

v1 is never modified and never shadowed for a model that has no v2
spec. Back-compatibility is total.

### Relationship with v1

ATKv2 is strictly additive. It depends on v1's public surface
(`resolve_adapter`, the `AteniaModelAdapter` trait, `ModelMetadata`,
`ModelFamily`) and changes nothing in `src/model_adapters/`, the
runtime core, or the graph builders. v1 adapters keep working
exactly as before; a v2 adapter is a thin, data-driven wrapper over
one of them.

---

## 15. Extending the System

### Adding a new family

A new *family* means a new architecture the engine must be able to
run. **That is a v1 task, not a v2 task.** You write a v1 adapter
(a struct implementing the seven traits, registered in the
`ADAPTERS` array in `src/model_adapters/mod.rs`) and, if the family
has its own tensor layout, a `FamilyTensorSpec` in
`tensor_spec.rs`. Once the v1 adapter exists, ATKv2 picks it up by
adding the family to two small mapping tables:

- `resolve_family` in `spec.rs` — DSL family string ⇒
  `(ModelFamily, architecture, model_type)`;
- `intern_architecture` / `intern_model_type` in `spec.rs` — the
  set of strings v2 will accept explicitly;
- `family_from_hf` / `family_from_gguf` in `inspect.rs` — for
  auto-detection.

ATKv2 cannot add a family on its own, by design: it never invents a
builder.

### Adding a new transform

Load-time tensor transforms live in v1's `TransformRecipe` /
`FamilyTensorSpec` (`src/model_adapters/tensor_spec.rs`). ATKv2 does
not own transforms — it delegates weight mapping wholesale to v1.
A new transform is a v1 change.

### Adding a new validation rule

This *is* a v2 task. Add the rule to `validate.rs`:

- a structural rule (reads only the raw `AdapterDsl`) goes in
  `structural_rules`;
- a family-aware rule (needs the resolved family) goes in
  `family_rules`.

Push a message onto `report.errors` (blocking) or
`report.warnings` (non-blocking), and add a test.

---

## 16. FAQ

**Why doesn't `atenia load` run generation?**
By design. `load` is a spec-construction and validation command —
it builds and checks the adapter, nothing more. Generation has its
own command (`atenia generate`) with its own arguments (prompt,
max-tokens, model directory). Keeping them separate means `load`
is fast, side-effect-free, and safe to run in scripts and CI.

**Why doesn't the YAML change `config.json`?**
Because the model's `config.json` (or GGUF metadata) is the single
source of truth for the runtime, and a YAML file that silently
overrode it would be a footgun. The DSL's `config` / `weights` /
`attention` sections are declarative: validated and shown in
introspection, never applied. If a real config value is wrong, fix
`config.json`.

**Why does Falcon fail?**
Modern Falcon3 does *not* fail — it declares `LlamaForCausalLM` and
resolves to the `llama` family. *Classic* Falcon
(`FalconForCausalLM` / `RWForCausalLM`) fails because it is a
genuinely different architecture — LayerNorm, parallel attention,
multi-query fused QKV — with no v1 graph builder. ATKv2 refuses to
pretend otherwise; it fails loud rather than producing garbage.

**Why doesn't `inspect` detect everything?**
`inspect` detects what the model files actually expose. HF
`config.json` is rich, so HF detection is lossless. GGUF metadata
is leaner — in particular it does not carry the RoPE variant
(llama.cpp folds it into the `rope_factors` tensors). `inspect`
emits a note for what it could not detect rather than guessing. See
[§7](#7-auto-detection-inspect).

**Can a v2 adapter behave differently from the v1 adapter?**
No, not on the v1 surface. `GeneratedAdapter` delegates every v1
trait method to the wrapped v1 adapter. The v2-only additions (EOS
set, turn terminators, overrides) are metadata, exposed separately
via `spec()`.

**Do I need a v2 spec for every model?**
No. v1 still resolves any supported model on its own. A v2 spec is
useful when you want a declared, validated, self-documenting record
of a model — or a per-checkpoint override.

---

## 17. MoE Specification v1 (MOE-INTEGRATE-1)

The toolkit can also **describe and validate** a Mixture-of-Experts family via
an optional `moe:` section. It is **declarative only** — like every other
section it does **not** execute, route, load, or lift the dense loader's
fail-loud guard (that is MOE-INTEGRATE-2). It gives a MoE family the same
authored, validated, self-documenting record the dense families have.

**Contract — no second source of truth.** Every `moe` field defaults to `auto`,
meaning *defer to `config.json`* (parsed by `moe_config`). Explicit values are
**expectations** checked against the model (via the `effective_*` accessors on
`ResolvedMoeSpec`, which combine *declared ⊕ config*), never injected into it. A
dense spec omits `moe:` entirely and behaves exactly as before.

Families: `mixtral`, `qwen-moe` (DeepSeek-MoE is **deferred** — MLA is a separate
runtime). Resolution lives in `src/adapter_toolkit/moe_spec.rs::ResolvedMoeSpec`;
checkpoint validation **reuses** `moe::family::validate_family_config` (no
duplicated logic).

### Example — Qwen-MoE (explicit)

```yaml
family: qwen-moe
architecture: Qwen2MoeForCausalLM
attention:
  type: gqa
  kv_heads: auto
moe:
  experts: 60            # or `auto` → config.json num_experts
  top_k: 4               # or `auto` → num_experts_per_tok
  shared_expert:
    present: true        # or `auto`
    gating: sigmoid      # sigmoid (Qwen) | none/ungated (Mixtral) | auto
  routing:
    renormalize_topk: false   # or `auto` → norm_topk_prob (Qwen default false)
  experts_layout: classic     # classic | packed | auto
```

### Example — Mixtral (minimal, all `auto`)

```yaml
family: mixtral
architecture: MixtralForCausalLM
moe: {}                   # every field auto: experts/top_k from config.json,
                          # no shared expert, renormalise top-k (family default)
```

### `auto` behaviour (resolution chain)

| Field | `auto` resolves to |
|---|---|
| `experts` | `config.json` num_experts / num_local_experts / n_routed_experts |
| `top_k` | `config.json` num_experts_per_tok (clamped to `[1, experts]`) |
| `routing.renormalize_topk` | `config.json` `norm_topk_prob`, else the family default (Mixtral = true, Qwen = false) |
| `shared_expert.present` | `config.json` (shared-expert size / count) |
| `experts_layout` / `router_naming` | detected from the checkpoint tensors at load (layout is checkpoint-dependent for Mixtral) |

### Validation (loud, no silent acceptance)

Resolve-time contradictions are hard errors: `shared_expert.present: false` with
`gating: sigmoid`; a `router_naming` that contradicts the family
(`block_sparse` ↔ non-Mixtral, `mlp_router` ↔ non-Qwen); `top_k > experts`; an
invalid keyword. Against a real checkpoint, `ResolvedMoeSpec::validate_against`
delegates to the runtime's `validate_family_config` (expert count, top-k bound,
shared-expert agreement).

### What v1 does NOT do

No execution, no `generate` → `MoeRuntime` routing, no fail-loud lift, no
runtime / loader / numerics change. The MoE spec front-ends the certified MoE
math and the working `MoeRuntime`; it does not re-derive or run them.

---

*This manual describes Adapter Toolkit v2 as implemented in
`src/adapter_toolkit/`. For the v1 adapter system it builds on, see
`src/model_adapters/`. For the functional validation of the model
families ATKv2 parameterises, see `docs/MODEL_FAMILY_VALIDATION.md`.
For project status and milestone history, see `docs/STATUS.md` and
`docs/MILESTONES.md`.*
