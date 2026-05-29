# Atenia CLI — User & Engineering Manual

This manual documents the `atenia` command-line interface as it is
actually implemented. Every command, flag, exit code and error
code described below corresponds to code that exists in the
repository and is covered by tests. It describes real behaviour,
not intentions.

---

## 1. Overview

### What the Atenia CLI is

`atenia` is the single command-line binary of Atenia Engine — a
hardware-adaptive LLM inference runtime written from scratch in
Rust. The CLI is the user-facing surface: it loads checkpoints,
generates text, runs an interactive chat, validates declarative
adapter specs, and diagnoses the host and individual models.

### What it lets you do

- **Generate text** from a checkpoint (`generate`).
- **Chat interactively** with a model (`chat`).
- **Diagnose** the host environment and a specific model before
  running it (`doctor`, `diagnose`, `capabilities`).
- **Work with Adapter Toolkit v2** declarative adapter specs
  (`load`, `debug`, `inspect`).

### Philosophy: simple for users, powerful for engineers

- **Simple for users.** The common path is one command:
  `atenia generate --model <dir> --prompt "..."`. Errors are
  human-readable, with a plain explanation and a fix. Default
  output is clean.
- **Powerful for engineers.** Global flags expose log levels
  (`--verbose`, `--debug`, `--trace`), a log file, a per-run trace
  id, and machine-readable JSON output. Diagnostics commands
  report the exact host and model state.

The CLI never hides failure: every error is explicit, every exit
code is stable, and unsupported inputs fail loud rather than
producing wrong output.

---

## 2. Installation & Running

### Building the binary

Atenia is built with Cargo. The default build produces a working
`atenia` binary:

```text
cargo build --release --bin atenia
```

The binary is then at `target/release/atenia`. This manual writes
commands as `atenia <...>`; substitute the built path, or install
it on `PATH`:

```text
cargo install --path .
```

### Build flavours

- **Default build** — CPU path plus, when a CUDA toolkit is
  present at build time, the CUDA backend. This is the build used
  throughout this manual.
- **CPU-only build** — produced on a host without CUDA; the engine
  is vendor-agnostic and runs on CPU alone.

The `run` subcommand needs the default `demo` feature (enabled by
default). The `probe` subcommand needs `--features hw-probe`.

### Minimum requirements

- **CPU.** Any x86-64 CPU. AVX2 is strongly recommended — without
  it the matmul path is much slower. `atenia doctor` reports the
  detected ISA.
- **RAM.** Depends on the model. Small models (TinyLlama 1.1B,
  SmolLM2, Llama 3.2 1B) run comfortably in 16 GB. The 13B-class
  target needs roughly 28 GB at BF16.
- **CUDA / GPU.** Optional. With an NVIDIA GPU and driver the
  engine uses VRAM with RAM/disk tiering; without one it runs on
  CPU. `atenia doctor` reports CUDA availability.

### Expected model directory layout

Atenia loads two kinds of checkpoint:

- **HuggingFace safetensors** — a directory containing
  `config.json`, one or more `*.safetensors` files (single-file or
  sharded with `model.safetensors.index.json`), and `tokenizer.json`.
- **GGUF** — a directory containing a single `*.gguf` file. The
  tokenizer is embedded in the GGUF file.

`--model` always points at the **directory**, not at an individual
file.

---

## 3. Quick Start

### Check the host

```text
atenia doctor
```

Reports CPU, RAM, CUDA and the build flavour. A good first step
before running any model.

### Generate

```text
atenia generate --model models/llama-3.2-1b-instruct --prompt "Hello"
```

The generated text streams to stdout; progress and statistics go
to stderr.

### Chat

```text
atenia chat --model models/llama-3.2-1b-instruct
```

Starts an interactive REPL. Type a message and press Enter; type
`/help` for commands, `/exit` (or Ctrl+D) to leave.

### Diagnose

```text
atenia diagnose --model models/llama-3.2-1b-instruct
```

Runs a pre-flight check of the model directory — format, family,
tokenizer, adapter resolution — without generating anything.

---

## 4. Command Overview

| Command | Purpose | Typical user |
|---------|---------|--------------|
| `generate` | One-shot text generation from a prompt | everyone |
| `chat` | Interactive multi-turn conversation | everyone |
| `download` | Fetch a curated, public checkpoint from Hugging Face | everyone |
| `quickstart` | First-run guided onboarding (`doctor` → `download` → `diagnose` → `chat`) | first-time |
| `doctor` | Diagnose the host: CPU, RAM, CUDA, build | everyone |
| `diagnose` | Pre-flight check of a specific model | everyone |
| `capabilities` | List supported families, formats, quants | engineer |
| `inspect` | Auto-detect a model and emit an adapter spec | engineer |
| `load` | Parse and validate an adapter spec file | engineer |
| `debug` | Like `load`, with a verbose adapter report | engineer |
| `search` | **Experimental (AQS).** Render a quantization certification report + draft manifest from a results file | engineer |

Three further subcommands exist but are outside the scope of this
manual: `run` (the tri-mode killer demo), `probe` (hardware
enumeration, needs `--features hw-probe`), and `explain` (a legacy
v13 narrative tool). They are preserved for compatibility.

---

## 5. Typical Workflows

### Workflow 1 — First-time user

Check the host, check the model, then generate:

```text
atenia doctor
atenia diagnose --model models/llama-3.2-1b-instruct
atenia generate --model models/llama-3.2-1b-instruct --prompt "Hello"
```

`doctor` confirms the environment can run models at all;
`diagnose` confirms this specific model is supported; `generate`
runs it. If `diagnose` reports a problem, fix it before generating.

### Workflow 2 — Debugging a model

When a model behaves unexpectedly, work from auto-detection down to
a full report:

```text
atenia inspect models/my-model            # auto-detect an adapter spec
atenia load my-model.yaml                  # validate the spec
atenia debug my-model.yaml                 # verbose adapter report
atenia diagnose --model models/my-model    # full model pre-flight
```

`inspect` produces a YAML spec; `load` validates it; `debug` shows
the resolved v1 adapter, its capabilities, and the tensor-name
mapping; `diagnose` ties it together against the real directory.

### Workflow 3 — Interactive usage

```text
atenia chat --model models/llama-3.2-1b-instruct
```

Inside the session:

```text
You> What is the capital of France?
The capital of France is Paris.
You> /history
User: What is the capital of France?
Assistant: The capital of France is Paris.
You> /reset
You> /exit
```

`/reset` (or `/clear`) starts a fresh conversation; `/history`
prints the turns so far; `/exit` leaves.

---

## 6. Commands

### 6.1 `generate`

One-shot text generation. Loads a checkpoint, applies the model's
chat template (unless disabled), runs greedy decoding until EOS or
the token limit, and streams the result to stdout. Designed for
scripting and pipelines.

**Flags:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <dir>` | — | Model directory (required). |
| `--prompt <text>` | — | Input prompt (required). |
| `--max-tokens <n>` | `100` | Maximum new tokens; EOS halts earlier. |
| `--output <text\|json>` | `text` | Output format. |
| `--no-chat-template` | off | Feed the prompt verbatim, no chat template. |
| `--no-progress` | off | Suppress the load/prefill heartbeat dots. |
| `--cache-dir <dir>` | — | Disk-tier scratch directory (reserved). |

**Example:**

```text
atenia generate --model models/tinyllama-1.1b \
  --prompt "What is the capital of France?" --max-tokens 32
```

**Expected output.** The generated text on stdout; on stderr the
load message, a prefill heartbeat, and a final statistics line
(`Generated: N tokens in Xs ... [EOS]`). With `--output json` a
single JSON object is printed on stdout instead (see [§11](#11-json-output)).

**Typical errors.** Model directory missing
(`E-IO-NOT-FOUND`), `config.json` missing (`E-CONFIG-MISSING`),
`tokenizer.json` missing (`E-TOKENIZER-MISSING`), load or
generation failure (`E-GENERATION-FAILED`).

### 6.2 `chat`

Interactive multi-turn REPL. Keeps the conversation in memory and
applies the model's chat template across the whole history.
Responses are streamed token-by-token to stdout.

**Flags:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <dir>` | — | Model directory (required). |
| `--max-tokens <n>` | `256` | Maximum new tokens per assistant turn. |
| `--temperature <f>` | `0.0` | Accepted but **not wired** — see below. |
| `--no-chat-template` | off | Use a plain `User:`/`Assistant:` transcript. |

**In-session commands:**

| Command | Effect |
|---------|--------|
| `/help` | List the commands. |
| `/history` | Print the conversation so far. |
| `/reset` | Clear the conversation history. |
| `/clear` | Alias for `/reset`. |
| `/exit` | Leave the chat. `/quit` and Ctrl+D also work. |

Anything that is not a `/command` is sent to the model as a
message. Empty input is ignored. The pipeline is **lazy-loaded** on
the first real message, so the commands work immediately.

`--temperature` is accepted for forward compatibility but does
nothing yet: chat uses greedy decoding. A non-zero value only
prints a warning. See [Limitations](#14-limitations).

The conversation history (`/history`) format:

```text
User: <message>
Assistant: <response>
```

See [§12](#12-chat-behavior) for prompt construction details.

### 6.3 `doctor`

Diagnoses the host and build. Read-only; runs nothing.

```text
atenia doctor
```

It checks: Atenia version, OS/architecture, CPU (threads + ISA),
RAM (total/available), build flavour (CPU-only vs CUDA-enabled),
CUDA runtime availability, backends, supported weight formats,
supported GGUF quantisations, and whether the temp directory is
writable.

**Output.** A checklist; each line is tagged `[ ok ]`, `[warn]`,
`[fail]`, or `[info]`:

```text
Atenia Engine — system diagnostics
  [info] version            0.1.0
  [info] os                 windows / x86_64
  [ ok ] cpu                24 threads, AVX2, FMA
  [ ok ] ram                16.8 GiB available / 31.7 GiB total
  [info] build              CUDA-enabled
  [ ok ] cuda               available
  [ ok ] backends           cpu, cuda
  [ ok ] formats            safetensors, GGUF
  [ ok ] gguf quants        F32, F16, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K
  [ ok ] temp dir writable  /tmp
```

`[info]` lines are neutral facts; `[warn]` flags a sub-optimal but
working condition (e.g. low RAM); `[fail]` would mark a real
problem. `doctor` exits `0` unless a check fails. With `--json` it
emits a JSON report instead.

### 6.4 `diagnose`

Pre-flight check of a **specific model directory**, before you try
to generate with it. Does not run generation.

```text
atenia diagnose --model models/llama-3.2-1b-instruct
```

**What it validates:** the path exists; the format (safetensors vs
GGUF); the family/architecture is supported; the tokenizer is
present; the inferred adapter spec is valid; the adapter resolves
to a v1 family adapter (a dry run, no weights loaded).

**What it does NOT validate:** it does not load model weights, does
not run a forward pass, and does not decode individual GGUF tensors
— so it cannot detect a single unsupported tensor dtype buried in
an otherwise-valid file. It checks *structure and resolution*, not
numerics.

**Output:**

```text
Atenia Engine — model diagnosis: models/llama-3.2-1b-instruct
  [ ok ] model path        exists
  [ ok ] format            safetensors
  [ ok ] family            llama
  [ ok ] tokenizer         tokenizer.json present
  [ ok ] adapter spec      valid
  [ ok ] adapter resolved  Llama (LlamaForCausalLM)

Result: model is ready to generate.
```

`diagnose` exits `0` when the model is ready, `2` when a check
fails (the report ends with `Result: model is NOT ready`), and
prints a typed error (exit `2`) if the path does not exist at all.
`--json` emits a JSON report with a `ready` boolean.

### 6.5 `capabilities`

Lists what the engine can and cannot do. Static; takes no model.

```text
atenia capabilities
```

It prints: the supported model families; the architectures that
are explicitly out of scope; the supported weight formats; the
supported GGUF quantisations; and the engine features. It is
deliberately honest — the unsupported list is as prominent as the
supported list. `--json` emits the same data as JSON.

### 6.6 `download`

A curated, one-shot model downloader. The catalog is small on
purpose — it lists three public, non-gated Hugging Face
checkpoints across the families Atenia supports, so a first-time
user can go from `cargo install` to a running chat in two
commands.

```text
atenia download list                       # show the catalog
atenia download <alias>                    # download to ./models/<default>
atenia download <alias> --dir DIR          # custom destination
atenia download <alias> --force            # overwrite an existing dir
atenia download <alias> --dry-run          # print the plan, write nothing
atenia download <alias> --no-suggest       # skip the post-download footer
```

**Current catalog (v0.2):**

| Alias | Family | Size | Format |
|-------|--------|------|--------|
| `smollm2-135m` | SmolLM | ~270 MB | safetensors |
| `tinyllama` | Llama | ~2.2 GB | safetensors |
| `qwen2.5-0.5b` | Qwen | ~1 GB | safetensors |

**What it does:** for each file listed in the catalog entry,
fetch it from `https://huggingface.co/<repo>/resolve/main/<file>`
into `<dest>/<file>.partial` then atomically rename. One simple
retry with backoff is applied per file. All progress is written
to **stderr**; stdout is left empty.

**What it explicitly does not do (v1):**

- arbitrary `--hf-repo <id>` downloads — use `huggingface-cli`
  for anything outside the catalog,
- gated or private checkpoints — no OAuth, no `--token`,
- resume of partial downloads — interrupted runs are restarted
  from scratch on re-execution,
- checksum verification — the post-condition check is
  "every expected file exists and is non-empty",
- parallel per-file downloads.

**Exit codes:**

| Code | Error | Meaning |
|------|-------|---------|
| `0` | — | Every catalog file landed cleanly. |
| `2` | `E-DOWNLOAD-UNKNOWN-MODEL` | Alias not in the curated catalog. |
| `2` | `E-DOWNLOAD-DESTINATION-EXISTS` | Destination directory exists and `--force` was not passed. |
| `1` | `E-DOWNLOAD-NETWORK` | Per-file HTTP/TLS/DNS/timeout fault. |
| `1` | `E-DOWNLOAD-INCOMPLETE` | A file fetched but ended empty or missing on disk. |
| `2` | `E-DOWNLOAD-GATED-MODEL` | Catalog entry is marked gated (no entries in v1). |

After a successful download the command prints a "Next:" footer
suggesting `atenia diagnose --model <dest>` and
`atenia chat --model <dest>` so the user has a one-paste path to
verifying the checkpoint and starting a conversation. The footer
is suppressed by `--no-suggest`.

### 6.7 `quickstart`

First-run onboarding. Prints the recommended four-step flow
(`doctor` → `download` → `diagnose` → `chat`) with the exact
commands the user should run, substituting the recommended model
into each step. With `--download` it also runs step 2 immediately,
reusing the CLI-6 downloader.

```text
atenia quickstart                       # print the plan, do nothing
atenia quickstart --download            # also run the download step
atenia quickstart --model tinyllama     # recommend / download tinyllama
atenia quickstart --dir ./scratch       # custom destination
atenia quickstart --no-suggest          # suppress the "Tip:" footer
```

**Defaults.** `--model smollm2-135m` — the smallest entry in the
CLI-6 curated catalog, chosen so a first download finishes in a
coffee break on any connection.

**`--dir` is a base directory, not a final destination.** This is
the one place quickstart intentionally diverges from `atenia
download`. The chosen alias always lands in a per-model
subdirectory underneath the base, so you can run quickstart twice
with two different models pointing at the same `--dir` and they
will not collide:

```text
atenia quickstart                                  # ./models/smollm2-135m
atenia quickstart --dir ./scratch                  # ./scratch/smollm2-135m
atenia quickstart --model tinyllama --dir ./scratch
                                                   # ./scratch/tinyllama-1.1b-chat
atenia quickstart --download --model smollm2-135m --dir ./scratch
                                                   # writes to ./scratch/smollm2-135m
```

`atenia download --dir` keeps its CLI-6 semantics (the literal
destination directory) — power-user command, no surprise wrapping.

**Exit codes.** Inherits the CLI-6 download surface:

| Code | Meaning |
|------|---------|
| `0` | Plan printed, or download succeeded. |
| `2` | `E-DOWNLOAD-UNKNOWN-MODEL` — `--model` was not in the catalog. |
| `1` | `E-DOWNLOAD-NETWORK` / `-INCOMPLETE` — only when `--download` is set. |

**Not implemented (intentional):** interactive prompts, auto-run
of `chat`/`generate`, benchmarks, arbitrary HF repos, JSON output.
The command is deliberately scriptable and non-interactive — the
same output every time you run it.

### 6.8 `load` / `inspect` / `debug`

These three commands belong to **Adapter Toolkit v2** (ATKv2), the
declarative adapter layer. For the full ATKv2 manual see
`docs/ADAPTER_TOOLKIT_V2.md`.

- **`atenia inspect <model_dir>`** — auto-detects a model directory
  and emits a YAML adapter spec that `load` can consume. Use it to
  bootstrap a spec for a model you have on disk.
- **`atenia load <file>`** — parses an adapter spec (`.yaml` /
  `.yml` / `.json`), validates it, builds the v2 adapter, and
  prints a summary. It does **not** run generation.
- **`atenia debug <file>`** — same as `load`, but prints a verbose
  report: the resolved v1 adapter, its capability flags, and a
  sample of the GGUF→HF tensor-name mapping.

`load`, `debug` and `inspect` take a **positional** path argument
(no `--model` flag). They print their report on stdout and exit
`2` on any spec/adapter error.

### 6.9 `search` (experimental — AQS)

`atenia search` is the command-line front-end for **AQS** (Atenia
Quantization Search), an experimental, CPU-only, opt-in research
subsystem. See `docs/AQS_OVERVIEW.md` for the full picture.

```bash
atenia search --results aqs-results.json --report
atenia search --results aqs-results.json --report --manifest
```

It reads a **pre-computed end-to-end results file** (JSON produced by the
AQS end-to-end harness), classifies each candidate policy against the
ADR-004 gate (`certified` / `useful_lossy` / `failed`), ranks them
deterministically, and renders a human report and/or a `3.0.0-draft`
manifest.

> **It does not run arbitrary model certification.** `atenia search` never
> loads a model, never runs a forward, and never produces an F64
> reference. It consumes previously generated evaluation results, and the
> draft manifest is never consumed by the runtime (production numeric
> certification is governed by ADR-004 / ADR-005 — see
> `docs/CERTIFICATION.md`).

Flags:

| Flag | Meaning |
|------|---------|
| `--results <FILE>` | Path to the pre-computed AQS end-to-end results JSON. **Required.** |
| `--report` | Print the human-readable certification table. |
| `--manifest` | Print the experimental `3.0.0-draft` manifest. |
| `--include-gptq` | Reserved for a future model-driving path; **inert** on the results-file path (prints a note). |

If `--results` is omitted the command exits `2` with a clear message that
AQS requires a results file produced by the end-to-end harness. A missing
file or invalid JSON also exits `2`. When neither `--report` nor
`--manifest` is given, the report is shown by default.

The results-file schema (each entry mirrors one end-to-end evaluation):

```json
{
  "model": "tinyllama-1.1b",
  "baseline_memory_bytes": 2260729856,
  "results": [
    {
      "candidate_name": "bf16",
      "max_abs_diff": 0.000063,
      "mean_abs_diff": 0.000008,
      "rmse": 0.00001,
      "argmax_match": true,
      "memory_bytes": 2260729856
    }
  ]
}
```

---

## 7. CLI Flags

### Global flags

These apply to every subcommand. They can appear before or after
the subcommand (`atenia --verbose load x.yaml` and
`atenia load x.yaml --verbose` are equivalent).

| Flag | Meaning |
|------|---------|
| `--quiet`, `-q` | Only human errors on stderr — no logs, no progress, no banners. |
| `--verbose`, `-v` | Show useful progress steps (log level `info`). |
| `--debug` | Show resolved configuration and internal decisions (log level `debug`). |
| `--trace` | Fine-grained CLI-frontier detail (log level `trace`). |
| `--log-level <level>` | Explicit level: `error`, `warn`, `info`, `debug`, `trace`. |
| `--log-file <path>` | Also write logs to this file (created if absent, with parent directories). |
| `--trace-id <id>` | Use this run id instead of an auto-generated one. |
| `--no-color` | Disable colored output. Accepted; colored output is not implemented yet, so this is currently a no-op. |

### Precedence

The effective log level is resolved in this order:

1. An explicit `--log-level` wins over everything.
2. Otherwise `--quiet` wins over `--verbose` / `--debug` / `--trace`.
3. Otherwise `--trace` > `--debug` > `--verbose`.
4. The default is `warn` — a user sees real warnings and errors,
   but no progress or diagnostic chatter.

So `atenia --quiet --debug ...` resolves to `quiet` (level
`error`), but `atenia --quiet --log-level debug ...` resolves to
`debug` because the explicit level wins.

---

## 8. Output Model

The CLI keeps a strict separation between stdout and stderr.

**stdout** carries the *result of the command*:

- generated text (`generate`, `chat`);
- JSON reports (`--output json` / `--json`);
- the adapter report (`load`, `debug`);
- the generated YAML (`inspect`);
- the diagnostic checklist (`doctor`, `diagnose`, `capabilities`);
- the `/history` dump in `chat`.

**stderr** carries everything else:

- log lines (`[INFO]`, `[DEBUG]`, ...);
- human errors (`error[E-...]`);
- progress messages and heartbeats;
- the `You> ` chat prompt and the `Thinking ...` indicator;
- banners.

This means a piped invocation yields a clean result. For example,
capturing only the generated text:

```text
atenia generate --model models/tinyllama-1.1b --prompt "Hi" 2>/dev/null
```

or only the assistant turns of a chat:

```text
echo "Hello" | atenia chat --model models/tinyllama-1.1b 2>/dev/null
```

`chat` also accepts a fully scripted, non-interactive session on
stdin — the assistant turns land on stdout, the commands drive the
session:

```text
printf 'Hello\n/exit\n' | atenia chat --model models/tinyllama-1.1b
```

Logs never contaminate stdout, even at `--trace`.

---

## 9. Error System

Every CLI failure is reported as a structured, human-readable
error with a stable code.

### Format

```text
error[E-CODE]: one-line summary

What happened:
  A plain-language explanation.

How to fix:
  A plain-language instruction.

Check it with:        (optional)
  a concrete command

Technical details:    (optional)
  key: value

Trace:                (when logging is initialised)
  run id: atenia-<...>
  log file: <path>     (only with --log-file)
```

The `error[E-CODE]:` line, `What happened:` and `How to fix:` are
always present. `Check it with:`, `Technical details:` and
`Trace:` appear when relevant.

### Error codes

| Code | Meaning |
|------|---------|
| `E-CLI-INVALID-ARGS` | A command-line argument is missing or invalid. |
| `E-IO-NOT-FOUND` | A path the user supplied does not exist. |
| `E-IO-PERMISSION` | A path exists but cannot be accessed. |
| `E-CONFIG-MISSING` | A model directory has no `config.json`. |
| `E-TOKENIZER-MISSING` | A HuggingFace checkpoint has no `tokenizer.json`. |
| `E-ADAPTER-UNSUPPORTED-ARCHITECTURE` | The model is not a supported family. |
| `E-ADAPTER-INVALID-SPEC` | An adapter DSL file is malformed or fails validation. |
| `E-ADAPTER-INSPECT-FAILED` | Auto-detection of a model directory failed. |
| `E-GENERATION-FAILED` | Model load or generation failed. |
| `E-INTERNAL-PANIC` | An unexpected internal panic was caught at the CLI boundary. |

### Exit codes

| Code | Category |
|------|----------|
| `0` | Success. |
| `1` | System / environment fault (permission denied, unreadable file). |
| `2` | User-input fault (bad argument, missing/invalid config or spec, missing path). |
| `3` | Runtime fault (model load or generation failure). |
| `101` | Internal panic caught at the CLI boundary. |

### Examples

**Model not found:**

```text
error[E-IO-NOT-FOUND]: the model directory was not found

What happened:
  Atenia could not find the model directory at the path you provided.

How to fix:
  Check the path for typos, or provide the correct location.

Technical details:
  path: models/does-not-exist
```

**Missing config:**

```text
error[E-CONFIG-MISSING]: config.json was not found

What happened:
  Atenia could not find config.json in the model directory, so it
  cannot tell which architecture this model uses.

How to fix:
  Make sure the directory is a complete HuggingFace checkpoint
  (config.json + weights + tokenizer), or point --model at a
  single .gguf file's directory instead.

Check it with:
  atenia inspect models/my-model
```

**Unsupported architecture** (classic Falcon, via `inspect`):

```text
error[E-ADAPTER-UNSUPPORTED-ARCHITECTURE]: model architecture is not supported

What happened:
  This model's architecture is not one Atenia can run. Atenia
  supports the Llama, Qwen, Gemma, Phi, Mistral (dense) and
  Falcon3 families.

How to fix:
  Use a model from a supported family. Classic Falcon,
  mixture-of-experts and multimodal models are out of scope.
```

**Invalid adapter YAML** (via `load`):

```text
error[E-ADAPTER-INVALID-SPEC]: adapter spec could not be parsed

What happened:
  The adapter spec file could not be parsed or did not pass
  validation.

How to fix:
  Fix the YAML/JSON spec. Run `atenia debug <file>` for a detailed
  report, or `atenia inspect <model_dir>` to generate a valid spec
  automatically.
```

---

## 10. Logging System

Logging is controlled entirely by the global flags ([§7](#7-cli-flags)).

### Levels

`error` < `warn` < `info` < `debug` < `trace`. The default is
`warn`. A message is shown when its level is at or below the
configured level.

- **default (`warn`)** — only warnings and errors.
- **`--verbose` (`info`)** — adds progress steps: command start,
  model loaded, generation finished.
- **`--debug`** — adds resolved configuration: model path,
  parameters, the env/hardware diagnostics block, matmul counters.
- **`--trace`** — adds fine-grained CLI-frontier detail.
- **`--quiet` (`error`)** — only human errors.

Higher log levels may slightly impact performance.

### Log file

`--log-file <path>` writes every emitted log line to a file in
addition to stderr. Parent directories are created if needed. If
the file cannot be created, the CLI fails with an `E-IO-*` error.

A log-file line carries a timestamp, level, trace id and target:

```text
1747929012 INFO [atenia-1747929012-a91f3c2d] atenia.cli: command start: generate
1747929012 INFO [atenia-1747929012-a91f3c2d] atenia.cli: model loaded in 6.3s
```

The stderr form is shorter — `[INFO] command start: generate` —
without the timestamp or trace id.

### Trace id

Every run has a trace id. If `--trace-id` is not given, one is
generated as `atenia-<unix_seconds>-<8 hex>`, e.g.
`atenia-1747929012-a91f3c2d`. The trace id appears in every
log-file line and in the `Trace:` footer of any error, so a user
reporting a failure has a stable id to quote.

---

## 11. JSON Output

Several commands can emit machine-readable JSON instead of the
human format:

- `generate --output json` — a single object with the prompt,
  generated text, token ids, timing, and counters.
- `doctor --json` — the host checklist as JSON.
- `diagnose --json` — the model checklist plus a `ready` boolean.
- `capabilities --json` — families, formats, quants, features.

(`load`, `debug` and `inspect` print human/YAML output only;
`inspect`'s YAML is itself machine-consumable.)

JSON goes to **stdout**; logs and errors still go to **stderr**, so
the JSON on stdout is always valid and parseable. This makes the
JSON modes safe for scripting:

```text
atenia diagnose --model models/my-model --json | jq .ready
atenia capabilities --json | jq .supported_families
```

If you also want a clean stdout regardless of log level, combine
with `--quiet` or redirect stderr.

---

## 12. Chat Behavior

### Prompt construction

`chat` keeps the conversation as an in-memory list of turns. On
each user message it builds the prompt for the model:

- **With a chat template.** If the model ships a chat template
  (most instruct models do) and `--no-chat-template` is not set,
  `chat` applies the model's own template to the **entire
  conversation history**. This is the correct multi-turn path —
  the model sees the full dialogue formatted the way it was
  trained.
- **Fallback.** If the model has no chat template, or
  `--no-chat-template` is set, `chat` builds a plain transcript:

  ```text
  User: <turn 1>
  Assistant: <turn 1 reply>
  User: <turn 2>
  Assistant:
  ```

  and lets the model complete the open `Assistant:` turn.

### History

`/history` prints the turns; `/reset` (or `/clear`) empties the
history. The history lives only in memory for the duration of the
session — it is not saved to disk.

Long conversations may become slower due to full prompt
reconstruction on each turn.

### Limitations

- **No persistent KV cache.** Each turn re-processes the whole
  conversation. For long chats this gets progressively slower; it
  is correct, just not optimised. Persistent KV caching is out of
  scope.
- **Greedy decoding only.** There is no sampling; `--temperature`
  is accepted but ignored (with a warning).
- **`--model` must be a directory.** An adapter spec file
  (`.yaml`) is rejected — see the FAQ.

---

## 13. Troubleshooting

### The model does not load

**Symptom:** `generate` or `chat` fails after the load starts, with
`error[E-GENERATION-FAILED]`.

**Why:** the model directory is incomplete or corrupt — a missing
shard, a truncated GGUF, a `config.json` that does not match the
weights.

**Fix:** run `atenia diagnose --model <dir>` first. If `diagnose`
is green but the load still fails, re-download the model files;
a partial download is the most common cause. Verify the file
sizes against the source.

### Empty output

**Symptom:** `generate` produces no text.

**Why:** the first generated token is an EOS token — usually a
wrong chat template or a wrong EOS set for the checkpoint.

**Fix:** try `atenia generate --no-chat-template` to isolate the
template. Confirm the model directory's `tokenizer.json` and
`config.json` match the weights.

### Tokenizer errors

**Symptom:** `error[E-TOKENIZER-MISSING]`, or garbled output.

**Why:** `tokenizer.json` is missing from a HuggingFace checkpoint,
or it is not the tokenizer that matches the weights.

**Fix:** re-download the full model files so `tokenizer.json` is
present and correct. `diagnose` reports tokenizer presence.

### GGUF incompatible

**Symptom:** the GGUF fails to load.

**Why:** the GGUF uses a quantisation Atenia cannot decode, or a
`general.architecture` outside the supported set. Atenia decodes
F32, F16, Q8_0, Q5_0, Q4_K, Q5_K and Q6_K. `general.architecture`
must be one of the supported families.

**Fix:** use a GGUF in a supported quantisation (Q4_K_M, Q5_K_M and
Q6_K are all decodable). `atenia capabilities` lists the supported
quants. A classic-Falcon GGUF (`general.architecture = "falcon"`)
is out of scope and will be rejected.

### CUDA not available

**Symptom:** `atenia doctor` shows `cuda: not available`.

**Why:** either this is a CPU-only build, or there is no NVIDIA
driver / GPU on the host.

**Fix:** this is not an error — the engine runs on CPU. If you
expect CUDA, confirm the build is CUDA-enabled (`doctor` shows the
build flavour) and that `nvidia-smi` works on the host.

### Lower performance than expected

**Why:** generation on CPU, or on a small VRAM budget with RAM/disk
tiering, is inherently slower than a fully-resident GPU run. A CPU
without AVX2 is much slower still.

**Fix:** check `atenia doctor` — it reports the CPU ISA and whether
CUDA is in use. Performance is not certified by the CLI; `doctor`
only tells you which path you are on. Performance optimisation is
out of scope for the CLI.

### Strange behaviour in chat

**Symptom:** the model rambles, repeats, or echoes template tokens.

**Why:** small instruct models (a few hundred million parameters)
are genuinely more repetitive under greedy decoding. With no
sampling, output is deterministic and can be verbose. Echoed
template tokens come from the model itself, not the CLI.

**Fix:** use a larger instruct model for better chat quality, or
shorten `--max-tokens`. This is a model-capacity property, not a
CLI defect.

---

## 14. Limitations

The CLI is honest about what the engine does not do.

- **No mixture-of-experts.** Mixtral and Mistral-MoE are not
  supported; there is no MoE code path.
- **No multimodal / vision.** Only dense causal language models.
- **No classic Falcon.** `FalconForCausalLM` / `RWForCausalLM` is a
  distinct architecture and is out of scope. Modern Falcon3 *is*
  supported (it is Llama-compatible).
- **No persistent KV cache.** `chat` re-processes the conversation
  each turn.
- **No real sampling.** Generation is greedy. `--temperature` is
  accepted but ignored.
- **Adapter YAML is not usable with `chat` or `generate`.** Those
  commands need a model directory with weights; an adapter spec
  describes a family, not a weight location. Use `load` to
  validate a spec.
- **Engine-internal logs are not fully controlled by the CLI.**
  Some low-level lines (`[APX] ...` printed before `main()` runs,
  `[ATENIA] ...` tier-planner output from the loader) are emitted
  directly by the engine on **stderr** and are not gated by the
  CLI log level. They never reach stdout, so the output contract
  holds, but they appear even at the default level. Gating them
  would require changes to the runtime core, which is out of scope
  for the CLI layer.

---

## 15. Best Practices

- **Run `diagnose` before `generate`.** It catches an unsupported
  or incomplete model in seconds, without a slow load.
- **Use `doctor` when something is environmental.** CPU ISA, RAM,
  CUDA availability — `doctor` answers those before you blame a
  model.
- **Use `chat` for exploration, `generate` for one-shot tasks and
  scripting.**
- **Use JSON output for scripting.** `--output json` / `--json`
  with `jq` gives stable, parseable results; stderr stays separate.
- **Use `--verbose` or `--debug` when investigating.** Default
  output is intentionally quiet; raise the level to see load
  steps, resolved configuration and counters.
- **Use `--log-file` and quote the trace id** when reporting a
  problem — it ties the error to the full log.
- **Bootstrap new models with `inspect`.** Its emitted YAML is
  correct by construction for HuggingFace models.

---

## 16. FAQ

**Why doesn't `load` generate text?**
`load` is a spec-construction and validation command. It parses an
adapter spec, validates it and builds the v2 adapter — nothing
more. Generation has its own command (`generate`) with its own
arguments. Keeping them separate makes `load` fast and
side-effect-free.

**Why doesn't `chat` accept an adapter YAML for `--model`?**
An adapter spec describes a model *family*; it contains no weights
and no path to weights. `chat` needs a real checkpoint to load, so
`--model` must be a model directory. To validate a spec file, use
`atenia load <file>`.

**Why is there no sampling / temperature?**
The generation path is greedy by design at this stage. Real
sampling needs engine-side support that does not exist yet;
`--temperature` is accepted only so the flag is stable for the
future. A non-zero value is ignored with a warning.

**Why does `diagnose` say OK but `generate` still fails?**
`diagnose` checks *structure and resolution* — the directory
layout, the family, the tokenizer presence, the adapter resolution.
It does **not** load weights or run a forward pass. A model can
have a valid structure but corrupt or truncated weight files;
that surfaces only when `generate` actually loads them.

**Why do I see `[APX]` and `[ATENIA]` lines I did not ask for?**
Those are engine-internal log lines (runtime initialisation, the
tier planner). They are printed by the runtime core directly on
**stderr**, before or independently of the CLI logging layer, so
the CLI log level does not gate them. They never reach stdout, so
piped output stays clean. See [Limitations](#14-limitations).

---

*This manual describes the `atenia` CLI as implemented. For the
declarative adapter layer used by `load` / `inspect` / `debug`, see
`docs/ADAPTER_TOOLKIT_V2.md`. For the validated model families, see
`docs/MODEL_FAMILY_VALIDATION.md`. For project status and milestone
history, see `docs/STATUS.md` and `docs/MILESTONES.md`.*
