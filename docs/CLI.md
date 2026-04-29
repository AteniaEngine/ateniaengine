# Atenia CLI

The `atenia` binary is the public reproduction surface for the
v20 killer demo. Three subcommands:

- **`atenia probe`** — cross-vendor hardware enumeration
  (CPU + GPU + RAM). Behind the `hw-probe` Cargo feature.
- **`atenia run`** — the tri-mode killer-demo runner
  (Modes A / B / C). Default-build feature.
- **`atenia explain`** — pre-existing v13 self-trainer
  narrative explainer. Legacy surface, preserved.

Build:

```bash
# Default build — `run` and `explain` work, `probe` reports
# "feature not enabled".
cargo install --path .

# Full build — `probe` operational.
cargo install --path . --features hw-probe
```

After installation, `atenia` is on the PATH; run from any
directory.

---

## Exit codes

| Code | Meaning                                                          |
|------|------------------------------------------------------------------|
| 0    | Success                                                          |
| 1    | Runtime error (load OOM, kernel failure, deep_degrade failure)   |
| 2    | Argument / configuration error (bad path, missing feature, …)   |
| 3    | Mathematical contract violation (Mode C argmax pre != post)      |

CI scripts can `case $?` on these to distinguish "demo
crashed" from "demo proved a contract violation".

---

## `atenia probe`

Cross-vendor GPU enumeration via `wgpu`, with NVIDIA
augmentation via NVML when present. Same logic as the legacy
`hardware_probe` binary (preserved via deprecation through
M4.9.e; dropped in M4.9.f).

```bash
atenia probe                       # text output
atenia probe --output json         # machine-readable JSON
```

Without the `hw-probe` feature the command exits 2 with a
clear message pointing at the rebuild flag — default builds
do not pull the wgpu dependency tree.

The JSON schema is documented in
`docs/HARDWARE_PROBE.md`. Stable across M4.9.+; new fields
are non-breaking, removed fields are a major bump.

---

## `atenia run`

The killer-demo runner. Three modes; same flag surface for
all three.

```bash
atenia run --model <PATH> --mode {a|b|c}
           [--seq N]
           [--output text|json]
           [--cache-dir PATH]
           [--no-progress]
```

### Required flags

- `--model <PATH>` — path to the model directory. Must
  contain `config.json`, the safetensors files (single-file
  or sharded with `model.safetensors.index.json`), and
  tokenizer files. Falls back to the
  `ATENIA_LLAMA2_13B_DIR` environment variable when omitted.

- `--mode {a|b|c}` — execution mode (see below).

### Optional flags

- `--seq N` — sequence length. Mode A defaults to 4 (the
  M4.7.6.d canonical input). Modes B and C default to 1
  (the M4.7.6.e wall-clock budget). The contracts hold at
  any `seq > 0`.

- `--output text|json` — output format. Default: `text`.
  JSON schema documented in **JSON schema** below.

- `--cache-dir PATH` — disk-tier directory for spill cache
  (Modes B and C). Defaults to `ATENIA_DISK_TIER_DIR` env
  var, then to a platform-specific scratch directory.
  **Should be on an NVMe drive**: Mode C writes ~13 GB of
  BF16 spill; the CLI runs a quick disk-throughput probe
  at startup and warns when the chosen path is below the
  200 MB/s practical floor.

- `--no-progress` — suppress the live heartbeat dots during
  long phases (load, forward). Phase durations and the final
  report still print.

### Hardware soft-warning

The CLI checks total system RAM via `sysinfo`. If less than
28 GB is detected, a warning prints to stderr and the run
proceeds — the operator may have memory hot-add, swap, or a
checkpoint smaller than 13 B parameters that does not need
the headroom.

### Modes

#### Mode A — clean RAM, no spill

Baseline run. No reactive context attached; M3-e reaction
loop does not fire. Produces argmax + logit on the canonical
input.

Token pattern: `seq=1 -> [BOS]`, `seq=4 -> [BOS, 100, 200, 300]`.

Reproduction target on the M4.7.6.d / M4.8.f baseline:

```
$ atenia run --model D:/Atenia/models/llama-2-13b-chat \
             --mode a --seq 4

  Build graph ....................   1.91s
  Load weights .................   167.84s   (~155 MB/s)
  Forward ......................   278.20s
  Pos 0: argmax id =     1   logit = 4.7747
  Total wall-clock: 7.5 minutes.
```

Exit code: 0 on completion (no contract to violate beyond
runtime errors).

#### Mode B — autonomous LRU spill trigger validation

Attaches a reactive context with high-pressure RAM/VRAM
probes. The M4.6 guard fires `Degrade` on its first
checkpoint; `dual_memory_pressure` promotes to `DeepDegrade`;
`deep_degrade_with_lru` fires autonomously. The forward is
wrapped in `catch_unwind` because a downstream activation
node hits the documented M4.7.5.e `ensure_cpu` gap (real but
structurally separate from the demo's transparency
contract — that is owned by Mode C).

Reproduction target:

```
$ atenia run --model D:/Atenia/models/llama-2-13b-chat \
             --mode b --cache-dir D:/Atenia/cache

  DeepDegrade events:          4
  Spilled to disk:             26031.7 MB
  [PASS] ✓ autonomous Degrade -> DeepDegrade promotion fired
          and the spill primitive wrote to disk
  Forward absorbed:
    catch_unwind absorbed a downstream panic — the documented
    M4.7.5.e activation-arm `ensure_cpu` gap (M5+ follow-up).
  Total wall-clock: 8.1 minutes.
```

Exit code: 0 if `deep_degrade_events_count > 0` AND
`spilled_bytes > 0`. The `catch_unwind` absorbing a panic is
**not** a failure — it is the documented activation-arm
gap. The verbatim panic message prints under "Forward
absorbed:" so CI scripts can grep for it.

#### Mode C — forced 50 % LRU spill (the *momento guau*)

The canonical transparency-contract path:

1. Warmup forward at low pressure → captures `argmax(pre)`.
2. Force `Graph::deep_degrade_with_lru(&cache_dir)`. The
   M4.7.5.d primitive spills the bottom 50 % of the
   touch-order LRU.
3. Post-spill forward → lazy-restore via the M4.7.4.d /
   M4.7.5.e `ensure_cpu` Disk-arm; captures `argmax(post)`.
4. Compare `argmax(pre)` vs `argmax(post)` bit-exactly.

Reproduction target:

```
$ atenia run --model D:/Atenia/models/llama-2-13b-chat \
             --mode c --seq 1 --cache-dir D:/Atenia/cache

  Build graph ....................   1.28s
  Load weights .................   166.78s   (~156 MB/s)
  Warmup forward ...............   470.98s
  Force LRU spill ................  72.52s   (866 tensors migrated)
  Forward ......................    31.15s
  Per-position argmax:
    Pos 0: argmax id =     1   logit = 4.7747
  Transparency contract:
    argmax(pre)  = 1, logit 4.7747
    argmax(post) = 1, logit 4.7747
    [PASS] ✓ argmax(pre-spill) == argmax(post-spill) bit-exactly
  Total wall-clock: 12.4 minutes.
```

Exit code:
- 0 if `argmax(pre) == argmax(post)` bit-exactly.
- 3 if either differs (contract violation).

This is the canonical *momento guau* — the same argmax
(`id = 1, logit = 4.7747`) appears in:
- The M4.7.6.e Mode C test (the original empirical baseline)
- The M4.7.6.d Mode A baseline (pos 0 of seq=4)
- The M4.8.f Mode A re-run after the perf upgrade
- This Mode C run via the public CLI

The transparency contract holds across the entire M4.7 /
M4.8 / M4.9 stack at 13 B parameter scale.

---

## Environment variables

The CLI honours the same env vars as the test harness:

| Variable                      | Used by             | Purpose                                                         |
|-------------------------------|---------------------|-----------------------------------------------------------------|
| `ATENIA_LLAMA2_13B_DIR`       | `--model` fallback  | Default model directory when `--model` is omitted.              |
| `ATENIA_DISK_TIER_DIR`        | `--cache-dir` fb.   | Default spill cache directory for Modes B / C.                  |
| `ATENIA_APX_MODE`             | engine              | Override the APX dispatch mode. Default: `"7.2"` (post-M4.8.b). |

No new env vars introduced by M4.9; all overrides go via CLI
flags.

---

## JSON schema

Every `--output json` invocation produces a stable schema.
Adding fields is non-breaking; removing fields is a major
bump.

### Mode A / Mode C — `DemoReport`

```jsonc
{
  "version": "atenia-engine 0.1.0",
  "mode": "a"|"c",
  "seq": <usize>,
  "model": {
    "path": "<PathBuf>",
    "layers": <usize>,
    "hidden_size": <usize>,
    "intermediate_size": <usize>,
    "vocab_size": <usize>,
    "param_count": <usize>,
    "storage": "bf16"
  },
  "phases": {
    "build_seconds": <f32>,
    "load_seconds": <f32>,
    "load_throughput_mb_s_estimate": <f32>,
    "forward_seconds": <f32>,
    "spill_seconds": <f32 | null>,            // Mode C only
    "spill_tensors_migrated": <usize | null>, // Mode C only
    "warmup_forward_seconds": <f32 | null>    // Mode C only
  },
  "argmax": [
    { "position": <usize>, "token_id": <usize>, "logit": <f32> }
  ],
  "logit_stats": {
    "max_abs": <f32>,
    "mean_abs": <f32>,
    "finite": <usize>,
    "total": <usize>
  },
  "contract": null | {                        // Mode C only
    "name": "transparency",
    "pre":  { "position": 0, "token_id": <usize>, "logit": <f32> },
    "post": { "position": 0, "token_id": <usize>, "logit": <f32> },
    "bit_exact": <bool>,
    "description": "<human-readable>"
  },
  "total_seconds": <f64>
}
```

### Mode B — `ModeBReport`

```jsonc
{
  "version": "atenia-engine 0.1.0",
  "mode": "b",
  "seq": <usize>,
  "model": { /* same shape as DemoReport.model */ },
  "phases": {
    "build_seconds": <f32>,
    "load_seconds": <f32>,
    "load_throughput_mb_s_estimate": <f32>,
    "forward_seconds": <f32>
  },
  "deep_degrade_events": <u64>,
  "spilled_bytes": <u64>,
  "forward_completed": <bool>,
  "panic_message": <string | null>,
  "total_seconds": <f64>
}
```

`forward_completed` is `false` whenever `panic_message` is
non-null. Both convey the same information; `forward_completed`
is the boolean check, `panic_message` carries the diagnostic.

---

## Reproducing the momento guau

The shortest path from `git clone` to a green transparency
contract on the dev-box hardware:

```bash
# Hardware: 8 GB VRAM + 32 GB RAM + NVMe.
# Software: Rust stable, CUDA Toolkit, MSVC BuildTools (Windows)
#           or gcc (Linux).

git clone https://github.com/AteniaEngine/ateniaengine.git
cd ateniaengine

# Download Llama 2 13B Chat (requires HuggingFace auth).
huggingface-cli download meta-llama/Llama-2-13b-chat-hf \
    --local-dir ./models/llama-2-13b-chat \
    --include '*.safetensors' '*.json' 'tokenizer*'

cargo install --path .

atenia run --mode c \
           --model ./models/llama-2-13b-chat \
           --cache-dir ./atenia-cache
```

Expected output: a `[PASS] ✓` line confirming
`argmax(pre-spill) == argmax(post-spill) bit-exactly`. Total
wall-clock ~12 minutes on RTX 4070 Laptop / 32 GB RAM.

If the cache directory's throughput is below 200 MB/s the CLI
warns up front; an NVMe path is the recommended runtime
prerequisite.
