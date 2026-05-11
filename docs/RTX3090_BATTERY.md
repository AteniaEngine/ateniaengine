# RTX 3090 benchmark battery

Portable smoke / benchmark harness for running Atenia on a larger NVIDIA box
(for example an RTX 3090 with 24 GiB VRAM).

## Goal

Collect comparable evidence, not just "it ran":

- model load / generation success
- valid JSON output from `atenia generate --output json`
- tokens generated, total seconds, tokens/second
- manifest mode selected by the loader
- tier plan from stderr: VRAM / RAM / Disk tensors and GiB
- raw stdout/stderr logs per case
- GPU / driver / RAM snapshot

## Expected layout

Copy or mount the model directory on an internal disk:

```powershell
D:\models
D:\Atenia
```

Run from the Atenia repo root.

## Dry listing

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rtx3090_battery.ps1 `
  -ModelsRoot D:\models `
  -WorkRoot D:\Atenia `
  -Suite quick `
  -ListOnly
```

This prints all cases and whether their model directories exist.

## Quick suite

Recommended first pass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rtx3090_battery.ps1 `
  -ModelsRoot D:\models `
  -WorkRoot D:\Atenia `
  -Suite quick `
  -MaxTokens 32
```

Quick suite cases:

- GGUF TinyLlama Q4_K_M
- GGUF TinyLlama Q8_0
- GGUF Llama 3.2 1B Q4_K_M
- GGUF SmolLM2 Q4_K_M
- GGUF Phi 3.5 Mini Q4_K_M
- Safetensors TinyLlama / SmolLM2 / Qwen 2.5 1.5B / Llama 3.2 1B
- Safetensors cases run both manifest/default and `ATENIA_FAST_MODE=1`

## Full suite

Use when the server can be left running:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rtx3090_battery.ps1 `
  -ModelsRoot D:\models `
  -WorkRoot D:\Atenia `
  -Suite full `
  -MaxTokens 32
```

Adds larger/top-10 safetensors cases such as Phi, Mistral, Gemma, Falcon, and
Llama 2 13B where present.

## Outputs

Each run creates:

```text
D:\Atenia\bench_logs\rtx3090_YYYYMMDD_HHMMSS\
  run_metadata.json
  summary.md
  summary.csv
  summary.jsonl
  raw\
    <case>.stdout.json
    <case>.stderr.log
    <case>.env.json
```

Bring back the whole run folder. `summary.md` is human-readable; `summary.csv`
is for spreadsheet comparison; `summary.jsonl` is easiest to diff or ingest.

## Notes

- Build artifacts go to `D:\Atenia\cargo-target-rtx3090`.
- Runtime disk-tier cache goes to `D:\Atenia\runtime-cache\<case>`.
- The script sets `ATENIA_M8_BF16_KERNEL=1` for all cases.
- Safetensors `fast` cases additionally set `ATENIA_FAST_MODE=1`.
- GGUF cases use manifest/quantized mode and do not force fast mode.
