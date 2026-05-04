# Models layout (operator-local, git-ignored)

**Why this file exists**: the `models/` directory is in
`.gitignore` (model weights are too large for the repo and
distribution-licensed independently of Atenia). That means every
new tool, test or agent that needs the weights has to ask "where
are they?" — and answers diverge over time. This file pins the
canonical layout so that question never needs to be asked again.

## Canonical root

```
F:\Proyectos\artenia_engine\atenia-engine\models\
```

Forward-slash form for cross-tool quoting:

```
F:/Proyectos/artenia_engine/atenia-engine/models/
```

Project root + `models/`. Never under `D:\Atenia\` (which is the
**runtime cache** root used by `ATENIA_DISK_TIER_DIR` for spilled
tensor files; see `HANDOFF_APX_V20_M7.md`). The two are distinct:

| Path                       | Purpose                                    |
|----------------------------|--------------------------------------------|
| `F:\...\atenia-engine\models\` | source weights (`.safetensors`)        |
| `D:\Atenia\disk_tier_*\`   | M7 runtime spill cache                     |
| `D:\atenia-m7-cache\`      | M7.3 13B smoke working dir                 |
| `D:\atenia-m8-pipeline\`   | M8.0b NVMe pipeline bench scratch          |
| `D:\atenia-m8-cache\`      | M8.5 13B BF16 smoke working dir            |

Operators may keep the model checkpoints on a different volume
(M5 era ran from an external USB SSD on `F:`); the path under
`models/` itself is the contract — not the absolute prefix.

## Directory layout

```
models/
├── gpt2-safetensors/                   M4 baseline checkpoint (HF gpt2)
│   └── model.safetensors
├── tinyllama-1.1b/                     M4.5 / M4.6 / M4.7 / M8.5
│   └── model.safetensors               (2.20 GB BF16)
├── smollm2-1.7b-instruct/              M4.6 phase A
│   └── model.safetensors               (3.42 GB BF16)
├── qwen2.5-1.5b-instruct/              M4.6 phase B
│   └── model.safetensors               (3.09 GB BF16)
├── llama-3.2-1b-instruct/              M4.6 phase C
│   └── model.safetensors               (2.47 GB BF16)
├── mistral-7b-v0.3/                    M4.7.1 sharded loader validation
│   ├── model-00001-of-00003.safetensors
│   ├── model-00002-of-00003.safetensors
│   ├── model-00003-of-00003.safetensors
│   └── consolidated.safetensors        (legacy single-file mirror)
└── llama-2-13b-chat/                   M5 / M6 / M7 / M8 13B
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```

Each directory also contains the HF `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `generation_config.json` files where
applicable — the loader pipeline (`src/nn/llama/pipeline.rs`)
requires them alongside the safetensors.

## Environment variables

The four 1-2 B models are reachable via test-time env vars
consumed by `tests/bf16_storage_full_family_validation_test.rs`,
`tests/m4_7_3_full_family_validation_test.rs`, and
`tests/m8_5_full_family_validation_test.rs`:

| Env var                       | Path under `models/`                                      |
|-------------------------------|-----------------------------------------------------------|
| `TINYLLAMA_SAFETENSORS_PATH`  | `tinyllama-1.1b/model.safetensors`                        |
| `SMOLLM2_SAFETENSORS_PATH`    | `smollm2-1.7b-instruct/model.safetensors`                 |
| `QWEN25_SAFETENSORS_PATH`     | `qwen2.5-1.5b-instruct/model.safetensors`                 |
| `LLAMA32_SAFETENSORS_PATH`    | `llama-3.2-1b-instruct/model.safetensors`                 |

## PowerShell one-liner — set all four

```powershell
$models = "F:\Proyectos\artenia_engine\atenia-engine\models"
$env:TINYLLAMA_SAFETENSORS_PATH = "$models\tinyllama-1.1b\model.safetensors"
$env:SMOLLM2_SAFETENSORS_PATH   = "$models\smollm2-1.7b-instruct\model.safetensors"
$env:QWEN25_SAFETENSORS_PATH    = "$models\qwen2.5-1.5b-instruct\model.safetensors"
$env:LLAMA32_SAFETENSORS_PATH   = "$models\llama-3.2-1b-instruct\model.safetensors"
```

To clean up afterwards:

```powershell
Remove-Item Env:TINYLLAMA_SAFETENSORS_PATH
Remove-Item Env:SMOLLM2_SAFETENSORS_PATH
Remove-Item Env:QWEN25_SAFETENSORS_PATH
Remove-Item Env:LLAMA32_SAFETENSORS_PATH
```

## CLI smoke targets (atenia generate)

The tier-aware loader takes a `--model` argument that must point
at the **directory**, not the file:

| Model            | `--model` argument                                                            |
|------------------|-------------------------------------------------------------------------------|
| Llama 2 7B Chat  | `F:/Proyectos/artenia_engine/atenia-engine/models/llama-2-7b-chat`            |
| Llama 2 13B Chat | `F:/Proyectos/artenia_engine/atenia-engine/models/llama-2-13b-chat`           |
| Mistral 7B v0.3  | `F:/Proyectos/artenia_engine/atenia-engine/models/mistral-7b-v0.3`            |

Forward slashes are mandatory on PowerShell — backslashes get
mangled when Cargo reparses argv (M6 close note).

## Provenance

These paths are the operator's local machine layout as of M8.5
(May 2026). Future moves (e.g. consolidating to a single SSD
or reorganising under a single-checkpoint root) should update
this document in the same commit that touches the directories
so the contract stays single-sourced.
