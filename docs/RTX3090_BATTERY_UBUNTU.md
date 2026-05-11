# RTX 3090 battery on Linux / WSL

This is the Linux/WSL companion to the PowerShell battery. Its default mode
wraps the smoke tests that already exist in the repo and records data useful
for comparing a 24 GiB RTX 3090 server against the dev laptop.

The script is:

- non-destructive
- interactive only when model autodetection is ambiguous
- able to auto-install missing userspace tools
- tolerant of failed model loads
- strict about keeping stdout JSON parseable
- able to run optional CLI generation probes when requested

## Prerequisites / auto-install

By default the script tries to install missing userspace tools:

- `build-essential`
- `pkg-config`
- `jq`
- Rust via `rustup` when `cargo` is missing
- Ubuntu CUDA toolkit packages when `nvcc`/CUDA headers are missing

It uses `sudo apt-get` for system packages and rustup for Rust.

It does **not** install NVIDIA drivers automatically. `nvidia-smi` must already
work. On WSL this usually means updating the Windows NVIDIA driver and
restarting WSL. The script can install the Ubuntu-side CUDA toolkit, but the
host driver remains external.

To disable auto-install:

```bash
AUTO_INSTALL=false bash scripts/rtx3090_battery.sh
```

To disable only CUDA toolkit installation:

```bash
bash scripts/rtx3090_battery.sh --no-install-cuda
```

To force a CUDA toolkit install attempt:

```bash
bash scripts/rtx3090_battery.sh --install-cuda
```

To reset the Ubuntu userspace dependencies installed by the helper:

```bash
bash scripts/rtx3090_battery.sh --clean-deps
```

That removes `~/.cargo`, `~/.rustup`, purges `jq`, `pkg-config`, and
`build-essential`, then runs apt autoremove/clean. It also removes Atenia's
generated work artifacts.

If you also want to purge Ubuntu CUDA toolkit packages for a from-scratch CUDA
toolkit check:

```bash
bash scripts/rtx3090_battery.sh --clean-deps --clean-cuda
```

This does not remove the Windows/host NVIDIA driver used by WSL `nvidia-smi`.

Useful modes restored from the original Linux helper:

```bash
bash scripts/rtx3090_battery.sh --auto       # auto-detect models root
bash scripts/rtx3090_battery.sh --list-only  # print cases only
bash scripts/rtx3090_battery.sh --clean      # remove generated work/log/cache dirs
bash scripts/rtx3090_battery.sh --clean-deps # remove helper-installed deps
```

The environment-variable form still works:

```bash
AUTO_MODE=true LIST_ONLY=true bash scripts/rtx3090_battery.sh
```

Manual checks:

```bash
cargo --version
jq --version
nvidia-smi
```

If `cargo` is installed through rustup but WSL does not see it, load Cargo's
environment first:

```bash
source ~/.cargo/env
```

The script also tries this automatically when `~/.cargo/env` exists.

## Recommended layout

For WSL, prefer Linux paths that point at the internal disk:

```bash
/mnt/d/models
/mnt/d/Atenia
```

Run from the repo root:

```bash
cd /mnt/f/Proyectos/artenia_engine/atenia-engine
bash scripts/rtx3090_battery.sh
```

If the repo is copied to the server's Linux filesystem, run from that repo
instead. The important part is that `MODELS_ROOT` points at the model folder.

When `/mnt/d` exists, the script defaults to `WORK_ROOT=/mnt/d/Atenia` so build
artifacts, runtime cache, and logs stay on the internal D: disk rather than the
USB F: disk. Otherwise it falls back to `$HOME/Atenia`.

When run interactively, the script asks for:

- models directory when autodetection is ambiguous
- repo smokes, CLI generation probes, or both
- normal/quick vs full suite
- output token count and prompt when CLI generation probes are enabled
- work/output directory
- whether to skip the build

Default mode is `smokes`: it runs the smoke tests already present in `tests/`
instead of re-encoding model coverage in this script. Use `--mode cli` only for
extra generation metrics, or `--mode both` when you want both artifacts.

The wrapper does not force `ATENIA_M8_BF16_KERNEL=1` by default. That is
intentional: the repo smoke tests set the precision mode they need, and CLI
probes should match a normal manual CLI run unless explicitly requested.
Use `--cli-force-bf16` only when you want to investigate the BF16 path itself.

## Dry listing

Use this first. It does not build or run models.

```bash
MODELS_ROOT=/mnt/d/models \
WORK_ROOT=/mnt/d/Atenia \
LIST_ONLY=true \
bash scripts/rtx3090_battery.sh
```

It prints every case and whether the model directory exists.
In default `smokes` mode it prints the repo smoke commands. In `--mode cli` it
prints the optional CLI generation probes and model-directory existence.

## Quick suite

Recommended first pass:

```bash
MODELS_ROOT=/mnt/d/models \
WORK_ROOT=/mnt/d/Atenia \
SUITE=quick \
MAX_TOKENS=32 \
bash scripts/rtx3090_battery.sh
```

Quick repo smokes:

- `cargo test --lib`
- `cargo test --lib gguf_`
- TinyLlama Q8_0 / Q4_K_M GGUF lib smokes
- TinyLlama, SmolLM2, Qwen 2.5 1.5B, and Llama 3.2 1B safetensors end-to-end smokes

The script maps the standard safetensors variables from `MODELS_ROOT`:

- `TINYLLAMA_SAFETENSORS_PATH`
- `SMOLLM2_SAFETENSORS_PATH`
- `QWEN25_SAFETENSORS_PATH`
- `LLAMA32_SAFETENSORS_PATH`

## Full suite

Use only if the machine can be left running:

```bash
MODELS_ROOT=/mnt/d/models \
WORK_ROOT=/mnt/d/Atenia \
SUITE=full \
MAX_TOKENS=32 \
bash scripts/rtx3090_battery.sh
```

Adds the heavier ignored diagnostics already present in the repo, including
TinyLlama BF16/GPU/disk-spill smokes, full-family safetensors validations, M8.5
family validation, and the M11.D GGUF ignored diagnostics.

## Skip build

If the binary is already built:

```bash
MODELS_ROOT=/mnt/d/models \
WORK_ROOT=/mnt/d/Atenia \
SKIP_BUILD=true \
EXE_PATH=/mnt/d/Atenia/cargo-target-rtx3090/release/atenia \
bash scripts/rtx3090_battery.sh
```

## Outputs

Each run creates:

```text
/mnt/d/Atenia/bench_logs/rtx3090_YYYYMMDD_HHMMSS/
  run_metadata.json
  summary.md
  summary.csv
  summary.jsonl
  smokes.csv
  smokes.jsonl
  diagnostics/
    nvidia-smi.txt
    nvidia-smi-query.txt
    df-hT.txt
    lsblk.txt
    lscpu.txt
    tool-versions.txt
    git.txt
    env-selected.txt
  raw/
    <case>.stdout.json
    <case>.stderr.log
    <case>.env.json
```

Bring back the entire run directory.

## What to look at first

Open `summary.md` first. Then use `summary.csv` for spreadsheet comparison.
`run_metadata.json` contains the server fingerprint, repo commit, toolchain
versions, storage snapshot, NVIDIA snapshot, and model inventory. The
`diagnostics/` folder keeps the raw command output in case we need to inspect
driver, disk, WSL, or toolchain details later.

The raw stderr logs contain the residency plan:

```text
VRAM: N tensors (X GiB)
RAM:  N tensors (Y GiB)
Disk: N tensors (Z GiB)
```

On a healthy 3090 run we expect many more tensors to stay in VRAM than on the
8 GiB laptop. If the 3090 still spills heavily to RAM/Disk on 1B-3B models,
that is a tier-planner or safety-threshold signal worth investigating.

## Notes

- Build artifacts go to `$WORK_ROOT/cargo-target-rtx3090`.
- Runtime disk-tier cache goes to `$WORK_ROOT/runtime-cache/<case>`.
- Raw benchmark logs go to `$WORK_ROOT/bench_logs/<run_id>`.
- The script does not force `ATENIA_M8_BF16_KERNEL=1` unless `--cli-force-bf16`
  is passed.
- Safetensors fast cases set `ATENIA_FAST_MODE=1`.
- GGUF cases use manifest/quantized mode and do not force fast mode.

## Common failures

### JSON parse failed

Usually means something wrote to stdout before/around the JSON payload. Check
`raw/<case>.stdout.json` and `raw/<case>.stderr.log`.

### BF16-to-VRAM GGUF upload failed

This is a real runtime/load failure for that case. The script records it in
`summary.*` and continues. The stderr log is the artifact to bring back.

### Wrong RAM numbers in summary

The current script anchors `VRAM:`, `RAM:`, and `Disk:` lines separately. If
this regresses, check `parse_plan` in `scripts/rtx3090_battery.sh`.
