# RTX 3090 Manual Test Runbook

Guia manual para ejecutar Atenia en un servidor Ubuntu/WSL con NVIDIA, sin usar
el script de bateria. La idea es llevar el proyecto en disco externo, preparar
el entorno, correr los smoke tests reales del repo y traer de vuelta una carpeta
de evidencias.

## 1. Ubicar Proyecto Y Modelos

Ejemplo si el disco externo queda montado en `/mnt/f`:

```bash
cd /mnt/f/Proyectos/artenia_engine/atenia-engine
pwd
```

Elegir un disco interno/NVMe para compilacion, cache y logs. Evitar el USB para
`target/` y runtime cache:

```bash
export ATENIA_WORK_ROOT=/mnt/d/Atenia
export CARGO_TARGET_DIR="$ATENIA_WORK_ROOT/cargo-target-rtx3090"
export ATENIA_TEST_DISK_TIER_BASE="$ATENIA_WORK_ROOT/test-cache"
export ATENIA_DISK_TIER_DIR="$ATENIA_WORK_ROOT/runtime-cache/manual"
export BENCH_LOGS="$ATENIA_WORK_ROOT/bench_logs/manual_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CARGO_TARGET_DIR" "$ATENIA_TEST_DISK_TIER_BASE" "$ATENIA_DISK_TIER_DIR" "$BENCH_LOGS"
```

Apuntar a la carpeta de modelos:

```bash
export ATENIA_MODELS_ROOT=/mnt/d/models
```

Si los modelos estan en otro lugar, cambiar solo ese path.

## 2. Instalar Dependencias Ubuntu

Actualizar paquetes base:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config jq curl ca-certificates git
```

Instalar Rust/Cargo si falta:

```bash
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
cargo --version
rustc --version
```

Verificar NVIDIA. Esto debe funcionar antes de correr tests:

```bash
nvidia-smi
```

Instalar CUDA toolkit de Ubuntu si falta `nvcc`:

```bash
if ! command -v nvcc >/dev/null 2>&1; then
  sudo apt-get install -y nvidia-cuda-toolkit
fi
nvcc --version || true
```

Nota: en WSL, `nvidia-smi` depende del driver NVIDIA del host Windows. El
toolkit Ubuntu da `nvcc`, headers y librerias, pero no reemplaza el driver host.

## 3. Exportar Rutas De Modelos Safetensors

Los tests existentes esperan estas variables:

```bash
export TINYLLAMA_SAFETENSORS_PATH="$ATENIA_MODELS_ROOT/tinyllama-1.1b/model.safetensors"
export SMOLLM2_SAFETENSORS_PATH="$ATENIA_MODELS_ROOT/smollm2-1.7b-instruct/model.safetensors"
export QWEN25_SAFETENSORS_PATH="$ATENIA_MODELS_ROOT/qwen2.5-1.5b-instruct/model.safetensors"
export LLAMA32_SAFETENSORS_PATH="$ATENIA_MODELS_ROOT/llama-3.2-1b-instruct/model.safetensors"
```

Comprobar que existen:

```bash
ls -lh "$TINYLLAMA_SAFETENSORS_PATH"
ls -lh "$SMOLLM2_SAFETENSORS_PATH"
ls -lh "$QWEN25_SAFETENSORS_PATH"
ls -lh "$LLAMA32_SAFETENSORS_PATH"
```

## 4. Capturar Diagnostico Del Servidor

Guardar contexto de hardware/software antes de correr:

```bash
{
  date -Iseconds
  uname -a
  lscpu
  free -h
  df -hT
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,ROTA
  cargo --version
  rustc --version
  jq --version
  nvcc --version || true
  nvidia-smi
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git status --short
} > "$BENCH_LOGS/server_diagnostics.txt" 2>&1

nvidia-smi -q > "$BENCH_LOGS/nvidia-smi-query.txt" 2>&1
```

## 5. Build

Compilar Atenia usando el disco interno/NVMe:

```bash
cargo build --release --bin atenia 2> "$BENCH_LOGS/build.stderr.log" | tee "$BENCH_LOGS/build.stdout.log"
```

## 6. Smokes Rapidos Del Repo

Estos son los smokes base que conviene correr primero.

```bash
cargo test --lib 2> "$BENCH_LOGS/lib_all.stderr.log" | tee "$BENCH_LOGS/lib_all.stdout.log"
```

```bash
cargo test --lib gguf_ 2> "$BENCH_LOGS/lib_gguf.stderr.log" | tee "$BENCH_LOGS/lib_gguf.stdout.log"
```

```bash
cargo test --lib tinyllama_q8_0_gguf -- --nocapture \
  2> "$BENCH_LOGS/tinyllama_q8_0_gguf.stderr.log" \
  | tee "$BENCH_LOGS/tinyllama_q8_0_gguf.stdout.log"
```

```bash
cargo test --lib tinyllama_q4_k_m_gguf -- --nocapture \
  2> "$BENCH_LOGS/tinyllama_q4_k_m_gguf.stderr.log" \
  | tee "$BENCH_LOGS/tinyllama_q4_k_m_gguf.stdout.log"
```

## 7. Smokes Safetensors End-To-End

```bash
cargo test --test tinyllama_end_to_end_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/tinyllama_end_to_end.stderr.log" \
  | tee "$BENCH_LOGS/tinyllama_end_to_end.stdout.log"
```

```bash
cargo test --test smollm2_end_to_end_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/smollm2_end_to_end.stderr.log" \
  | tee "$BENCH_LOGS/smollm2_end_to_end.stdout.log"
```

```bash
cargo test --test qwen25_end_to_end_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/qwen25_end_to_end.stderr.log" \
  | tee "$BENCH_LOGS/qwen25_end_to_end.stdout.log"
```

```bash
cargo test --test llama_3_2_end_to_end_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/llama32_end_to_end.stderr.log" \
  | tee "$BENCH_LOGS/llama32_end_to_end.stdout.log"
```

## 8. Diagnosticos Pesados Opcionales

Usar si hay tiempo y la maquina puede quedar trabajando.

```bash
cargo test --test tinyllama_bf16_storage_smoke_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/tinyllama_bf16_storage.stderr.log" \
  | tee "$BENCH_LOGS/tinyllama_bf16_storage.stdout.log"
```

```bash
cargo test --test tinyllama_gpu_matmul_smoke_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/tinyllama_gpu_matmul.stderr.log" \
  | tee "$BENCH_LOGS/tinyllama_gpu_matmul.stdout.log"
```

```bash
cargo test --test m4_7_4_e_tinyllama_disk_spill_smoke_test --release -- --ignored --nocapture \
  2> "$BENCH_LOGS/tinyllama_disk_spill.stderr.log" \
  | tee "$BENCH_LOGS/tinyllama_disk_spill.stdout.log"
```

```bash
cargo test --test bf16_storage_full_family_validation_test --release -- --ignored --nocapture --test-threads=1 \
  2> "$BENCH_LOGS/bf16_storage_full_family.stderr.log" \
  | tee "$BENCH_LOGS/bf16_storage_full_family.stdout.log"
```

```bash
cargo test --test m4_7_3_full_family_validation_test --release -- --ignored --nocapture --test-threads=1 \
  2> "$BENCH_LOGS/m47_full_family.stderr.log" \
  | tee "$BENCH_LOGS/m47_full_family.stdout.log"
```

```bash
cargo test --test m8_5_full_family_validation_test --release -- --ignored --nocapture --test-threads=1 \
  2> "$BENCH_LOGS/m85_full_family.stderr.log" \
  | tee "$BENCH_LOGS/m85_full_family.stdout.log"
```

```bash
cargo test --test tinyllama_f64_validation_test gguf -- --ignored --nocapture --test-threads=1 \
  2> "$BENCH_LOGS/m11d_gguf_diagnostics.stderr.log" \
  | tee "$BENCH_LOGS/m11d_gguf_diagnostics.stdout.log"
```

## 9. CLI Generation Probes Opcionales

Estos no reemplazan los tests. Sirven para traer tokens/s, planes VRAM/RAM/Disk
y errores de carga por CLI.

Preparar helper para correr cada caso y guardar stdout/stderr sin cortar toda
la bateria si un modelo falla:

```bash
export CLI_PROMPT="Tell me about the history of Rome"
export CLI_MAX_TOKENS=20
export ATENIA_BIN="$CARGO_TARGET_DIR/release/atenia"

run_cli_probe() {
  name="$1"
  model="$2"
  fast="${3:-0}"
  cache="$ATENIA_WORK_ROOT/runtime-cache/cli_$name"
  mkdir -p "$cache"

  echo "[CLI] $name model=$model fast=$fast"
  if [ "$fast" = "1" ]; then
    ATENIA_FAST_MODE=1 ATENIA_DISK_TIER_DIR="$cache" "$ATENIA_BIN" generate \
      --prompt "$CLI_PROMPT" \
      --model "$model" \
      --max-tokens "$CLI_MAX_TOKENS" \
      --output json \
      --no-progress \
      > "$BENCH_LOGS/cli_$name.json" \
      2> "$BENCH_LOGS/cli_$name.stderr.log"
  else
    ATENIA_DISK_TIER_DIR="$cache" "$ATENIA_BIN" generate \
      --prompt "$CLI_PROMPT" \
      --model "$model" \
      --max-tokens "$CLI_MAX_TOKENS" \
      --output json \
      --no-progress \
      > "$BENCH_LOGS/cli_$name.json" \
      2> "$BENCH_LOGS/cli_$name.stderr.log"
  fi

  code=$?
  echo "$code" > "$BENCH_LOGS/cli_$name.exitcode"
  if [ "$code" -ne 0 ]; then
    echo "[CLI FAIL] $name exit=$code"
    tail -n 5 "$BENCH_LOGS/cli_$name.stderr.log"
  else
    echo "[CLI OK] $name"
    jq -r '{tokens_generated,total_seconds,tokens_per_second,eos_reached}' "$BENCH_LOGS/cli_$name.json" 2>/dev/null || true
  fi
}
```

GGUF probes:

```bash
run_cli_probe gguf_tinyllama_q4 "$ATENIA_MODELS_ROOT/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF" 0
run_cli_probe gguf_tinyllama_q8 "$ATENIA_MODELS_ROOT/tinyllama-q8_0" 0
run_cli_probe gguf_llama32_1b_q4 "$ATENIA_MODELS_ROOT/Llama-3.2-1B-Instruct-Q4_K_M-GGUF" 0
run_cli_probe gguf_smollm2_q4 "$ATENIA_MODELS_ROOT/SmolLM2-1.7B-Instruct-GGUF" 0
run_cli_probe gguf_phi35_q4 "$ATENIA_MODELS_ROOT/Phi-3.5-mini-instruct-Q4_K_M-GGUF" 0
```

Safetensors probes, modo manifest/default:

```bash
run_cli_probe st_tinyllama_manifest "$ATENIA_MODELS_ROOT/tinyllama-1.1b" 0
run_cli_probe st_smollm2_manifest "$ATENIA_MODELS_ROOT/smollm2-1.7b-instruct" 0
run_cli_probe st_qwen25_15b_manifest "$ATENIA_MODELS_ROOT/qwen2.5-1.5b-instruct" 0
run_cli_probe st_llama32_1b_manifest "$ATENIA_MODELS_ROOT/llama-3.2-1b-instruct" 0
```

Safetensors probes, modo fast:

```bash
run_cli_probe st_tinyllama_fast "$ATENIA_MODELS_ROOT/tinyllama-1.1b" 1
run_cli_probe st_smollm2_fast "$ATENIA_MODELS_ROOT/smollm2-1.7b-instruct" 1
run_cli_probe st_qwen25_15b_fast "$ATENIA_MODELS_ROOT/qwen2.5-1.5b-instruct" 1
run_cli_probe st_llama32_1b_fast "$ATENIA_MODELS_ROOT/llama-3.2-1b-instruct" 1
```

Top-10 / modelos grandes si estan disponibles:

```bash
[ -d "$ATENIA_MODELS_ROOT/phi-3.5-mini-instruct" ] && run_cli_probe st_phi35_manifest "$ATENIA_MODELS_ROOT/phi-3.5-mini-instruct" 0
[ -d "$ATENIA_MODELS_ROOT/phi-3.5-mini-instruct" ] && run_cli_probe st_phi35_fast "$ATENIA_MODELS_ROOT/phi-3.5-mini-instruct" 1

[ -d "$ATENIA_MODELS_ROOT/mistral-7b-v0.3" ] && run_cli_probe st_mistral7b_manifest "$ATENIA_MODELS_ROOT/mistral-7b-v0.3" 0
[ -d "$ATENIA_MODELS_ROOT/mistral-7b-v0.3" ] && run_cli_probe st_mistral7b_fast "$ATENIA_MODELS_ROOT/mistral-7b-v0.3" 1

[ -d "$ATENIA_MODELS_ROOT/gemma-2-2b-it" ] && run_cli_probe st_gemma2_2b_manifest "$ATENIA_MODELS_ROOT/gemma-2-2b-it" 0
[ -d "$ATENIA_MODELS_ROOT/gemma-2-2b-it" ] && run_cli_probe st_gemma2_2b_fast "$ATENIA_MODELS_ROOT/gemma-2-2b-it" 1

[ -d "$ATENIA_MODELS_ROOT/falcon3-7b-instruct" ] && run_cli_probe st_falcon3_7b_manifest "$ATENIA_MODELS_ROOT/falcon3-7b-instruct" 0
[ -d "$ATENIA_MODELS_ROOT/falcon3-7b-instruct" ] && run_cli_probe st_falcon3_7b_fast "$ATENIA_MODELS_ROOT/falcon3-7b-instruct" 1

[ -d "$ATENIA_MODELS_ROOT/llama-2-13b-chat" ] && run_cli_probe st_llama2_13b_manifest "$ATENIA_MODELS_ROOT/llama-2-13b-chat" 0
[ -d "$ATENIA_MODELS_ROOT/llama-2-13b-chat" ] && run_cli_probe st_llama2_13b_fast "$ATENIA_MODELS_ROOT/llama-2-13b-chat" 1
```

Generar resumen rapido de CLI:

```bash
{
  echo "name,exit_code,tokens_generated,total_seconds,tokens_per_second,error"
  for codefile in "$BENCH_LOGS"/cli_*.exitcode; do
    name="$(basename "$codefile" .exitcode | sed 's/^cli_//')"
    code="$(cat "$codefile")"
    json="$BENCH_LOGS/cli_$name.json"
    stderr="$BENCH_LOGS/cli_$name.stderr.log"
    if [ "$code" = "0" ] && jq -e . "$json" >/dev/null 2>&1; then
      jq -r --arg name "$name" --arg code "$code" \
        '[ $name, $code, (.tokens_generated // ""), (.total_seconds // ""), (.tokens_per_second // ""), "" ] | @csv' "$json"
    else
      err="$(grep -E '^(error:|thread .+ panicked|.*failed)' "$stderr" | head -n1 | tr ',' ';')"
      printf '"%s","%s","","","","%s"\n' "$name" "$code" "$err"
    fi
  done
} > "$BENCH_LOGS/cli_summary.csv"
```

Para no cambiar el comportamiento respecto de una ejecucion manual normal, no
forzar `ATENIA_M8_BF16_KERNEL=1` salvo que se quiera investigar especificamente
ese camino.

## 10. Resumen Rapido

Al terminar, generar un indice simple:

```bash
{
  echo "# Manual RTX 3090 run"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Repo: $(git rev-parse HEAD)"
  echo "- Models root: $ATENIA_MODELS_ROOT"
  echo "- Work root: $ATENIA_WORK_ROOT"
  echo
  echo "## Logs"
  find "$BENCH_LOGS" -maxdepth 1 -type f | sort
} > "$BENCH_LOGS/README.md"
```

Traer de vuelta completa esta carpeta:

```bash
echo "$BENCH_LOGS"
```

## 11. Que Mirar Primero

1. `server_diagnostics.txt`
2. `lib_all.stdout.log` / `lib_all.stderr.log`
3. Smokes safetensors end-to-end
4. `m11d_gguf_diagnostics.stdout.log`
5. CLI `.stderr.log` para planes:

```text
VRAM: N tensors (X GiB)
RAM:  N tensors (Y GiB)
Disk: N tensors (Z GiB)
```

Si algo falla, no borrar los logs: esos fallos son parte de la evidencia que
queremos traer para Atenia.
